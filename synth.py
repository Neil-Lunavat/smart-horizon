#!/usr/bin/env python3
"""Physics-based synthetic GNSS error generator.

WHY THIS FILE EXISTS
The delivered SH-DST-03 files give us three windows but they're not enough
to prove a model's usefulness or learnings.

So we generate our own from the orbital mechanics in `PHYSICS.md`, sampled
at the *same irregular timestamps and gaps* as the real files, with noise 
of the same size. We know the true answer by construction to ask the question:
    **given seven days of this, does the model recover day eight?**

WHAT GOES INTO THE SIGNAL, AND WHY EACH TERM IS THERE

  1. static bias        Every satellite sits on a fixed offset (reference-frame
                        and hardware constants). Large, and completely harmless,
                        it cancels in any differenced treatment.
  2. orbital harmonics  The physics. Orbit error is quasi-periodic at the orbital
                        period. In Earth-fixed (ECEF) coordinates the axes split:
                        z stays at the orbital period T, while x and y are mixed
                        with Earth's rotation and appear at the two BEAT periods
                        1/(1/T +- 1/T_sid). For a geostationary satellite T =
                        T_sid, so everything collapses onto the sidereal day.
  3. slow drift         Orbit solutions age between ground-segment uploads;
                        modelled as a low-order polynomial over the window.
  4. white noise        Measurement and rounding noise.
  5. clock              Deliberately NOT periodic. An atomic clock's error is a
                        random walk (thermal + frequency drift) punctuated by
                        step resets when the ground segment uploads new
                        coefficients. There is no 24-hour rhythm to find, which
                        is why the model is only allowed level estimators here.

  6. the artefact       Optional. The alternating sign-flip pattern found in the
                        delivered GEO file, injected on demand, so we can prove
                        what it does to a model by injecting it into data that
                        was clean a moment before.

Usage:
    from synth import make_window, SAMPLERS
    t, xyz, truth = make_window("MEO", seed=0)

    The orbital periods are published constants (they define the constellations), 
    and T_sid is the length of a sidereal day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["x_error", "y_error", "z_error", "satclockerror"]

#------------------------------------------------------------ constants
T_SID = 86164.0905                # sidereal day (s), Earth's true rotation period
DAY = 86400.0

# Published orbital periods. These are what DEFINE the constellations; 
ORBITAL_PERIOD = {
    "GPS":     43082.0,           # 2 revs per sidereal day
    "GALILEO": 50567.0,           # 17 revs per 10 sidereal days
    "BEIDOU":  46393.0,           # 13 revs per 7 sidereal days
}


def beat_periods(T, t_sid=T_SID):
    """The two periods x and y appear at, in Earth-fixed coordinates.

    An orbit error that is fixed in the orbital plane rotates with the satellite
    at period T. The ECEF frame itself rotates with the Earth at T_sid. Seen from
    ECEF, the x and y components are the product of the two rotations, which by
    the product-to-sum identity is a sum of two tones at the SUM and DIFFERENCE
    of the frequencies:

        f_beat = 1/T +- 1/T_sid       ->      P = 1 / |1/T -+ 1/T_sid|

    The z axis is the rotation axis, so it is untouched and stays at T.
    Full derivation in PHYSICS.md.
    """
    out = []
    for s in (+1.0, -1.0):
        f = 1.0 / T + s / t_sid
        if abs(f) > 1e-12:
            P = abs(1.0 / f)
            if 1e3 < P < 1e7:
                out.append(P)
    return out


#-------------------------------------------------------- the generator
def _harmonic(t, periods, amps, phases):
    y = np.zeros_like(t)
    for P, a, ph in zip(periods, amps, phases):
        y = y + a * np.sin(2 * np.pi * t / P + ph)
    return y


def _clock(t, rng, sd_walk, n_reset, reset_size):
    """Random walk plus step resets, NOT periodic, by construction.

    An atomic clock's error accumulates as a random walk (temperature, ageing,
    frequency noise). Periodically the ground segment uploads fresh coefficients,
    which steps the error. Neither component has a daily rhythm, which is exactly
    why day-8 clock level is not predictable from the previous week.
    """
    order = np.argsort(t)
    ts = t[order]
    steps = rng.normal(0.0, sd_walk, len(ts))
    dt = np.diff(ts, prepend=ts[0])
    walk = np.cumsum(steps * np.sqrt(np.maximum(dt, 0.0) / 3600.0))
    if n_reset > 0:
        for tr in rng.uniform(ts.min(), ts.max(), n_reset):
            walk = walk + np.where(ts >= tr, rng.normal(0.0, reset_size), 0.0)
    out = np.empty_like(walk)
    out[order] = walk
    return out


def true_signal(t, orbit="MEO", seed=0, constellation=None,
                amp_xy=0.45, amp_z=0.40, drift=0.25,
                clock_walk=0.05, clock_resets=3, clock_reset_size=0.15):
    """The underlying continuous error, evaluated at times `t` (seconds).

    This is GROUND TRUTH: we can evaluate it at any instant, including the day-8
    timestamps the model never sees. Returns an (n, 4) array.
    """
    rng = np.random.default_rng(seed)
    geo = str(orbit).upper().startswith("G")

    if geo:
        T = T_SID
        per_xy = [T_SID, T_SID / 2.0]
        per_z = [T_SID, T_SID / 2.0]
    else:
        name = constellation or rng.choice(list(ORBITAL_PERIOD))
        T = ORBITAL_PERIOD[name]
        per_xy = beat_periods(T)              # the two beat periods
        per_z = [T, T / 2.0]                  # z stays at the orbital period

    out = np.zeros((len(t), 4))
    tn = (t - t.min()) / DAY                  # days, for the drift polynomial

    for j, per in enumerate([per_xy, per_xy, per_z]):
        amp = amp_xy if j < 2 else amp_z
        a = amp * rng.uniform(0.6, 1.4, len(per))
        ph = rng.uniform(0, 2 * np.pi, len(per))
        bias = rng.normal(0.0, 1.0)                       # static per-satellite offset
        dr = drift * rng.normal() * (tn / max(tn.max(), 1e-9))
        out[:, j] = bias + _harmonic(t, per, a, ph) + dr

    out[:, 3] = (rng.normal(0.0, 0.3)
                 + _clock(t, rng, clock_walk, clock_resets, clock_reset_size))
    return out


#----------------------------------------------------- sampling patterns
def sih_sampling(pattern="MEO-2", days=8, seed=0):
    """Timestamps mimicking the delivered files: irregular, gappy, ~13-32% covered.

    The delivered files are NOT a uniform grid. Reproducing that matters, a
    model that only works on clean grids would be useless here.
    """
    rng = np.random.default_rng(seed)
    cad = {"GEO": 900.0, "MEO-1": 3600.0, "MEO-2": 600.0}[pattern]
    cover = {"GEO": 0.32, "MEO-1": 0.135, "MEO-2": 0.18}[pattern]

    grid = np.arange(0, days * DAY, cad)
    keep = np.zeros(len(grid), bool)
    # samples arrive in bursts separated by long holes, as in the real files
    n_burst = max(3, int(days * 3))
    burst_len = max(3, int(cover * len(grid) / n_burst))
    for st in rng.choice(len(grid) - burst_len, n_burst, replace=False):
        keep[st:st + burst_len] = True
    t = grid[keep]
    t = t + rng.uniform(-cad * 0.15, cad * 0.15, len(t))     # jitter off-grid
    return np.sort(t)


SAMPLERS = ["GEO", "MEO-1", "MEO-2"]


#------------------------------------------------------- the sign artefact
def inject_artifact(V, frac=0.35, seed=0):
    """Inject the alternating sign-flip found in the delivered GEO file.

    In `DATA_GEO_Test.csv` the 24 largest excursions alternate sign +-+-+-...
    24 times out of 24. We reproduce that: pick a subset of rows, blow them up,
    and force their signs to alternate, ALL FOUR COLUMNS TOGETHER, which is the
    physically impossible part (position and clock come from unrelated
    mechanisms and cannot move in lockstep).

    Injecting it into clean data lets us measure what it does to a model, which
    is the closest we can get to proving what happened to their file.
    """
    rng = np.random.default_rng(seed)
    V = V.copy()
    n = len(V)
    idx = np.sort(rng.choice(n, max(2, int(frac * n)), replace=False))
    scale = 8.0 + 6.0 * rng.random(len(idx))
    for k, i in enumerate(idx):
        V[i] = np.abs(V[i]) * scale[k] * (1.0 if k % 2 == 0 else -1.0)
    return V, idx


#----------------------------------------------------------- convenience
def make_window(orbit="MEO", pattern=None, seed=0, artifact=False,
                noise=0.05, query_step=900.0, **kw):
    """One complete 7-day train window + day-8 truth, at a known answer.

    Returns (train_df, query_times_s, truth_day8, meta).
      train_df   the 5 columns a competitor is given, days 0-7
      query_s    day-8 query times (seconds from window start)
      truth      the TRUE error at those query times, what we grade against
    """
    pattern = pattern or ("GEO" if str(orbit).upper().startswith("G") else "MEO-2")
    t_all = sih_sampling(pattern, days=8, seed=seed)
    t_tr = t_all[t_all < 7 * DAY]
    if len(t_tr) < 30:                       # guarantee a usable window
        t_tr = np.sort(np.concatenate([t_tr, sih_sampling(pattern, 7, seed + 500)]))
        t_tr = t_tr[t_tr < 7 * DAY]

    t_q = np.arange(7 * DAY, 8 * DAY, query_step)

    rng = np.random.default_rng(seed + 991)
    V_tr = true_signal(t_tr, orbit, seed=seed, **kw)
    V_q = true_signal(t_q, orbit, seed=seed, **kw)

    V_tr_obs = V_tr + rng.normal(0.0, noise, V_tr.shape)
    meta = {"artifact_idx": None}
    if artifact:
        V_tr_obs, idx = inject_artifact(V_tr_obs, seed=seed)
        meta["artifact_idx"] = idx

    base = pd.Timestamp("2025-09-01")
    tr = pd.DataFrame({"utc_time": base + pd.to_timedelta(t_tr, "s")})
    for j, c in enumerate(COLS):
        tr[c] = V_tr_obs[:, j]
    meta["query_times"] = base + pd.to_timedelta(t_q, "s")
    meta["t_train_s"] = t_tr
    meta["t_query_s"] = t_q
    meta["clean_train"] = V_tr
    return tr, meta["query_times"], V_q, meta


if __name__ == "__main__":
    print("orbital periods and their ECEF beat periods:\n")
    print(f"{'constellation':14s} {'T (s)':>9s} {'T (h)':>7s}   beat periods (h)")
    for k, T in ORBITAL_PERIOD.items():
        b = beat_periods(T)
        print(f"{k:14s} {T:9.0f} {T/3600:7.2f}   " +
              ", ".join(f"{x/3600:.2f}" for x in b))
    print(f"\n{'GEO/GSO':14s} {T_SID:9.1f} {T_SID/3600:7.2f}   "
          f"(T = T_sid, so the beats collapse onto the sidereal day itself)")
    print("\nsanity: a 7-day window contains")
    print(f"  {7*DAY/T_SID:.2f} sidereal days   (GEO: plenty of cycles to fit)")
    for k, T in ORBITAL_PERIOD.items():
        print(f"  {7*DAY/T:.2f} orbits of {k}")
    tr, q, truth, meta = make_window("MEO", seed=1)
    print(f"\nexample MEO window: {len(tr)} training rows, {len(q)} day-8 query times")
    print(tr.head(3).to_string(index=False))
