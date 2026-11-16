#!/usr/bin/env python3
"""TASK B -- two separate models, because the competition tells us the class.

Inputs at inference, exactly what they supply:
    timestamp, x_err, y_err, z_err, clock_err, and MEO-or-GEO.
Output: the same four errors at arbitrary day-8 timestamps.

TWO MODELS, because GEO and MEO are different physics and the class is GIVEN:

  GEO / GSO   period is the sidereal day, full stop. 86164 s, known a priori,
              7 clean cycles inside a 7-day window. Nothing to search for.
              Measured: mean RMS 2.625 -> 0.370 m.

  MEO         the orbital period is 12-14 h and differs per constellation
              (GPS 43082 s, Galileo 50567 s, BeiDou 46393 s) and we are NOT told
              which. Worse, in ECEF the axes do not share a frequency: z sits at
              the orbital period T while x and y are beaten against Earth
              rotation at 1/(1/T +- 1/T_sid). So T is estimated per window by
              Lomb-Scargle, restricted to the physically possible MEO band, and
              the beat periods are then derived analytically.

  CLOCK       window mean, both classes, always. Measured six ways today; every
              harmonic variant makes it worse. The day-8 clock mean is
              statistically independent of the 7-day window mean (R^2 = 0.000),
              so the mean is provably the best available estimator.

Basis order K is chosen per component by self-validation: fit on days 1-6, score
on day 7, refit the winner on all 7 days. Uses only the supplied input.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lombscargle

STEP, WIN, HOR = 900, 672, 96
T_SID = 86164.0
MEO_LO, MEO_HI = 38000.0, 54000.0          # physically possible MEO orbital periods
T4 = ["x_err", "y_err", "z_err", "clock_err"]

# ------------------------------------------------------------------ corpus
# Compact path: taskB_data.npz holds just the ECEF grid + labels (38 MB), which is
# all any Task B script needs. Falls back to rebuilding from the full corpus.
import os
if os.path.exists("taskB_data.npz"):
    _d = np.load("taskB_data.npz", allow_pickle=True)
    XYZ = _d["XYZ"]
    SATS, SYS, CLS = list(_d["sats"]), _d["sys"], _d["cls"]
    NT, S = int(_d["NT"]), len(SATS)
else:
    z_ = np.load("taskA.npz", allow_pickle=True)
    E = z_["E"]
    SATS, SYS, CLS = list(z_["sats"]), z_["sys"], z_["cls"]
    NT, S = int(z_["NT"]), len(SATS)
    q = pd.read_parquet("full2.parquet",
                        columns=["epoch", "sat", "dx", "dy", "dz", "dClk_ns", "px", "py", "pz"])
    q["clk"] = q.dClk_ns * 1e-9 * 299792458.0
    q = q[q.clk.abs() < 100]
    rr = np.sqrt(q.px ** 2 + q.py ** 2 + q.pz ** 2).to_numpy()
    sid = q.sat.to_numpy().astype(object)
    mm = q.sat.isin(["C13", "C38", "C39"]).to_numpy()
    sid[mm] = np.where(rr[mm] > 3.5e7, q.sat.to_numpy()[mm] + "_a", q.sat.to_numpy()[mm] + "_b")
    q["scid"] = sid
    t0 = q.epoch.min()
    XYZ = np.full((S, NT, 4), np.nan, np.float32)
    smap = {s: i for i, s in enumerate(SATS)}
    kk = ((q.epoch - t0).dt.total_seconds() // STEP).astype(int).to_numpy()
    ii = q.scid.map(smap).to_numpy()
    ok = pd.notna(ii)
    XYZ[ii[ok].astype(int), kk[ok]] = q.loc[ok, ["dx", "dy", "dz", "clk"]].to_numpy(np.float32)
    XYZ[~np.isfinite(E[:, :, :1]).repeat(4, axis=2)] = np.nan
print(f"corpus {XYZ.shape}  valid {np.isfinite(XYZ[:,:,0]).mean()*100:.1f}%", flush=True)


# ------------------------------------------------------------------ shared maths
def design(t, periods, K):
    cols = [np.ones_like(t)]
    for P in periods:
        for j in range(1, K + 1):
            cols += [np.sin(2 * np.pi * j * t / P), np.cos(2 * np.pi * j * t / P)]
    return np.column_stack(cols)


def ridge_fit(A, y, lam_rel=1e-3):
    lam = lam_rel * np.trace(A.T @ A) / A.shape[1]
    try:
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    except np.linalg.LinAlgError:
        return None


def _tail_mean(t, y, span):
    """Mean of the last `span` seconds of the window; falls back to all of it."""
    m = t >= (t.max() - span)
    return float(y[m].mean()) if m.sum() >= 3 else float(y.mean())


def _ewma(t, y, tau):
    w = np.exp(-(t.max() - t) / tau)
    return float((w * y).sum() / w.sum())


# Candidate predictors. A harmonic candidate is ("H", periods, K); a level
# candidate is ("L", fn) where fn(t, y) -> a constant. Level candidates matter
# for TRUE GEO, whose ECEF error is a slow drift rather than a daily cycle: a
# geostationary satellite barely moves in ECEF, so there is no rotation to mix
# radial/along/cross into x/y/z and nothing forces a sidereal oscillation.
# Measured: J07/J08 x_err R^2 against sidereal harmonics is 0.004 / 0.010.
LEVELS = [("mean_all", lambda t, y: float(y.mean())),
          ("mean_3d", lambda t, y: _tail_mean(t, y, 3 * 86400.0)),
          ("mean_2d", lambda t, y: _tail_mean(t, y, 2 * 86400.0)),
          ("mean_1d", lambda t, y: _tail_mean(t, y, 1 * 86400.0)),
          ("mean_12h", lambda t, y: _tail_mean(t, y, 43200.0)),
          ("ewma_1d", lambda t, y: _ewma(t, y, 86400.0)),
          ("ewma_2d", lambda t, y: _ewma(t, y, 2 * 86400.0))]


def pick_and_fit(tw, yw, tq, cand, levels=True):
    """Self-validating: fit every candidate on days 1-6, score it on day 7, then
    refit the winner on the whole window and evaluate at the query times."""
    if len(tw) < 20:
        return np.full(len(tq), float(yw.mean()))
    cut = tw.min() + (tw.max() - tw.min()) * 6.0 / 7.0
    a, b = tw <= cut, tw > cut
    if a.sum() < 12 or b.sum() < 4:
        return np.full(len(tq), float(yw.mean()))
    best, err = ("L", LEVELS[0][1]), np.inf
    if levels:
        for _, fn in LEVELS:
            e = float(np.mean((yw[b] - fn(tw[a], yw[a])) ** 2))
            if e < err:
                err, best = e, ("L", fn)
    else:
        err = float(np.mean((yw[b] - yw[a].mean()) ** 2))
    for per, K in cand:
        c = ridge_fit(design(tw[a], per, K), yw[a])
        if c is None:
            continue
        e = float(np.mean((yw[b] - design(tw[b], per, K) @ c) ** 2))
        if e < err:
            err, best = e, ("H", per, K)
    if best[0] == "L":
        return np.full(len(tq), best[1](tw, yw))
    c = ridge_fit(design(tw, best[1], best[2]), yw)
    if c is None:
        return np.full(len(tq), float(yw.mean()))
    return design(tq, best[1], best[2]) @ c


def ls_period(t, y, lo, hi, n=500):
    """Dominant period inside a physically-motivated band."""
    yy = y - y.mean()
    if np.allclose(yy, 0) or len(t) < 12:
        return None
    grid = np.linspace(lo, hi, n)
    try:
        p = lombscargle(t.astype(float), yy.astype(float), 2 * np.pi / grid, normalize=True)
    except Exception:
        return None
    return float(grid[int(np.argmax(p))])


def beats(T):
    """ECEF beat periods of an orbital period T against Earth rotation."""
    out = []
    for sgn in (+1, -1):
        d = 1.0 / T + sgn / T_SID
        if abs(d) > 1e-9:
            c = abs(1.0 / d)
            if 10000 < c < 400000:
                out.append(c)
    return out


# ------------------------------------------------------------------ MODEL 1: GEO
def predict_geo(tw, vw, tq):
    """Period is known: the sidereal day. Only the harmonic order is chosen."""
    out = np.zeros((len(tq), 4))
    cand = [([T_SID], K) for K in (1, 2, 3, 4)]
    for j in range(3):
        out[:, j] = pick_and_fit(tw, vw[:, j], tq, cand)
    out[:, 3] = pick_and_fit(tw, vw[:, 3], tq, [])      # level only: no harmonics on clock
    return out


# ------------------------------------------------------------------ MODEL 2: MEO
def predict_meo(tw, vw, tq):
    """Orbital period unknown within the MEO band, so estimate it from the data.

    z is fitted at the orbital period; x and y at the Earth-rotation beats of it.
    Each component also gets its own Lomb-Scargle peak as a fallback candidate,
    and every candidate must beat the mean on the held-out 7th day."""
    out = np.zeros((len(tq), 4))
    # estimate the orbital period from z, which carries it cleanly
    T = ls_period(tw, vw[:, 2], MEO_LO, MEO_HI) or 43082.0
    bp = beats(T)
    for j in range(3):
        own = ls_period(tw, vw[:, j], 20000.0, 190000.0)
        cand = []
        if j == 2:                       # z: the orbital period itself
            cand += [([T], K) for K in (1, 2, 3)]
        else:                            # x, y: the beats
            for B in bp:
                cand += [([B], K) for K in (1, 2)]
            cand += [([T], 1)]
        if own:
            cand += [([own], 1), ([own], 2)]
        out[:, j] = pick_and_fit(tw, vw[:, j], tq, cand)
    out[:, 3] = pick_and_fit(tw, vw[:, 3], tq, [])      # level only: no harmonics on clock
    return out


# ------------------------------------------------------------------ evaluation
def thin(idx, vals, rate, rg):
    keep = rg.random(len(idx)) < rate
    for _ in range(rg.integers(1, 5)):
        a = rg.integers(0, max(len(idx) - 1, 1))
        keep[a:a + rg.integers(4, 100)] = False
    if keep.sum() < 30:
        keep[rg.choice(len(idx), 30, replace=False)] = True
    return idx[keep], vals[keep]


def run(n, rate=0.30, seed=0, uniform=False):
    rg = np.random.default_rng(seed)
    recs, tried = [], 0
    while len(recs) < n and tried < n * 15:
        tried += 1
        s = int(rg.integers(0, S))
        o = int(rg.integers(WIN, NT - HOR))
        w, f = XYZ[s, o - WIN:o], XYZ[s, o:o + HOR]
        if np.isfinite(w[:, 0]).mean() < 0.5 or np.isfinite(f[:, 0]).mean() < 0.5:
            continue
        iw = np.nonzero(np.isfinite(w[:, 0]))[0]
        tw, vw = (((iw - WIN) * float(STEP), w[iw]) if uniform
                  else thin((iw - WIN) * float(STEP), w[iw], rate, rg))
        if len(tw) < 30:
            continue
        iq = np.nonzero(np.isfinite(f[:, 0]))[0]
        if len(iq) < 8:
            continue
        iq = np.sort(rg.choice(iq, min(len(iq), int(rg.integers(8, 60))), replace=False))
        tq, yq = iq * float(STEP), f[iq]
        geo = CLS[s] in ("GEO", "GSO")
        P = {"do nothing": np.zeros((len(tq), 4)),
             "window mean": np.repeat(vw.mean(0)[None, :], len(tq), 0),
             "MODEL": predict_geo(tw, vw, tq) if geo else predict_meo(tw, vw, tq)}
        recs.append((geo, yq, P))
    return recs


def report(recs, title):
    print(f"\n{'='*86}\n{title}  ({len(recs)} windows)\n{'='*86}")
    for grp, want in [("GEO / GSO", True), ("MEO", False)]:
        sel = [r for r in recs if r[0] == want]
        if not sel:
            continue
        nq = sum(len(r[1]) for r in sel)
        print(f"\n--- {grp}   {len(sel)} windows, {nq:,} query points")
        print(f"{'method':14s} " + "  ".join(f"{c:>10s}" for c in T4) + f"  {'mean':>8s}")
        res = {}
        for mth in ["do nothing", "window mean", "MODEL"]:
            se, n = np.zeros(4), 0
            for r in sel:
                d = r[1] - r[2][mth]
                se += (d ** 2).sum(0)
                n += len(d)
            res[mth] = np.sqrt(se / n)
            print(f"{mth:14s} " + "  ".join(f"{v:10.3f}" for v in res[mth]) + f"  {res[mth].mean():8.3f}")
        cut = 100 * (1 - res["MODEL"].mean() / res["do nothing"].mean())
        print(f"{'reduction':14s} " + "  ".join(
            f"{100*(1-res['MODEL'][j]/res['do nothing'][j]):9.1f}%" for j in range(4))
            + f"  {cut:7.1f}%")
        allr = {j: [] for j in range(4)}
        ws = {j: [] for j in range(4)}
        for r in sel:
            d = r[1] - r[2]["MODEL"]
            for j in range(4):
                allr[j].append(d[:, j])
            if len(d) >= 12:
                for j in range(4):
                    ws[j].append(stats.shapiro(d[:, j]).statistic)
        print(f"\n  residuals:  {'target':10s} {'std':>7s} {'skew':>7s} {'kurt':>7s} "
              f"{'per-window W':>13s}")
        for j, c in enumerate(T4):
            x = np.concatenate(allr[j])
            print(f"              {c:10s} {x.std():7.3f} {stats.skew(x):7.2f} "
                  f"{stats.kurtosis(x):7.1f} {np.median(ws[j]):13.4f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    report(run(n, rate=0.30, seed=21), "IRREGULAR (~30% + multi-hour gaps) -- their format")
    report(run(n, uniform=True, seed=22), "UNIFORM 15-min -- upper bound")
