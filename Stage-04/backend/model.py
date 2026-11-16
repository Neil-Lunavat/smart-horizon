#!/usr/bin/env python3
"""SMART-HORIZON -- a physics-constrained regression model for GNSS error.

    from model import predict
    out = predict(train_df, query_times, orbit="MEO")

  train_df       columns utc_time, x_error, y_error, z_error, satclockerror
                 (7 days, any sampling, gaps fine, duplicates fine)
  query_times    timestamps to predict, any spacing
  orbit          "MEO" or "GEO"
  returns        predictions plus a 1-sigma uncertainty per column

WHAT KIND OF MODEL THIS IS
A regularised (ridge) regression onto a basis of orbital harmonics, with the
basis itself chosen by held-out validation inside each window. Three layers:

  1. HYPOTHESIS SPACE -- fixed by orbital mechanics, not by fitting. Which
     frequencies are physically admissible for each orbit class (see PHYSICS.md).
  2. MODEL SELECTION -- for every window the candidate bases and level
     estimators compete on a held-out day; the winner is kept.
  3. COEFFICIENTS -- fitted by ridge regression, refit on the full window once
     the basis is chosen.

Layers 2 and 3 run at call time on the caller's own data. That is deliberate: it
is what lets the model adapt to a satellite it has never encountered, and it is
why performance does not depend on the model having seen a similar training
period. The design was fixed and validated against synthetic data generated from
orbital mechanics (`synth.py`) before ever being run on the delivered files.

THE HYPOTHESIS SPACE, AND WHERE IT COMES FROM
  GEO/GSO   a geosynchronous ground track repeats at the sidereal day (86164 s),
            so a 7-day window holds ~7 clean cycles and the period needs no
            search -- only the harmonic order K is selected
  true GEO  a geostationary satellite barely moves in ECEF, so nothing forces a
            daily oscillation and the error can be a slow DRIFT instead; level
            estimators therefore stay in the GEO candidate set
  MEO       z sits at the orbital period T (12-14 h, and we are NOT told which
            constellation), while x and y are mixed with Earth rotation and
            appear at the beat periods 1/(1/T +- 1/T_sid). T is estimated per
            window by Lomb-Scargle; the beats are then derived analytically
  clock     an atomic clock's error is a random walk with step resets, not a
            periodic signal -- so harmonics are barred on the clock channel and
            only level estimators compete there

MEASURED PERFORMANCE  (mean RMS over x/y/z/clock, metres)
                       do nothing   window mean   this model
  synthetic GEO/GSO         --            --          see notebook
  synthetic MEO             --            --          see notebook
  PS-08 MEO-1              0.444         0.328        0.190
  PS-08 MEO-2              0.164         0.144        0.133
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T_SID = 86164.0                      # sidereal day, seconds
MEO_LO, MEO_HI = 38000.0, 54000.0    # possible MEO orbital periods
COLS = ["x_error", "y_error", "z_error", "satclockerror"]

# The one empirical constant in this file. The held-out day both SELECTS the
# winning basis and SCORES it, so its RMS is optimistic as a day-8 forecast
# error. Measured across many synthetic windows (`synth.py`), actual/reported ran
# 1.15 (median) to 1.30 (variance-pooled, which a few blown-up windows dominate);
# 1.25 is the compromise. It scales the reported uncertainty ONLY -- predictions
# are untouched by it.
SIGMA_CAL = 1.25


# ----------------------------------------------------------------- internals
def _design(t, periods, K):
    cols = [np.ones_like(t)]
    for P in periods:
        for j in range(1, K + 1):
            cols += [np.sin(2 * np.pi * j * t / P), np.cos(2 * np.pi * j * t / P)]
    return np.column_stack(cols)


def _ridge(A, y, lam_rel=1e-3):
    lam = lam_rel * np.trace(A.T @ A) / A.shape[1]
    try:
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    except np.linalg.LinAlgError:
        return None


def _tail(t, y, span):
    m = t >= (t.max() - span)
    return float(y[m].mean()) if m.sum() >= 3 else float(y.mean())


def _ewma(t, y, tau):
    w = np.exp(-(t.max() - t) / tau)
    return float((w * y).sum() / w.sum())


_LEVELS = [lambda t, y: float(y.mean()),
           lambda t, y: _tail(t, y, 3 * 86400.0),
           lambda t, y: _tail(t, y, 2 * 86400.0),
           lambda t, y: _tail(t, y, 86400.0),
           lambda t, y: _tail(t, y, 43200.0),
           lambda t, y: _ewma(t, y, 86400.0),
           lambda t, y: _ewma(t, y, 2 * 86400.0)]


def _lomb(t, y, lo, hi, n=500):
    """Dominant period in [lo, hi]. Plain Lomb-Scargle so scipy.signal is optional."""
    yy = y - y.mean()
    if len(t) < 12 or np.allclose(yy, 0):
        return None
    grid = np.linspace(lo, hi, n)
    best, bp = -1.0, None
    for P in grid:
        w = 2 * np.pi / P
        s2, c2 = np.sin(2 * w * t).sum(), np.cos(2 * w * t).sum()
        tau = 0.5 * np.arctan2(s2, c2) / w
        ct, st = np.cos(w * (t - tau)), np.sin(w * (t - tau))
        dc, ds = (ct ** 2).sum(), (st ** 2).sum()
        if dc < 1e-9 or ds < 1e-9:
            continue
        p = 0.5 * ((yy * ct).sum() ** 2 / dc + (yy * st).sum() ** 2 / ds)
        if p > best:
            best, bp = p, P
    return bp


def _beats(T):
    out = []
    for sgn in (+1, -1):
        d = 1.0 / T + sgn / T_SID
        if abs(d) > 1e-9 and 10000 < abs(1.0 / d) < 400000:
            out.append(abs(1.0 / d))
    return out


def _pick(tw, yw, tq, cand, levels=True):
    """Fit every candidate on days 1-6, score on day 7, refit the winner on all 7.

    The held-out day is the only model-selection signal used, and it comes from
    the caller's own data, so the predictor never needs anything external.

    `levels=False` bars the flat estimators from competing. That is used for GEO
    x/y/z, where the sidereal period is known exactly and a noisy held-out day
    was occasionally picking a flat line where a harmonic would have
    extrapolated correctly. Kept available but NOT used for GEO -- see the note
    in predict(): barring the levels lost more on off-grid query times than it
    won on-grid."""
    if len(tw) < 20:
        return np.full(len(tq), float(yw.mean())), float(yw.std())
    cut = tw.min() + (tw.max() - tw.min()) * 6.0 / 7.0
    a, b = tw <= cut, tw > cut
    if a.sum() < 12 or b.sum() < 4:
        return np.full(len(tq), float(yw.mean())), float(yw.std())
    best, err = ("L", _LEVELS[0]), np.inf
    if levels:
        for fn in _LEVELS:
            e = float(np.mean((yw[b] - fn(tw[a], yw[a])) ** 2))
            if e < err:
                err, best = e, ("L", fn)
    for per, K in cand:
        c = _ridge(_design(tw[a], per, K), yw[a])
        if c is None:
            continue
        e = float(np.mean((yw[b] - _design(tw[b], per, K) @ c) ** 2))
        if e < err:
            err, best = e, ("H", per, K)
    if not np.isfinite(err):                       # no candidate fitted; fall back
        err = float(np.mean((yw[b] - _LEVELS[0](tw[a], yw[a])) ** 2))
    sigma = float(np.sqrt(max(err, 1e-12)))        # held-out RMS = honest 1-sigma
    if best[0] == "L":
        return np.full(len(tq), best[1](tw, yw)), sigma
    c = _ridge(_design(tw, best[1], best[2]), yw)
    if c is None:
        return np.full(len(tq), float(yw.mean())), sigma
    p = _design(tq, best[1], best[2]) @ c
    # Guard against a harmonic fit diverging outside the window it was fitted on.
    # Any extrapolator can run away once past its data; barring the flat
    # estimators (GEO x/y/z) removes the implicit safety net, so make it explicit.
    lo, hi = float(yw.min()), float(yw.max())
    pad = 2.0 * max(hi - lo, 1e-9)
    return np.clip(p, lo - pad, hi + pad), sigma


# ----------------------------------------------------------------- public API
def predict(train, query_times, orbit="MEO", time_col="utc_time"):
    """Predict day-8 errors at arbitrary timestamps from a 7-day window.

    Duplicated rows are dropped, and the input need not be sorted or regular.
    Returns predictions plus a 1-sigma uncertainty estimated by holding out the
    window's own final day.
    """
    df = train.copy()
    df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
    ren = {"clock_error": "satclockerror", "clock": "satclockerror",
           "x_err": "x_error", "y_err": "y_error", "z_err": "z_error"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.drop_duplicates().sort_values(time_col).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"need at least 20 unique rows, got {len(df)}")

    qt = pd.to_datetime(pd.Series(list(query_times)))
    t0 = df[time_col].max()
    tw = (df[time_col] - t0).dt.total_seconds().to_numpy(float)
    tq = (qt - t0).dt.total_seconds().to_numpy(float)
    V = df[COLS].to_numpy(float)

    geo = str(orbit).upper().startswith("G")
    out = np.zeros((len(tq), 4))
    sig = np.zeros(4)
    if geo:
        # Level estimators are KEPT for GEO. Benchmarking said a fixed harmonic
        # basis with no selection scored slightly better on average, but that gain
        # did not replicate on an independent sample and it reversed badly on
        # off-grid query times (all of the loss in x). The levels are not dead
        # weight: a true geostationary satellite barely moves in ECEF, so its error
        # can be a slow drift with no daily oscillation at all, and on those
        # windows a flat estimator is simply the correct answer.
        cand = [([T_SID], K) for K in (1, 2, 3, 4)]
        for j in range(3):
            out[:, j], sig[j] = _pick(tw, V[:, j], tq, cand)
    else:
        T = _lomb(tw, V[:, 2], MEO_LO, MEO_HI) or 43082.0
        bp = _beats(T)
        for j in range(3):
            cand = [([T], K) for K in (1, 2, 3)] if j == 2 else \
                   [([B], K) for B in bp for K in (1, 2)] + [([T], 1)]
            own = _lomb(tw, V[:, j], 20000.0, 190000.0, n=300)
            if own:
                cand += [([own], 1), ([own], 2)]
            out[:, j], sig[j] = _pick(tw, V[:, j], tq, cand)
    out[:, 3], sig[3] = _pick(tw, V[:, 3], tq, [])      # clock: level estimators only

    res = pd.DataFrame({time_col: qt.to_numpy()})
    for j, c in enumerate(COLS):
        res[c] = out[:, j]
    for j, c in enumerate(COLS):
        res[c + "_sigma"] = sig[j] * SIGMA_CAL
    return res


def predict_csv(train_csv, query_csv, orbit="MEO", out_csv=None):
    """Same, straight from files. `query_csv` needs only a timestamp column."""
    tr = pd.read_csv(train_csv)
    qs = pd.read_csv(query_csv)
    tcol = [c for c in qs.columns if "time" in str(c).lower()][0]
    res = predict(tr, qs[tcol], orbit=orbit)
    if out_csv:
        res.to_csv(out_csv, index=False)
    return res


if __name__ == "__main__":
    # Self-test on synthetic data with the documented structure.
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-01-01", periods=672, freq="900s")
    keep = rng.random(672) < 0.3
    tt = np.arange(672) * 900.0
    sig_ = (1.4 + 0.6 * np.sin(2 * np.pi * tt / T_SID)
            + 0.25 * np.cos(4 * np.pi * tt / T_SID))
    tr = pd.DataFrame({"utc_time": t[keep],
                       "x_error": (sig_ + rng.normal(0, .05, 672))[keep],
                       "y_error": (0.5 * sig_ + rng.normal(0, .05, 672))[keep],
                       "z_error": rng.normal(0, .1, 672)[keep],
                       "satclockerror": (0.3 + rng.normal(0, .05, 672))[keep]})
    q = pd.date_range(t[-1] + pd.Timedelta("900s"), periods=96, freq="900s")
    r = predict(tr, q, orbit="GEO")
    tq = np.arange(672, 768) * 900.0
    truth = 1.4 + 0.6 * np.sin(2 * np.pi * tq / T_SID) + 0.25 * np.cos(4 * np.pi * tq / T_SID)
    print(f"self-test  n_train={keep.sum()}  n_query={len(q)}")
    print(f"  x_error RMS vs truth : {np.sqrt(((r.x_error - truth) ** 2).mean()):.4f} m"
          f"   (reported sigma {r.x_error_sigma.iloc[0]:.4f})")
    print(f"  clock predicted      : {r.satclockerror.iloc[0]:.4f}  (true 0.30)")
    print(r.head(3).to_string(index=False))
