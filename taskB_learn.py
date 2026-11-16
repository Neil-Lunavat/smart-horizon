#!/usr/bin/env python3
"""Does LEARNING add anything on top of the deterministic predictor?

The deterministic model has no trained parameters -- it refits inside each 7-day
window. Horizon shrinkage, the first learned layer tried, made things slightly
WORSE (GEO 0.244 -> 0.254, MEO 0.397 -> 0.401), because the self-validating basis
choice is already calibrated.

This is the fair remaining test: a gradient-boosted model that sees the window's
summary statistics AND the deterministic prediction, and tries to predict that
prediction's error. If it cannot beat zero, the honest conclusion is that the
corpus's value is in having DESIGNED and VALIDATED the model, not in supplying
weights -- and we say so.

Trained on the first 60% of the timeline, tested on the last 30%, gap between.
"""
import sys

import numpy as np
from scipy import stats

from taskB_final import (XYZ, CLS, S, NT, WIN, HOR, STEP, T4, T_SID,
                         predict_geo, predict_meo, thin, ls_period, MEO_LO, MEO_HI)

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def feats(tw, vw, tq, pred, j, geo):
    """Everything derivable from the supplied window, per query point."""
    y = vw[:, j]
    n = len(y)
    day = tw >= tw.max() - 86400.0
    half = tw >= tw.max() - 43200.0
    d3 = tw >= tw.max() - 3 * 86400.0
    mu = float(y.mean())
    sd = float(y.std())
    rec_sd = float(y[day].std()) if day.sum() > 3 else sd
    drift = float(y[day].mean() - y[~day].mean()) if 3 < day.sum() < n - 3 else 0.0
    dif = np.diff(y)
    rough = float(np.abs(dif).mean()) if len(dif) else 0.0
    P = ls_period(tw, y, MEO_LO, MEO_HI) if not geo else T_SID
    cov = n / (WIN * 1.0)
    base = [mu, sd, rec_sd, float(y[half].mean() - mu) if half.sum() > 3 else 0.0,
            float(y[d3].mean() - mu) if d3.sum() > 3 else 0.0,
            drift, rough, sd / (rec_sd + 1e-6), cov, float(P or 0) / 1e4,
            float(np.median(np.abs(y - np.median(y)))), float(stats.skew(y)),
            float(stats.kurtosis(y)), float(y[-1] - mu), float(n)]
    h = tq / 3600.0
    dev = pred[:, j] - mu
    return np.column_stack([np.repeat([base], len(tq), 0), h, dev, np.abs(dev),
                            dev / (sd + 1e-6)])


def gather(n, seed, lo, hi, rate=0.30):
    rg = np.random.default_rng(seed)
    out, tried = [], 0
    while len(out) < n and tried < n * 15:
        tried += 1
        s = int(rg.integers(0, S))
        o = int(rg.integers(max(lo, WIN), max(hi, WIN + 1)))
        if o + HOR >= NT:
            continue
        w, f = XYZ[s, o - WIN:o], XYZ[s, o:o + HOR]
        if np.isfinite(w[:, 0]).mean() < 0.5 or np.isfinite(f[:, 0]).mean() < 0.5:
            continue
        iw = np.nonzero(np.isfinite(w[:, 0]))[0]
        tw, vw = thin((iw - WIN) * float(STEP), w[iw], rate, rg)
        if len(tw) < 30:
            continue
        iq = np.nonzero(np.isfinite(f[:, 0]))[0]
        if len(iq) < 8:
            continue
        iq = np.sort(rg.choice(iq, min(len(iq), int(rg.integers(8, 60))), replace=False))
        tq = iq * float(STEP)
        geo = CLS[s] in ("GEO", "GSO")
        p = predict_geo(tw, vw, tq) if geo else predict_meo(tw, vw, tq)
        out.append(dict(geo=geo, tw=tw, vw=vw, tq=tq, y=f[iq], p=p))
    return out


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    TR_HI, TE_LO = int(NT * 0.60), int(NT * 0.70)
    tr = gather(N, 401, WIN, TR_HI)
    te = gather(N, 402, TE_LO, NT - HOR)
    print(f"train {len(tr)} windows, test {len(te)} windows", flush=True)
    if xgb is None:
        print("xgboost missing"); sys.exit(0)
    for geo, nm in ((True, "GEO/GSO"), (False, "MEO")):
        TR = [r for r in tr if r["geo"] == geo]
        TE = [r for r in te if r["geo"] == geo]
        if len(TR) < 40 or len(TE) < 20:
            print(f"\n{nm}: too few windows"); continue
        print(f"\n{'='*76}\n{nm}: {len(TR)} train / {len(TE)} test windows\n{'='*76}")
        print(f"{'target':10s} {'determ.':>9s} {'+learned':>9s} {'change':>9s} {'verdict':>12s}")
        for j, c in enumerate(T4):
            Xtr = np.vstack([feats(r["tw"], r["vw"], r["tq"], r["p"], j, geo) for r in TR])
            ytr = np.concatenate([r["y"][:, j] - r["p"][:, j] for r in TR])
            Xte = np.vstack([feats(r["tw"], r["vw"], r["tq"], r["p"], j, geo) for r in TE])
            yte = np.concatenate([r["y"][:, j] - r["p"][:, j] for r in TE])
            m = xgb.XGBRegressor(tree_method="hist", max_depth=5, learning_rate=0.05,
                                 n_estimators=350, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=2.0, verbosity=0)
            m.fit(np.nan_to_num(Xtr), ytr)
            corr = m.predict(np.nan_to_num(Xte))
            a = float(np.sqrt((yte ** 2).mean()))
            b = float(np.sqrt(((yte - corr) ** 2).mean()))
            v = "helps" if b < a * 0.98 else ("no gain" if b < a * 1.02 else "HURTS")
            print(f"{c:10s} {a:9.3f} {b:9.3f} {100*(b/a-1):+8.1f}% {v:>12s}")
