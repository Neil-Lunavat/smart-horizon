#!/usr/bin/env python3
"""The 24-fold benchmark said fixed harmonic regression BEATS our self-validating
predictor on GEO (0.372 vs 0.382, winning 15/24 folds). This isolates why.

Hypothesis: on GEO the sidereal period is known exactly, so there is nothing
useful for self-validation to select -- and letting LEVEL estimators into the
candidate set lets a noisy held-out day occasionally pick a flat line where a
harmonic would have extrapolated correctly. On MEO the period is unknown, so
selection earns its keep; on GEO it may be pure added variance.

Variants, same windows, same 24 blocked folds as taskB_bench2.py:
  ours          current: levels + harmonics K=1..4, self-validated
  ours-noLvl    harmonics K=1..4 self-validated, NO level estimators on x/y/z
  fixed-K1..K4  fixed harmonic order, no selection at all
  fixed-K2+lvl  fixed K=2, but a level estimator allowed to win on the held-out day
The clock always keeps level estimators; that is settled separately.
"""
import sys
import time

import numpy as np

from taskB_final import (XYZ, CLS, S, NT, WIN, HOR, STEP, T_SID,
                         design, ridge_fit, pick_and_fit, thin)

NFOLD, GAPW, BLOCK = 24, 768, 1344


def fixed(tw, y, tq, K):
    c = ridge_fit(design(tw, [T_SID], K), y)
    if c is None:
        return np.full(len(tq), float(y.mean()))
    return design(tq, [T_SID], K) @ c


VARIANTS = ["ours", "ours-noLvl", "fixed-K1", "fixed-K2", "fixed-K3", "fixed-K4",
            "fixed-K2+lvl", "noLvl-K234", "noLvl-K34"]


def score(r):
    tw, vw, tq, y = r["tw"], r["vw"], r["tq"], r["y"]
    sse = {k: np.zeros(4) for k in VARIANTS}
    cand4 = [([T_SID], K) for K in (1, 2, 3, 4)]
    for j in range(3):                                   # x, y, z
        yj, tj = vw[:, j], y[:, j]
        sse["ours"][j] = ((tj - pick_and_fit(tw, yj, tq, cand4)) ** 2).sum()
        sse["ours-noLvl"][j] = ((tj - pick_and_fit(tw, yj, tq, cand4,
                                                   levels=False)) ** 2).sum()
        for K in (1, 2, 3, 4):
            sse[f"fixed-K{K}"][j] = ((tj - fixed(tw, yj, tq, K)) ** 2).sum()
        sse["fixed-K2+lvl"][j] = ((tj - pick_and_fit(tw, yj, tq,
                                                     [([T_SID], 2)])) ** 2).sum()
        sse["noLvl-K234"][j] = ((tj - pick_and_fit(
            tw, yj, tq, [([T_SID], K) for K in (2, 3, 4)], levels=False)) ** 2).sum()
        sse["noLvl-K34"][j] = ((tj - pick_and_fit(
            tw, yj, tq, [([T_SID], K) for K in (3, 4)], levels=False)) ** 2).sum()
    # clock: identical everywhere, level estimators only
    ck = pick_and_fit(tw, vw[:, 3], tq, [])
    for k in VARIANTS:
        sse[k][3] = ((y[:, 3] - ck) ** 2).sum()
    return sse, len(y)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    ok = [i for i in range(S) if CLS[i] in ("GEO", "GSO")]
    rg = np.random.default_rng(3)
    pool = []
    while len(pool) < N:
        s = int(ok[rg.integers(0, len(ok))])
        o = int(rg.integers(WIN, NT - HOR))
        w, f = XYZ[s, o - WIN:o], XYZ[s, o:o + HOR]
        if np.isfinite(w[:, 0]).mean() < 0.5 or np.isfinite(f[:, 0]).mean() < 0.5:
            continue
        iw = np.nonzero(np.isfinite(w[:, 0]))[0]
        tw, vw = thin((iw - WIN) * float(STEP), w[iw], 0.30, rg)
        if len(tw) < 30:
            continue
        iq = np.nonzero(np.isfinite(f[:, 0]))[0]
        if len(iq) < 8:
            continue
        pool.append(dict(tw=tw, vw=vw, tq=iq * float(STEP), y=f[iq], orig=o))
    print(f"GEO pool {len(pool)}", flush=True)

    t0 = time.time()
    scored = [score(r) for r in pool]
    print(f"scored in {time.time()-t0:.0f}s", flush=True)
    SSE = {k: np.array([s[0][k] for s in scored]) for k in VARIANTS}
    CNT = np.array([s[1] for s in scored])

    O = np.array([r["orig"] for r in pool])
    rgf = np.random.default_rng(9)
    folds = []
    for cut in np.unique((rgf.integers(int(NT * 0.30), NT - BLOCK, NFOLD * 6) // 96) * 96):
        tri = np.nonzero(O < cut - GAPW)[0]
        tei = np.nonzero((O >= cut) & (O < cut + BLOCK))[0]
        if len(tri) > 150 and len(tei) > 20:
            folds.append(tei)
        if len(folds) >= NFOLD:
            break

    per = {k: [] for k in VARIANTS}
    for tei in folds:
        n = int(CNT[tei].sum())
        for k in VARIANTS:
            per[k].append(np.sqrt(SSE[k][tei].sum(0) / n))
    per = {k: np.array(v) for k, v in per.items()}
    base = per["ours"].mean(1)

    print(f"\n{'='*94}\nGEO variants -- RMS metres, {len(folds)} blocked folds\n{'='*94}")
    print(f"{'variant':14s} {'mean':>7s} {'sd':>6s} {'worst':>6s} | "
          f"{'x':>6s} {'y':>6s} {'z':>6s} {'clock':>6s} | vs ours")
    for k, v in sorted(per.items(), key=lambda kv: kv[1].mean()):
        m = v.mean(1)
        print(f"{k:14s} {m.mean():7.3f} {m.std():6.3f} {m.max():6.3f} | "
              + " ".join(f"{x:6.3f}" for x in v.mean(0))
              + f" | {100*(m.mean()/base.mean()-1):+7.1f}%  wins {int((m<base).sum())}/{len(m)}")
