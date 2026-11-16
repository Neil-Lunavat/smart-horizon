#!/usr/bin/env python3
"""Should we do outlier treatment? -- the data description invites it, so test it.

    python outliers.py [n_trials]

`SIH_Data_Discription.pdf` says: "The data may contain outliers that require
suitable treatment." That is an explicit invitation, so it gets tested rather
than assumed either way.

Filter: per-channel MAD, keep |v - median| <= k * 1.4826 * MAD, applied to the
TRAINING window only (the test set is the answer; it is never touched).

WHY THIS FILE EXISTS RATHER THAN A ONE-LINE ANSWER
On the SIH data, MAD-5 cleaning looks like a clear win on RMS -- MEO-1 goes
0.190 -> 0.137 (28% better) and MEO-2 0.133 -> 0.128. Shipping on that basis
would be repeating Stage-02's central mistake: quoting a single window. So the
same comparison is run on real corpus data through each pair's own sampling
pattern, which is where the apparent win evaporates.

It also splits the two criteria, which turn out to DISAGREE on the SIH data:
cleaning improves RMS and simultaneously lowers W, the metric that is actually
scored. Trimming the training window's tails makes the predictor smoother, which
helps accuracy slightly and hurts residual normality slightly.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sih
from sih import COLS
from sw_score import score
from smart_horizon import predict
from replicate import pattern

CORPUS = "../taskB_data.npz"
GRID = 900.0


def mad_clean(df, k=5.0):
    """Drop rows where any channel exceeds k robust sigmas from its own median."""
    keep = np.ones(len(df), bool)
    for c in COLS:
        v = df[c].to_numpy(float)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826
        if mad > 0:
            keep &= np.abs(v - med) <= k * mad
    return df[keep].reset_index(drop=True)


def on_sih():
    print("=" * 78)
    print("1. MAD cleaning on the SIH training windows -- the single-window view")
    print("=" * 78)
    print(f"{'pair':6s} {'filter':8s} {'n_train':>7s} {'RMS':>9s} {'W':>9s}")
    for key, *_ in sih.PAIRS:
        tr, te, orb = sih.load_pair(key)
        T = te[COLS].reset_index(drop=True)
        for lbl, k in [("none", None), ("MAD 5", 5.0), ("MAD 4", 4.0), ("MAD 3", 3.0)]:
            t2 = tr if k is None else mad_clean(tr, k)
            if len(t2) < 20:
                print(f"{key:6s} {lbl:8s} {len(t2):7d}   too few rows to predict")
                continue
            P = predict(t2, te["utc_time"], orbit=orb)[COLS].reset_index(drop=True)
            _, s = score(P - T)
            print(f"{key:6s} {lbl:8s} {len(t2):7d} {s['rms']:9.4f} {s['W']:9.4f}")
        print()
    print("  Read on its own this says: clean the data, it helps MEO RMS a lot.")
    print("  It also says cleaning LOWERS W on every pair -- and W is what is scored.")


def on_corpus(n_trials=200, k=5.0, seed=7):
    if not os.path.exists(CORPUS):
        print(f"corpus {CORPUS} not found -- skipping the control")
        return
    d = np.load(CORPUS, allow_pickle=True)
    XYZ, C = d["XYZ"], np.array([str(c) for c in d["cls"]])
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 78)
    print(f"2. The same comparison on REAL data, {n_trials} windows per sampling pattern")
    print("=" * 78)
    print(f"{'pair':6s} {'n':>4s} {'RMS raw':>9s} {'RMS clean':>10s} {'helps':>7s} "
          f"{'W raw':>8s} {'W clean':>9s} {'helps':>7s}")
    for key, *_ in sih.PAIRS:
        a, b, orb = pattern(key)
        need = int(b.max()) + 1
        pool = np.where(np.isin(C, ["GEO", "GSO"] if orb == "GEO" else ["MEO"]))[0]
        acc, tries = [], 0
        while len(acc) < n_trials and tries < 40 * n_trials:
            tries += 1
            M = XYZ[pool[rng.integers(len(pool))]]
            st = int(rng.integers(0, M.shape[0] - need))
            W = M[st:st + need]
            tv, ev = W[a], W[b]
            if not (np.isfinite(tv).all() and np.isfinite(ev).all()):
                continue
            base = pd.Timestamp("2026-01-01")
            trd = pd.DataFrame({"utc_time": base + pd.to_timedelta(a * GRID, "s"),
                                **{c: tv[:, j] for j, c in enumerate(COLS)}})
            qt = base + pd.to_timedelta(b * GRID, "s")
            T = pd.DataFrame(ev, columns=COLS)
            t2 = mad_clean(trd, k)
            if len(t2) < 20:
                continue
            try:
                P0 = predict(trd, qt, orbit=orb)[COLS].reset_index(drop=True)
                P1 = predict(t2, qt, orbit=orb)[COLS].reset_index(drop=True)
            except Exception:
                continue
            _, s0 = score(P0 - T)
            _, s1 = score(P1 - T)
            acc.append((s0["rms"], s1["rms"], s0["W"], s1["W"]))
        R = np.array(acc)
        print(f"{key:6s} {len(R):4d} {np.median(R[:,0]):9.4f} {np.median(R[:,1]):10.4f} "
              f"{100*np.mean(R[:,1]<R[:,0]):6.0f}% {np.median(R[:,2]):8.4f} "
              f"{np.median(R[:,3]):9.4f} {100*np.mean(R[:,3]>R[:,2]):6.0f}%")

    print("\n  On real data MAD-5 cleaning helps in 3-10% of windows -- i.e. it is")
    print("  neutral to harmful almost always, on BOTH criteria. The SIH gain was one")
    print("  window, which is exactly the trap Stage-02 fell into and documented.")
    print("\n  DECISION: no outlier filtering. The predictor already handles outliers")
    print("  where it matters -- self-validation scores each candidate on a held-out")
    print("  day, so a basis that chased an outlier loses there and is not selected.")
    print("  Deleting rows up front also destroys real signal: on these files a large")
    print("  value after a long gap is usually a genuine stale-ephemeris excursion")
    print("  (Stage-01 SS10: 5.3x fresh-to-stale), not a defect.")


if __name__ == "__main__":
    on_sih()
    on_corpus(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
