#!/usr/bin/env python3
"""Do the Stage-03 single-shot results mean anything? -- resampling control.

    python replicate.py [n_trials]

THE PROBLEM THIS SOLVES
apply_model.py scores each pair ONCE, on 6, 18 and 69 test points. Stage-02's
hardest-won lesson was that a single window can be sold as anything: the figure
quoted all session there turned out to be the easiest fold in the corpus, and the
honest number was double it. Three single windows here are exactly that trap
again, and the bootstrap CIs on W ([0.58, 0.95] at n = 6) say so directly.

THE CONTROL
Take each SIH pair's SAMPLING PATTERN -- the train timestamps, the test
timestamps, the gaps, the horizon -- and lay it over REAL corpus data of the same
orbit class at many random start epochs. Run the identical predictor and the
identical scorer. That gives the distribution the SIH number is one draw from,
so we can say whether it is typical, lucky or unlucky.

This is the same device Stage-01 used to prove the upload-observability claim
(real SIH timestamps laid over real arc boundaries), reused for the model.

Timestamps are snapped to the corpus 900 s grid, which preserves sampling
density, gap structure and horizon while losing sub-grid jitter -- so this is a
control on the SAMPLING, not a claim about off-grid behaviour (Stage-02 §2
measured that separately at ~10-20% degradation).
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

CORPUS = "../taskB_data.npz"
GRID = 900.0


def pattern(key):
    """(train offsets, test offsets) in grid steps, relative to train start."""
    tr, te, orb = sih.load_pair(key)
    t0 = tr.utc_time.min()
    a = np.round((tr.utc_time - t0).dt.total_seconds().to_numpy() / GRID).astype(int)
    b = np.round((te.utc_time - t0).dt.total_seconds().to_numpy() / GRID).astype(int)
    return np.unique(a), np.unique(b), orb


def main(n_trials=300):
    if not os.path.exists(CORPUS):
        print(f"corpus {CORPUS} not found -- this control needs it")
        return
    d = np.load(CORPUS, allow_pickle=True)
    XYZ, C, S = d["XYZ"], np.array([str(c) for c in d["cls"]]), d["sats"]
    rng = np.random.default_rng(0)

    actual = pd.read_csv("out/scores.csv")
    out = []
    for key, *_ in sih.PAIRS:
        a, b, orb = pattern(key)
        need = int(b.max()) + 1
        pool = np.where(np.isin(C, ["GEO", "GSO"] if orb == "GEO" else ["MEO"]))[0]
        got = 0
        while got < n_trials:
            i = pool[rng.integers(len(pool))]
            M = XYZ[i]
            st = int(rng.integers(0, M.shape[0] - need))
            W = M[st:st + need]
            tr_v, te_v = W[a], W[b]
            if not (np.isfinite(tr_v).all() and np.isfinite(te_v).all()):
                continue
            base = pd.Timestamp("2026-01-01")
            trd = pd.DataFrame({"utc_time": base + pd.to_timedelta(a * GRID, "s"),
                                **{c: tr_v[:, j] for j, c in enumerate(COLS)}})
            qt = base + pd.to_timedelta(b * GRID, "s")
            try:
                P = predict(trd, qt, orbit=orb)[COLS].reset_index(drop=True)
            except Exception:
                continue
            T = pd.DataFrame(te_v, columns=COLS)
            _, sm = score(P - T)
            _, sz = score(pd.DataFrame(-T.to_numpy(), columns=COLS))   # predict zero
            out.append({"pair": key, "sat": str(S[i]), "rms": sm["rms"], "W": sm["W"],
                        "rms_zero": sz["rms"], "W_zero": sz["W"]})
            got += 1

    R = pd.DataFrame(out)
    R.to_csv("out/replicate.csv", index=False)

    print("=" * 84)
    print(f"SIH sampling patterns replayed on real corpus data, {n_trials} trials each")
    print("=" * 84)
    for key, *_ in sih.PAIRS:
        g = R[R.pair == key]
        act = actual[(actual.pair == key) & (actual.method == "SMART-HORIZON")].iloc[0]
        az = actual[(actual.pair == key) & (actual.method == "zero")].iloc[0]
        n_te = int(act.n_test)
        pr = float((g.rms <= act.RMS).mean())
        pw = float((g.W <= act.W).mean())
        print(f"\n{key}  (orbit {act.orbit}, test n={n_te})")
        print(f"  corpus RMS   : median {g.rms.median():.3f}  "
              f"[p05 {g.rms.quantile(.05):.3f}, p95 {g.rms.quantile(.95):.3f}]"
              f"   beats zero in {100*(g.rms < g.rms_zero).mean():.0f}% of trials")
        print(f"  SIH   RMS    : {act.RMS:.3f}   -> {100*pr:.0f}th pct of the corpus distribution")
        print(f"  corpus W     : median {g.W.median():.3f}  "
              f"[p05 {g.W.quantile(.05):.3f}, p95 {g.W.quantile(.95):.3f}]")
        print(f"  SIH   W      : {act.W:.3f}   -> {100*pw:.0f}th pct")
        print(f"  (SIH 'do nothing' RMS {az.RMS:.3f}, W {az.W:.3f})")

    print("\n" + "=" * 84)
    print("READ THIS AS: where the one SIH draw sits in the distribution of what")
    print("this predictor does on REAL data sampled exactly the same way.")
    print("=" * 84)
    print("\nwrote out/replicate.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
