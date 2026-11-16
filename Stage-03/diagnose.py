#!/usr/bin/env python3
"""Is the PS-08 data dirty, or is it broken? -- answered per channel.

    python diagnose.py

Stage-01 established that the delivered files cannot support the stated task.
This asks the narrower question that decides how to USE them: for each channel,
is the damage a cleaning problem (fixable by reading the file correctly) or a
content problem (the numbers are not a sampled physical series)?

THE TEST
Broadcast-minus-precise error is a smooth function of time.  At the spacings
these files use, real error has lag-1 autocorrelation close to 1.  So ACF at the
file's own modal spacing separates the two cases, PROVIDED two confounds are
removed first:

  1. gap pairs   consecutive ROWS are not consecutive SAMPLES -- coverage is
                 13-32% and gaps reach 26 h.  Mixing a 600 s neighbour with a
                 6 h jump biases ACF toward zero and makes clean data look like
                 noise.  sih.spacing() masks to genuinely adjacent pairs.
  2. static bias the corpus carries a large per-satellite offset (GEO x/y median
                 -1.14 / -2.93 m) that the SIH files do not.  A constant inflates
                 ACF toward 1 for free.  Both sides are de-meaned.

Control is the Stage-02 corpus (`taskB_data.npz`, 103 satellites x 23424 epochs),
subsampled to each SIH file's own spacing and de-meaned identically -- so the
comparison is like-for-like rather than against a textbook value.

SEPARATELY, for GEO: the antiparallel artefact is tested for REPAIRABILITY.  If
the file were a correct series emitted with an inconsistent sign convention, the
signs could be propagated back into consistency and the series recovered.  That
is tried, and `|v|` -- the vector magnitude, which no sign flip can change --
is used as the sign-independent referee.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sih
from sih import COLS

CORPUS = "../taskB_data.npz"


# --------------------------------------------------------------- smoothness
def acf1(df, mask=None):
    """Lag-1 ACF per channel plus |v|, de-meaned, on adjacent pairs only."""
    V = df[COLS].to_numpy(float)
    V = V - V.mean(0)
    if mask is None:
        mask = np.ones(len(V) - 1, bool)
    out = []
    for j in range(4):
        a, b = V[:-1, j][mask], V[1:, j][mask]
        out.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 4 and np.ptp(a) > 0 else np.nan)
    v = np.linalg.norm(V[:, :3], axis=1)
    av = float(np.corrcoef(v[:-1][mask], v[1:][mask])[0, 1]) if mask.sum() > 4 else np.nan
    return out, av


def corpus_control(step_s, cls, n_win=400, seed=0):
    """Same statistic on real data of the same class at the same spacing."""
    if not os.path.exists(CORPUS):
        return None
    d = np.load(CORPUS, allow_pickle=True)
    XYZ, C = d["XYZ"], np.array([str(c) for c in d["cls"]])
    want = ["GEO", "GSO"] if cls == "GEO" else ["MEO"]
    idx = np.where(np.isin(C, want))[0]
    stride = max(1, int(round(step_s / 900.0)))
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n_win):
        i = idx[rng.integers(len(idx))]
        M = XYZ[i]
        st = rng.integers(0, M.shape[0] - 672)
        W = M[st:st + 672:stride]
        if len(W) < 20 or not np.isfinite(W).all():
            continue
        W = W - W.mean(0)
        a = [np.corrcoef(W[:-1, j], W[1:, j])[0, 1] for j in range(4)]
        v = np.linalg.norm(W[:, :3], axis=1)
        acc.append(a + [np.corrcoef(v[:-1], v[1:])[0, 1]])
    return np.nanmedian(np.array(acc, float), axis=0) if acc else None


# ----------------------------------------------------------- GEO sign repair
def sign_repair(V):
    """Propagate a consistent sign: s[i+1] = s[i] * sign(v_i . v_i+1).

    If the file were a real series emitted with a flipping sign convention, this
    inverts that exactly and the result should be smooth.
    """
    s = np.ones(len(V))
    for i in range(len(V) - 1):
        s[i + 1] = s[i] * (1.0 if V[i] @ V[i + 1] >= 0 else -1.0)
    return V * s[:, None], int((np.diff(s) != 0).sum())


def main():
    print("=" * 82)
    print("1. DUPLICATION -- a cleaning problem, and a lossless one")
    print("=" * 82)
    print(sih.dup_report().to_string(index=False))
    print("\n  Each MEO file has exactly ONE backward time step, spanning the whole file:")
    print("  it is the same block written twice, not interleaved repeats. Dedup is")
    print("  therefore lossless and not a judgement call. GEO has no duplication.")

    print("\n" + "=" * 82)
    print("2. SMOOTHNESS -- de-meaned, adjacent pairs only, vs matched corpus control")
    print("=" * 82)
    print(f"{'file':12s} {'dt':>6s} {'pairs':>6s} {'x':>7s}{'y':>7s}{'z':>7s}{'clk':>7s} {'|v|':>8s}")
    rows = []
    for name, tr, te, orb in [(p[0] + "/train", p[1], None, p[3]) for p in sih.PAIRS] + \
                             [(p[0] + "/test", p[2], None, p[3]) for p in sih.PAIRS]:
        df = sih.load(tr)
        md, m = sih.spacing(df)
        a, av = acf1(df, m)
        rows.append((name, orb, md, int(m.sum()), a, av))
        print(f"{name:12s} {md:6.0f} {int(m.sum()):6d} " +
              "".join(f"{x:7.3f}" for x in a) + f" {av:8.3f}")
    print()
    for cls, step in [("GEO", 1260), ("GEO", 900), ("MEO", 3600), ("MEO", 600)]:
        c = corpus_control(step, cls)
        if c is not None:
            print(f"{'CORPUS ' + cls:12s} {step:6.0f} {'':>6s} " +
                  "".join(f"{x:7.3f}" for x in c[:4]) + f" {c[4]:8.3f}   <- real data, same spacing")

    print("\n" + "=" * 82)
    print("3. IS THE GEO ARTEFACT REPAIRABLE?")
    print("=" * 82)
    for f in ["DATA_GEO_Train.csv", "DATA_GEO_Test.csv"]:
        df = sih.load(f)
        V = df[COLS].to_numpy(float)
        md, m = sih.spacing(df)
        R, nf = sign_repair(V)
        a0, v0 = acf1(df, m)
        a1, v1 = acf1(pd.concat([df[["utc_time"]],
                                 pd.DataFrame(R, columns=COLS)], axis=1), m)
        print(f"\n  {f[5:-4]}  ({nf} sign flips inferred over {len(V)-1} steps)")
        print(f"    as delivered : acf = {np.round(a0,3)}   |v| = {v0:.3f}")
        print(f"    sign-repaired: acf = {np.round(a1,3)}   |v| = {v1:.3f}")
        print(f"    |v| is INVARIANT to any sign flip, so it is the referee:")
        print(f"      real GEO/GSO at this spacing has |v| acf ~0.97; this file has {v0:.3f}.")
    print("\n  Sign repair lifts the position channels but cannot lift |v|, because it")
    print("  does not touch |v|. The magnitude series is itself incoherent, so the")
    print("  corruption is NOT merely a sign convention -- there is no consistent")
    print("  underlying series for a sign fix to recover.")

    print("\n" + "=" * 82)
    print("4. HOW MUCH SIGNAL IS THERE AT ALL? -- oracle bound per pair")
    print("=" * 82)
    print("  'oracle constant' cheats: it is the TEST set's own mean, which no")
    print("  legitimate model can know. It bounds every constant predictor.")
    print(f"\n{'pair':6s} {'chan':14s} {'test sd':>9s} {'RMS zero':>9s} {'RMS oracle':>11s} {'oracle gain':>12s}")
    for k, *_ in sih.PAIRS:
        tr, te, orb = sih.load_pair(k)
        T = te[COLS].to_numpy(float)
        z = np.sqrt((T ** 2).mean(0))
        o = np.sqrt(((T - T.mean(0)) ** 2).mean(0))
        for j, c in enumerate(COLS):
            print(f"{k:6s} {c:14s} {T[:,j].std(ddof=1):9.3f} {z[j]:9.3f} {o[j]:11.3f} "
                  f"{100*(1-o[j]/z[j]):11.1f}%")
        print(f"{k:6s} {'MEAN':14s} {'':9s} {z.mean():9.3f} {o.mean():11.3f} "
              f"{100*(1-o.mean()/z.mean()):11.1f}%")


if __name__ == "__main__":
    main()
