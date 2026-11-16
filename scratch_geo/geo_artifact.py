#!/usr/bin/env python3
"""GEO sign-alternation artefact: can modelling it raise the Shapiro-Wilk W?

Research question
-----------------
The PS-08 brief says a normal residual means "the developed model has REMOVED THE
SYSTEMATIC ERRORS".  DATA_GEO_*.csv carries a blatant systematic pattern: large
excursions arrive in near-exact NEGATION pairs and lag-1 autocorrelation is
negative.  Every constant predictor scores W = 0.7708 on GEO_Test.  Does a model
that captures the artefact beat that?

Nothing here is fitted on the test set except variants explicitly labelled
ORACLE, which are reported as bounds and cannot be shipped.

Run:  .venv/Scripts/python.exe scratch_geo/geo_artifact.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Stage-03"))

import sih          # noqa: E402  loader/cleaner
import sw_score     # noqa: E402  the competition's calibrated swtest

C = sih.COLS
SEED = 0


# ------------------------------------------------------------------ scoring
def score(pred, V):
    """(W avg, W per channel, RMS) for residual = pred - test."""
    per, s = sw_score.score(pd.DataFrame(np.asarray(pred, float) - V, columns=C))
    return s["W"], per.W.to_numpy(), s["rms"]


def row(name, pred, V, note=""):
    w, pc, r = score(pred, V)
    return {"variant": name, "W_x": pc[0], "W_y": pc[1], "W_z": pc[2],
            "W_clk": pc[3], "W_avg": w, "RMS": r, "note": note}


def show(rows, title):
    df = pd.DataFrame(rows)
    print("\n" + title + "\n" + "-" * len(title))
    with pd.option_context("display.width", 220):
        print(df.to_string(index=False, float_format=lambda v: "%8.4f" % v))
    return df


# ------------------------------------------------------------------ helpers
def sign_propagate(V):
    """s1 = +1, s_{i+1} = s_i * sign(v_i . v_{i+1}).  Uses the series itself."""
    n = len(V)
    s = np.ones(n)
    for k in range(n - 1):
        s[k + 1] = s[k] * (1.0 if V[k] @ V[k + 1] >= 0 else -1.0)
    return s


def big_mask(V, thr=3.0):
    mag = np.linalg.norm(V, axis=1)
    return mag > thr * np.median(mag)


def fit_parity_amplitude(tr):
    """Choose the dither amplitude `a` on GEO_TRAIN ONLY.

    Train is split into a fit half and a held-out half; `a` is the multiple of
    the fit half's sd that maximises W of (a*(-1)^i - v) on the held-out half.
    No test data is touched.
    """
    Vt = tr[C].to_numpy(float)
    h = len(Vt) // 2
    sd = Vt[:h].std(0)
    hold = Vt[h:]
    par = (-1.0) ** np.arange(len(hold))
    grid = np.arange(0.0, 6.01, 0.25)
    ws = [score(par[:, None] * (a * sd)[None, :], hold)[0] for a in grid]
    return float(grid[int(np.argmax(ws))]), pd.DataFrame({"a": grid, "W_heldout": ws})


# ------------------------------------------------------------------ robustness
def sub_blocks(n, k):
    e = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(e[j], e[j + 1]) for j in range(k)]


def robustness(pred, V, label):
    """Does W(variant) - W(zero) hold inside halves and thirds of the test set?"""
    out = []
    pred = np.asarray(pred, float)
    for k in (2, 3):
        for j, ix in enumerate(sub_blocks(len(V), k)):
            w0 = score(np.zeros((len(ix), 4)), V[ix])[0]
            w1 = score(pred[ix], V[ix])[0]
            out.append({"variant": label, "split": "%d-way blk%d" % (k, j + 1),
                        "n": len(ix), "W_zero": w0, "W_var": w1, "dW": w1 - w0})
    return out


def bootstrap_dW(pred, V, n_boot=2000, seed=SEED):
    """Bootstrap the paired W difference (variant - zero) over resampled rows."""
    rng = np.random.default_rng(seed)
    n = len(V)
    pred = np.asarray(pred, float)
    d = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        w0 = score(np.zeros((n, 4)), V[ix])[0]
        w1 = score(pred[ix], V[ix])[0]
        if np.isfinite(w0) and np.isfinite(w1):
            d.append(w1 - w0)
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), float((d > 0).mean())


# ------------------------------------------------------------------ main
def main():
    tr, te, _ = sih.load_pair("GEO")
    Vtr = tr[C].to_numpy(float)
    V = te[C].to_numpy(float)
    n = len(V)
    par = (-1.0) ** np.arange(n)
    sd_tr = Vtr.std(0)
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("GEO artefact study   train n=%d   test n=%d" % (len(tr), n))
    print("=" * 78)

    # -- 1. the artefact is real ------------------------------------------
    print("\n1. THE ARTEFACT IS REAL (descriptive, no model yet)")
    for nm, X in (("train", Vtr), ("test", V)):
        nrm = np.linalg.norm(X, axis=1)
        cos = np.sum(X[:-1] * X[1:], axis=1) / (nrm[:-1] * nrm[1:])
        b = big_mask(X)
        sb = np.sign(X[b, 0])
        print("   %-5s median cos(v_i,v_i+1) = %+.3f | frac cos < -0.9 = %.3f | "
              "large excursions k=%2d, consecutive-pair sign alternation = %.3f"
              % (nm, np.median(cos), np.mean(cos < -0.9), b.sum(),
                 np.mean(sb[:-1] * sb[1:] < 0)))
    print("   test large-excursion sign(x) sequence: "
          + "".join("+" if s > 0 else "-" for s in np.sign(V[big_mask(V), 0])))
    print("   -> in GEO_Test every one of the 24 large excursions flips sign relative")
    print("      to the previous one (24/24, stable for thresholds 2x-3x median).")
    print("      The pattern is not in doubt.")

    # -- 2. is the flip locked to the schedule? ---------------------------
    print("\n2. IS THE FLIP LOCKED TO SAMPLE INDEX / TIMESTAMP? (it must be, to be usable)")
    for nm, df, X in (("train", tr, Vtr), ("test", te, V)):
        p = (-1.0) ** np.arange(len(X))
        mp = 1.0 - 2.0 * (df.utc_time.dt.minute.to_numpy() % 2)
        print("   %-5s corr(v, (-1)^i)        = %s" % (nm, np.round(
            [np.corrcoef(X[:, j], p)[0, 1] for j in range(4)], 3)))
        print("         corr(v, minute parity) = %s" % np.round(
            [np.corrcoef(X[:, j], mp)[0, 1] for j in range(4)], 3))
    b = big_mask(V)
    print("   test: large-excursion row indices = %s" % np.where(b)[0])
    print("         their index parity          = %s" % (np.where(b)[0] % 2))
    print("   -> the SIGN alternates over the large events, but neither the sign nor")
    print("      WHICH rows are large is recoverable from the timestamp or the row")
    print("      index. A shippable model sees only the schedule: the pattern is")
    print("      observable but not predictable.")

    # -- 3. baselines -----------------------------------------------------
    base = [
        row("zero (do nothing)", np.zeros((n, 4)), V, "BASELINE"),
        row("train mean", np.tile(Vtr.mean(0), (n, 1)), V, "constant"),
        row("train median", np.tile(np.median(Vtr, 0), (n, 1)), V, "constant"),
        row("last train value", np.tile(Vtr[-1], (n, 1)), V, "constant"),
        row("ORACLE test mean", np.tile(V.mean(0), (n, 1)), V, "cheats, still identical"),
        row("ORACLE perfect model x 0.9", 0.9 * V, V, "90% accurate, cheats"),
        row("ORACLE perfect model x 0.99", 0.99 * V, V, "99% accurate, cheats"),
    ]
    show(base, "3. BASELINES  (W is affine-invariant, so EVERY constant scores the same)")
    print("   W of the RAW TEST DATA itself is identical to predict-zero, by construction")
    print("   (residual = 0 - v = -v, and W(-v) = W(v)).  Note the last two rows: a")
    print("   cheating model that is 90% or 99% accurate scores EXACTLY the baseline W,")
    print("   because a proportional residual is an affine map and W is affine-invariant.")
    print("   Accuracy per se cannot move W at all on this file.")
    W0 = base[0]["W_avg"]
    RMS0 = base[0]["RMS"]

    # -- 4. sign / parity models -----------------------------------------
    a_fit, sweep = fit_parity_amplitude(tr)
    print("\n4. AMPLITUDE FITTED ON GEO_TRAIN ONLY -> a = %.2f (x train sd)" % a_fit)
    s_prop = sign_propagate(V)
    mpar = (1.0 - 2.0 * (te.utc_time.dt.minute.to_numpy() % 2))
    sgn = [
        row("parity a=1.0*sd_tr", par[:, None] * (1.0 * sd_tr), V, "s_i=(-1)^i"),
        row("parity a=2.0*sd_tr", par[:, None] * (2.0 * sd_tr), V, "s_i=(-1)^i"),
        row("parity a=%.2f*sd_tr TRAIN-FITTED" % a_fit,
            par[:, None] * (a_fit * sd_tr), V, "shippable in principle"),
        row("minute-parity a=2.0*sd_tr", mpar[:, None] * (2.0 * sd_tr), V, "timestamp-locked"),
        row("magnitude-only |pred|=median|v|_tr, sign=+",
            np.tile(np.median(np.abs(Vtr), 0), (n, 1)), V, "constant -> no change"),
        row("ORACLE sign-propagated a=2.0*sd_tr", s_prop[:, None] * (2.0 * sd_tr), V,
            "sign taken from TEST"),
        row("ORACLE pred = -v[i-1]", np.vstack([np.zeros((1, 4)), -V[:-1]]), V,
            "the artefact used directly"),
        row("ORACLE pred = +v[i-1]", np.vstack([np.zeros((1, 4)), V[:-1]]), V, "cheats"),
    ]
    show(sgn, "4. SIGN-PATTERN MODELS")

    # -- 5. the control that settles it -----------------------------------
    print("\n5. CONTROL: does the SIGN SEQUENCE carry the gain, or is it just dither?")
    ws = np.array([score(rng.choice([-1.0, 1.0], size=(n, 1)) * (2.0 * sd_tr), V)[0]
                   for _ in range(400)])
    wp = score(par[:, None] * (2.0 * sd_tr), V)[0]
    wo = score(s_prop[:, None] * (2.0 * sd_tr), V)[0]
    print("   RANDOM +-1 sign, same amplitude  : W = %.4f  [5-95%% %.4f .. %.4f]  (400 draws)"
          % (ws.mean(), np.percentile(ws, 5), np.percentile(ws, 95)))
    print("   PARITY sign, same amplitude      : W = %.4f   pctile of random = %2.0f"
          % (wp, 100 * (ws < wp).mean()))
    print("   ORACLE TRUE sign, same amplitude : W = %.4f   pctile of random = %2.0f"
          % (wo, 100 * (ws < wo).mean()))
    wsp = np.array([score(-V[rng.permutation(n)], V)[0] for _ in range(400)])
    wnp = score(np.vstack([np.zeros((1, 4)), -V[:-1]]), V)[0]
    print("   RANDOM -v[randperm]              : W = %.4f  [5-95%% %.4f .. %.4f]"
          % (wsp.mean(), np.percentile(wsp, 5), np.percentile(wsp, 95)))
    print("   ORACLE -v[i-1] (real artefact)   : W = %.4f   pctile of random = %2.0f"
          % (wnp, 100 * (wsp < wnp).mean()))
    print("   -> the TRUE sign sequence does no better than a coin flip of the same")
    print("      amplitude, and -v[i-1] does no better than -v[randperm]. Every bit of")
    print("      the movement in W is AMPLITUDE, none of it is the sign structure.")

    # -- 6. dither sweep ---------------------------------------------------
    print("\n6. PURE DITHER SWEEP  pred = a * sd_train * N(0,1)   (uses no data at all)")
    Z = np.random.default_rng(SEED).standard_normal((n, 4))
    dit = [row("gaussian dither a=%g" % a, Z * (a * sd_tr), V, "zero information")
           for a in (0.5, 1, 2, 3, 4, 6, 8, 12)]
    show(dit, "6. GAUSSIAN DITHER")
    print("   -> W rises monotonically to ~0.98 while RMS grows without bound. The")
    print("      metric is driven arbitrarily high by a predictor that knows nothing.")

    # -- 7. the ceiling ----------------------------------------------------
    print("\n7. THE CEILING: an ORACLE that perfectly removes the systematic excursions")
    orc = []
    for thr in (2.0, 3.0, 4.0, 6.0):
        bm = big_mask(V, thr)
        orc.append(row("ORACLE remove excursions >%gx median (k=%d)" % (thr, bm.sum()),
                       np.where(bm[:, None], V, 0.0), V, "cannot be shipped"))
    orc.append(row("ORACLE shrink excursions 50%",
                   np.where(b[:, None], 0.5 * V, 0.0), V, "cannot be shipped"))
    show(orc, "7. ORACLE ARTEFACT REMOVAL  (upper bound on capturing the artefact)")
    wo2, _, ro2 = score(np.where(b[:, None], V, 0.0), V)
    print("   -> Perfectly deleting all %d systematic excursions cuts RMS %.2f -> %.2f"
          % (b.sum(), RMS0, ro2))
    print("      (%.0f%% more accurate) and moves W THE WRONG WAY: %.4f -> %.4f."
          % (100 * (1 - ro2 / RMS0), W0, wo2))
    print("      This is a BOUND: no legitimate model can remove the artefact better")
    print("      than one handed the truth, and the truth-handed model LOWERS W.")

    # -- 8. heteroscedastic ------------------------------------------------
    print("\n8. HETEROSCEDASTIC VARIANT (standardised residual r_i / sigma_i)")
    mag = np.linalg.norm(V, axis=1)
    sig = np.clip(pd.Series(mag).rolling(5, center=True, min_periods=1).median().to_numpy(),
                  1e-3, None)
    per, s = sw_score.score(pd.DataFrame((np.zeros((n, 4)) - V) / sig[:, None], columns=C))
    print("   zero-predictor residual / ORACLE rolling-median |v| : W = %.4f  per-ch %s"
          % (s["W"], np.round(per.W.to_numpy(), 3)))
    print("   INADMISSIBLE twice over: (a) sigma_i is built from the test values")
    print("   themselves; (b) Note.pdf scores W on the residual (predicted - test) in")
    print("   metres -- dividing by a self-chosen scale is not that residual, and any")
    print("   sigma_i proportional to |r_i| drives W to 1 by construction.")

    # -- 9. robustness -----------------------------------------------------
    print("\n9. ROBUSTNESS OF THE APPARENT WINNERS")
    cands = {
        "parity a=%.2f TRAIN-FITTED" % a_fit: par[:, None] * (a_fit * sd_tr),
        "parity a=2.0": par[:, None] * (2.0 * sd_tr),
        "gaussian dither a=4": Z * (4.0 * sd_tr),
        "ORACLE remove excursions": np.where(b[:, None], V, 0.0),
    }
    rb = []
    for lab, P in cands.items():
        rb += robustness(P, V, lab)
    show(rb, "9a. CONTIGUOUS SPLIT-HALF / SPLIT-THIRD  (dW = W_variant - W_zero)")

    print("\n9b. BOOTSTRAP OF THE PAIRED W DIFFERENCE (2000 resamples of the 69 rows)")
    for lab, P in cands.items():
        m, lo, hi, pos = bootstrap_dW(P, V)
        print("   %-30s dW = %+.4f  95%% CI [%+.4f, %+.4f]  P(dW>0) = %.3f"
              % (lab, m, lo, hi, pos))

    # -- 10. verdict --------------------------------------------------------
    wfit, _, rfit = score(par[:, None] * (a_fit * sd_tr), V)
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  baseline, any constant predictor            W = %.4f  RMS %6.2f" % (W0, RMS0))
    print("  best legitimate variant (train-fitted       W = %.4f  RMS %6.2f"
          % (wfit, rfit))
    print("    parity dither)   -- but section 5 shows a random sign sequence of the")
    print("    same amplitude scores the same, so this is DITHER, not a model of the")
    print("    artefact, and it costs %.0f%% accuracy." % (100 * (rfit / RMS0 - 1)))
    print("  ORACLE that perfectly removes the artefact  W = %.4f  RMS %6.2f"
          % (wo2, ro2))
    print("    -- %.0f%% more accurate and LESS normal." % (100 * (1 - ro2 / RMS0)))
    print("")
    print("  ANSWER TO THE RESEARCH QUESTION: NO. Capturing the systematic")
    print("  sign-alternation artefact does not raise W on GEO_Test; done perfectly")
    print("  it LOWERS it. W can only be raised on this file by inflating the")
    print("  residual with noise, which no honest submission should do.")
    print("  SHIP: nothing.  DO-NOT-SHIP: parity dither, gaussian dither,")
    print("        standardised residuals.  See FINDINGS.md.")


if __name__ == "__main__":
    main()
