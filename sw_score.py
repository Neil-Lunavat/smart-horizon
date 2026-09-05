#!/usr/bin/env python3
"""The competition's actual scoring metric, reverse-engineered and calibrated.

Note.pdf (issued after the original data) states the evaluation criteria:

  Priority 1  Shapiro-Wilk W on the RESIDUAL (predicted - test), its p-value, and
              the hypothesis result at alpha = 0.05, averaged over x/y/z/clock
              with equal weight.  HIGHER W IS BETTER.
  Priority 2  mean and standard deviation of the residual  (tiebreak)
  Priority 3  Q-Q plot of the residual                     (tiebreak)

and requires participants to write their own code, benchmarked against
SW_ReferenceData.xlsx, for which the stated answer is

  W = 0.9810,  p = 0.5840,  H = 0

WHICH TEST THEY ACTUALLY RAN
The reference sample is 45 values and is exactly MATLAB `rng('default');
randn(1,45)`.  Plain Shapiro-Wilk (scipy / R `shapiro.test` / Royston AS R94)
gives W = 0.9852, p = 0.8262 on it, NOT their numbers.  Shapiro-FRANCIA gives
W = 0.9814, p = 0.5838, which matches their p to 2e-4.

That identifies the scorer as MATLAB's `swtest.m` (BenSaida, File Exchange),
which switches test on the sample's own kurtosis:

    kurtosis > 3  ->  Shapiro-Francia   (leptokurtic)
    kurtosis <= 3 ->  Shapiro-Wilk      (platykurtic)

The reference sample has kurtosis 3.33, so it took the Francia branch.  Their
quoted W = 0.9810 is inconsistent with their own quoted p (W = 0.9810 implies
p = 0.567); the p-value is the reliable anchor and it pins the algorithm.

CONSEQUENCE, and it is not cosmetic: the metric changes algorithm depending on
the residual being scored.  A residual with kurtosis 2.9 and one with 3.1 are
graded by two different statistics.  `swtest()` below reproduces that behaviour
so our scores are on the same footing as theirs; `which` in the output records
which branch each column took.

WHAT THE METRIC REWARDS, and it is not accuracy.
W measures whether the residual is NORMALLY DISTRIBUTED, not whether it is small.
It is scale-invariant: multiply every residual by 1000 and W does not move.  So a
predictor wrong by a Gaussian 5 m outscores one wrong by a skewed 5 cm.  RMS and
W are separate axes; this stage reports both everywhere.

Usage:  python sw_score.py          # runs the calibration check
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
COLS = ["x_error", "y_error", "z_error", "satclockerror"]


#----------------------------------------------------------- the statistic
def shapiro_francia(x):
    """W' and p by Royston (1993). Used by swtest.m when kurtosis > 3."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    m = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    w = m / np.sqrt(m @ m)
    ss = np.sum((x - x.mean()) ** 2)
    if ss <= 0:
        return np.nan, np.nan
    W = float((w @ x) ** 2 / ss)
    nu = np.log(n)
    mu = -1.2725 + 1.0521 * (np.log(nu) - nu)
    sg = 1.0308 - 0.26758 * (np.log(nu) + 2 / nu)
    z = (np.log(1 - W) - mu) / sg
    return W, float(1 - stats.norm.cdf(z))


def swtest(x, alpha=ALPHA):
    """MATLAB swtest.m: Shapiro-Francia if kurtosis > 3, else Shapiro-Wilk.

    Returns (W, p, H, which). H = 1 rejects normality at alpha.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.ptp(x) <= 0:
        return np.nan, np.nan, np.nan, "n/a"
    if stats.kurtosis(x, fisher=False, bias=True) > 3.0:
        W, p = shapiro_francia(x)
        which = "SF"
    else:
        W, p = stats.shapiro(x)          # Royston AS R94, same branch swtest uses
        W, p, which = float(W), float(p), "SW"
    return W, p, int(p < alpha), which


#--------------------------------------------------------------- scoring
def score(resid: pd.DataFrame, cols=COLS):
    """Per-column and equal-weight-averaged scores for a residual frame.

    Returns (per_column_DataFrame, summary_dict). The summary is what the
    competition asks to be reported: W, p, H averaged over the four parameters,
    plus the priority-2 mean/sd, and RMS for our own use.
    """
    rows = []
    for c in cols:
        r = resid[c].to_numpy(float)
        r = r[np.isfinite(r)]
        W, p, H, which = swtest(r)
        rows.append({"param": c, "n": len(r), "W": W, "p": p, "H": H,
                     "which": which,
                     "mean": float(np.mean(r)) if len(r) else np.nan,
                     "sd": float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
                     "rms": float(np.sqrt(np.mean(r ** 2))) if len(r) else np.nan})
    per = pd.DataFrame(rows)
    summary = {"W": float(per.W.mean()), "p": float(per.p.mean()),
               "H": float(per.H.mean()), "n_reject": int(per.H.sum()),
               "mean": float(per["mean"].mean()), "sd": float(per.sd.mean()),
               "rms": float(per.rms.mean())}
    return per, summary


def ci_W(x, n_boot=2000, seed=0):
    """Bootstrap 95% CI on W, Note.pdf asks for a confidence interval.

    With n = 6 (the size of DATA_MEO_Test) this interval is enormous, which is
    the point: it puts the sample-size problem inside the organisers' own metric.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    ws = [w for w in (swtest(rng.choice(x, len(x), replace=True))[0]
                      for _ in range(n_boot)) if np.isfinite(w)]
    if not ws:
        return np.nan, np.nan
    return float(np.percentile(ws, 2.5)), float(np.percentile(ws, 97.5))


def calibrate(path="data/SW_ReferenceData.xlsx", verbose=True):
    """Reproduce the organisers' reference numbers, and show why SW alone cannot."""
    x = pd.read_excel(path, header=None).iloc[:, 0].to_numpy(float)  # row 1 is data
    k = stats.kurtosis(x, fisher=False, bias=True)
    sw = stats.shapiro(x)
    sf = shapiro_francia(x)
    W, p, H, which = swtest(x)
    ok = abs(p - 0.5840) < 1e-3 and which == "SF" and H == 0
    if verbose:
        print(f"reference sample: n = {len(x)}, kurtosis = {k:.4f}  -> swtest picks {which}")
        print(f"  stated by organisers : W = 0.9810  p = 0.5840  H = 0")
        print(f"  plain Shapiro-Wilk   : W = {sw[0]:.4f}  p = {sw[1]:.4f}   <- does NOT match")
        print(f"  Shapiro-Francia      : W = {sf[0]:.4f}  p = {sf[1]:.4f}   <- matches p to 2e-4")
        print(f"  MATCH: {ok}")
    return ok, W, p, H


if __name__ == "__main__":
    import os, sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(0 if calibrate()[0] else 1)
