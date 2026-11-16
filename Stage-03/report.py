#!/usr/bin/env python3
"""Produce exactly what Note.pdf asks a team to hand in.

    python report.py

  Priority 1  W, p-value, hypothesis result at alpha = 0.05, averaged over the
              four parameters with equal weight, plus a confidence interval
  Priority 2  mean and standard deviation of the residual
  Priority 3  Q-Q plot of the residual

Writes out/submission_report.md and out/qq_<pair>.png.

The Gaussian null is printed alongside every W because without it the number is
unreadable: at n = 6 a residual that is EXACTLY normal still scores a median
0.921 with a 5-95% range of [0.785, 0.981]. Reporting W = 0.907 without that
context invites the reader to think there is 0.09 of headroom when there is
almost none.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sih
from sih import COLS
from sw_score import score, ci_W, swtest
from smart_horizon import predict

OUT = "out"
NICE = {"x_error": "x", "y_error": "y", "z_error": "z", "satclockerror": "clock"}


def gaussian_null(n, n_draw=4000, seed=1):
    rng = np.random.default_rng(seed)
    w = np.array([swtest(rng.normal(size=n))[0] for _ in range(n_draw)])
    return np.median(w), np.percentile(w, 5), np.percentile(w, 95)


def qq_plot(resid, key, path):
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    for i, c in enumerate(COLS):
        r = resid[c].to_numpy(float)
        r = r[np.isfinite(r)]
        stats.probplot(r, dist="norm", plot=ax[i])
        W, p, H, which = swtest(r)
        ax[i].set_title(f"{NICE[c]}   {which} W={W:.3f}  p={p:.3f}", fontsize=10)
        ax[i].set_xlabel("theoretical quantiles", fontsize=8)
        ax[i].set_ylabel("residual (m)" if i == 0 else "", fontsize=8)
        ax[i].tick_params(labelsize=7)
        ax[i].get_lines()[0].set(marker="o", ms=3.5, mfc="#3b6ea5", mec="none", ls="none")
        ax[i].get_lines()[1].set(color="#c44", lw=1.2)
    fig.suptitle(f"PS-08 {key}: Q-Q of residual (predicted - test)   [Priority 3]",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    L = ["# PS-08 submission report — SMART-HORIZON",
         "",
         "Scored with `sw_score.swtest`, calibrated against the organisers'",
         "`SW_ReferenceData.xlsx` (their stated p = 0.5840; ours = 0.5838).",
         "Residual is **predicted − test**, per Note.pdf.",
         "",
         "`which` records which test MATLAB `swtest.m` would take on that sample:",
         "Shapiro-Francia when kurtosis > 3, Shapiro-Wilk otherwise.",
         ""]

    for key, *_ in sih.PAIRS:
        tr, te, orbit = sih.load_pair(key)
        P = predict(tr, te["utc_time"], orbit=orbit)
        resid = P[COLS].reset_index(drop=True) - te[COLS].reset_index(drop=True)
        per, s = score(resid)
        n = len(te)
        med, lo5, hi95 = gaussian_null(n)

        qq = f"{OUT}/qq_{key}.png"
        qq_plot(resid, key, qq)

        L += [f"## {key}  (orbit {orbit}, train n={len(tr)}, test n={n})", ""]
        L += ["| parameter | n | test | W | p | H | mean (m) | sd (m) | RMS (m) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in per.iterrows():
            L.append(f"| {NICE[r['param']]} | {int(r['n'])} | {r['which']} | "
                     f"{r['W']:.4f} | {r['p']:.4f} | {int(r['H'])} | "
                     f"{r['mean']:+.4f} | {r['sd']:.4f} | {r['rms']:.4f} |")
        L += [f"| **average** | | | **{s['W']:.4f}** | **{s['p']:.4f}** | "
              f"**{s['H']:.2f}** | **{s['mean']:+.4f}** | **{s['sd']:.4f}** | "
              f"**{s['rms']:.4f}** |", ""]
        L += [f"- **Priority 1** — W = **{s['W']:.4f}**, p = {s['p']:.4f}, "
              f"H = {s['n_reject']}/4 parameters reject normality at α = 0.05.",
              f"- **Priority 2** — residual mean {s['mean']:+.4f} m, sd {s['sd']:.4f} m.",
              f"- **Priority 3** — `{os.path.basename(qq)}`.",
              f"- **Context** — a residual that is *exactly* Gaussian at n = {n} scores "
              f"a median W of {med:.3f} (5–95%: {lo5:.3f}–{hi95:.3f}). "
              f"That is the ceiling, not 1.0.", ""]
        L.append("95% bootstrap CI on W per parameter: " + ", ".join(
            f"{NICE[c]} [{ci_W(resid[c].to_numpy())[0]:.3f}, "
            f"{ci_W(resid[c].to_numpy())[1]:.3f}]" for c in COLS) + "\n")

        print(f"{key:6s} W={s['W']:.4f}  p={s['p']:.4f}  H={s['n_reject']}/4  "
              f"mean={s['mean']:+.4f}  sd={s['sd']:.4f}  RMS={s['rms']:.4f}   "
              f"(Gaussian ceiling at n={n}: {med:.3f})   -> {qq}")

    with open(f"{OUT}/submission_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nwrote {OUT}/submission_report.md")


if __name__ == "__main__":
    main()
