#!/usr/bin/env python3
"""Figures for the judge-facing summary.  python plots.py -> out/fig*.png

Four figures, one argument each:
  fig1  the GEO file is not physical      (side by side against real GEO)
  fig2  the model works on MEO            (predictions over their own test day)
  fig3  how much better, per pair         (RMS by method, faceted)
  fig4  their metric has a ceiling        (what a PERFECT answer scores)

Palette: slots 1-3 of the validated categorical set (blue/orange/aqua), which
clears CVD and normal-vision separation on all pairs. Aqua sits below 3:1 on the
light surface, so every series carries a direct label -- the relief rule -- and
identity is never colour-alone.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sih
from sih import COLS
from sw_score import score, swtest
from smart_horizon import predict

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8c8b85"
SURF = "#fcfcfb"
GRID = "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.size": 9, "axes.titlesize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def _tidy(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- figure 1
def fig1():
    """Their GEO file next to real GEO data. Same quantity, same units."""
    tr = sih.load("DATA_GEO_Test.csv")
    t = (tr.utc_time - tr.utc_time.min()).dt.total_seconds() / 3600.0
    m = t <= 12
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))

    ax[0].plot(t[m], tr.y_error[m], "-o", color=ORANGE, lw=1.6, ms=4.5,
               mec=SURF, mew=0.8)
    ax[0].set_title("PS-08 GEO file, first 12 hours", loc="left")
    ax[0].set_xlabel("hours"); ax[0].set_ylabel("y error (m)")
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].text(0.98, 0.05, "flips sign almost every sample",
               transform=ax[0].transAxes, ha="right", color=ORANGE, fontsize=9)

    d = np.load("../taskB_data.npz", allow_pickle=True)
    XYZ, S = d["XYZ"], np.array([str(x) for x in d["sats"]])
    i = int(np.where(S == "J07")[0][0])            # a true GEO, verified finite here
    seg = XYZ[i][0:48, 1]
    ts = np.arange(len(seg)) * 0.25
    ax[1].plot(ts, seg, "-o", color=BLUE, lw=1.6, ms=4.5, mec=SURF, mew=0.8)
    ax[1].set_title("A real GEO satellite (J07), same 12 hours", loc="left")
    ax[1].set_xlabel("hours")
    ax[1].text(0.98, 0.05, "smooth, as orbital physics requires",
               transform=ax[1].transAxes, ha="right", color=BLUE, fontsize=9)

    # SHARED y-axis: the size difference is as much the point as the shape.
    lim = max(abs(tr.y_error[m]).max(), abs(seg).max()) * 1.12
    for a in ax:
        a.set_ylim(-lim, lim)
        a.axhline(0, color=MUTED, lw=0.8, zorder=0)
    # The real series is nearly flat at this scale -- that is itself the finding --
    # so an inset carries its actual shape.
    ins = ax[1].inset_axes([0.30, 0.62, 0.40, 0.30])
    ins.plot(ts, seg, "-", color=BLUE, lw=1.5)
    ins.set_title("same data, zoomed 25x", fontsize=7.5, color=MUTED, pad=3)
    ins.tick_params(labelsize=6.5, length=0, colors=MUTED)
    ins.grid(color=GRID, lw=0.5)
    for sp in ins.spines.values():
        sp.set_color(GRID)
    ins.set_facecolor(SURF)

    for a in ax:
        _tidy(a)
    fig.suptitle("A satellite cannot move 40 m and back in four minutes",
                 fontsize=11, x=0.007, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("out/fig1_geo_broken.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2():
    """Model vs truth on their own MEO test day."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, ch in [(axes[0], "MEO-1", "z_error"), (axes[1], "MEO-2", "x_error")]:
        tr, te, orb = sih.load_pair(key)
        P = predict(tr, te.utc_time, orbit=orb)
        t = (te.utc_time - te.utc_time.min()).dt.total_seconds() / 3600.0
        ax.plot(t, te[ch], "o", color=INK, ms=6, label="truth", zorder=3,
                mec=SURF, mew=0.9)
        ax.plot(t, P[ch], "-o", color=BLUE, lw=2, ms=4.5, mec=SURF, mew=0.8,
                label="SMART-HORIZON", zorder=2)
        ax.axhline(tr[ch].mean(), color=ORANGE, lw=2, ls="--",
                   label="7-day average", zorder=1)
        ax.set_title(f"{key}  ·  {ch.replace('_error',' error')}", loc="left")
        ax.set_xlabel("hours into day 8"); ax.set_ylabel("error (m)")
        ax.legend(loc="best", fontsize=8)
        _tidy(ax)
    fig.suptitle("On their MEO data the model tracks the real shape, not just the level",
                 fontsize=11, x=0.007, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("out/fig2_meo_works.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig3():
    """RMS by method, one panel per pair (scales differ 100x -- never one axis)."""
    R = pd.read_csv("out/scores.csv")
    order = ["zero", "last value", "window mean", "SMART-HORIZON"]
    nice = {"zero": "do nothing", "last value": "last value",
            "window mean": "7-day average", "SMART-HORIZON": "SMART-HORIZON"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, key in zip(axes, ["GEO", "MEO-1", "MEO-2"]):
        g = R[R.pair == key].set_index("method").loc[order]
        cols = [MUTED] * 3 + [BLUE]
        b = ax.barh(range(4), g.RMS.to_numpy(), color=cols, height=0.62)
        ax.set_yticks(range(4)); ax.set_yticklabels([nice[o] for o in order], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for r, v in zip(b, g.RMS.to_numpy()):
            ax.text(v + g.RMS.max() * 0.03, r.get_y() + r.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, g.RMS.max() * 1.28)
        ax.set_xlabel("error, metres (lower is better)")
        note = "nothing helps" if key == "GEO" else f"{100*(1-g.RMS.iloc[3]/g.RMS.iloc[0]):.0f}% better"
        ax.set_title(f"{key}   —   {note}", loc="left")
        _tidy(ax)
    fig.suptitle("MEO: the model wins. GEO: every method scores the same, because the file has no signal.",
                 fontsize=11, x=0.007, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("out/fig3_rms.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig4():
    """What a PERFECT answer scores on their metric, at their sample sizes."""
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ns = [6, 18, 69]
    ours = {6: 0.9067, 18: 0.7977, 69: 0.7715}
    labels = {6: "MEO-1 (n=6)", 18: "MEO-2 (n=18)", 69: "GEO (n=69)"}
    for k, n in enumerate(ns):
        w = np.array([swtest(rng.normal(size=n))[0] for _ in range(3000)])
        lo, hi, med = np.percentile(w, 5), np.percentile(w, 95), np.median(w)
        y = len(ns) - 1 - k
        ax.plot([lo, hi], [y, y], color=AQUA, lw=7, solid_capstyle="round",
                alpha=0.9, zorder=1)
        ax.plot([med], [y], "o", color=INK, ms=7, zorder=3, mec=SURF, mew=1.2)
        ax.plot([ours[n]], [y], "D", color=ORANGE, ms=8, zorder=4, mec=SURF, mew=1.2)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([labels[n] for n in ns][::-1], fontsize=9, color=INK2)
    ax.set_ylim(-0.9, len(ns) - 0.3)
    ax.set_xlabel("Shapiro-Wilk W  (their Priority 1 score — higher is better)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.60, 1.02)
    ax.text(0.985, len(ns) - 1 + 0.42, "a PERFECT answer\nlands in here",
            color=AQUA, fontsize=8.5, ha="right", va="bottom", weight="bold")
    ax.plot([], [], "o", color=INK, label="median of a perfect (pure-noise) answer")
    ax.plot([], [], "D", color=ORANGE, label="our score")
    ax.plot([], [], "-", color=AQUA, lw=7, label="5–95% range of a perfect answer")
    ax.legend(loc="lower left", fontsize=8, ncol=3, columnspacing=1.1,
              handletextpad=0.5, bbox_to_anchor=(-0.02, -0.02))
    _tidy(ax)
    fig.suptitle("Their score is mostly luck at these sample sizes",
                 fontsize=11, x=0.007, ha="left", weight="bold")
    fig.text(0.007, 0.885, "A flawless model cannot score 1.0. With 6 test points it lands anywhere "
             "from 0.79 to 0.98 by chance alone.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig("out/fig4_ceiling.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
        print("wrote", f.__doc__.splitlines()[0])
    print("-> out/fig1_geo_broken.png fig2_meo_works.png fig3_rms.png fig4_ceiling.png")
