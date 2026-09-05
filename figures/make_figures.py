#!/usr/bin/env python3
"""Every figure in Stage-04.   python figures/make_figures.py [name ...]

Lives in `figures/` and writes beside itself. Paths are resolved relative to the
repository root (its parent) so it can be run from anywhere.

Palette: slots 1-3 of a CVD-validated categorical set (blue / orange / aqua),
which clears colour-blind and normal-vision separation on all pairs. Every series
also carries a direct label or legend, so identity is never colour-alone.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))   # .../Stage-04/figures
ROOT = os.path.dirname(HERE)                        # .../Stage-04
os.chdir(ROOT)                 # so data/ resolves for clean.py
sys.path.insert(0, ROOT)       # so the modules beside it import

import clean
import synth
import experiments as E
from clean import COLS
from sw_score import score, swtest
from model import predict, _lomb, _design, _ridge

FIG = HERE                     # write beside this script
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, FAINT = "#0b0b0b", "#52514e", "#8c8b85", "#b9b8b2"
SURF, GRID = "#fcfcfb", "#e6e5e0"
NICE = {"x_error": "x", "y_error": "y", "z_error": "z", "satclockerror": "clock"}
DAY = 86400.0

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 130,
})


def _tidy(*axes):
    for ax in axes:
        ax.set_axisbelow(True)
        ax.tick_params(length=0)


def _save(fig, name, title=None, sub=None):
    """Reserve the header in INCHES, so it never collides regardless of figure height.

    Fraction-based placement breaks on short figures, 0.945 of a 3.4in figure is
    only 0.19in below the top, which the suptitle already occupies.
    """
    head = (0.34 if title else 0.0) + (0.26 if sub else 0.0)
    if head:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h + head)
        h += head
        if title:
            fig.text(0.006, 1 - 0.20 / h, title, fontsize=11.5, ha="left",
                     va="center", weight="bold", color=INK)
        if sub:
            fig.text(0.006, 1 - (0.44 if title else 0.16) / h, sub, fontsize=8.6,
                     color=INK2, ha="left", va="center")
    fig.tight_layout(rect=[0, 0, 1, 1 - head / fig.get_size_inches()[1]])
    p = os.path.join(FIG, name + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT).replace(os.sep, "/"))


# ═══════════════════════════════════════════════════ ACT 1, THE DATA
def f01_raw():
    """What we were actually given."""
    files = ["DATA_GEO_Train.csv", "DATA_MEO_Train.csv", "DATA_MEO_Train2.csv"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.4))
    for ax, f in zip(axes, files):
        d = clean.load(f)
        t = (d.utc_time - d.utc_time.min()).dt.total_seconds() / DAY
        for c, col in zip(COLS, [BLUE, ORANGE, AQUA, INK]):
            ax.plot(t, d[c], "-o", color=col, lw=0.9, ms=2.6, label=NICE[c], alpha=0.9)
        ax.set_title(f"{f[5:-4]}   ·   {len(d)} rows over "
                     f"{(d.utc_time.max()-d.utc_time.min()).total_seconds()/DAY:.2f} days",
                     loc="left")
        ax.set_ylabel("error (m)")
        ax.set_xlim(-0.1, 7.1)
    axes[0].legend(ncol=4, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("days from first sample")
    _tidy(*axes)
    _save(fig, "f01_raw_data", "The three training files, exactly as delivered",
          "Four error channels each. Note the y-axis: the GEO file (top) runs to +-40 m, the MEO files to about +-1 m.")


def f02_coverage():
    """Where the samples actually are."""
    order = ["DATA_GEO_Train.csv", "DATA_MEO_Train.csv", "DATA_MEO_Train2.csv",
             "DATA_GEO_Test.csv", "DATA_MEO_Test.csv", "DATA_MEO_Test2.csv"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for k, f in enumerate(order):
        d = clean.load(f)
        t = (d.utc_time - d.utc_time.min()).dt.total_seconds() / 3600.0
        y = len(order) - 1 - k
        ax.barh(y, 168, height=0.62, color=GRID, zorder=1)
        ax.vlines(t, y - 0.31, y + 0.31, color=BLUE, lw=1.0, zorder=2)
        cov = 100 * np.unique(np.round(t * 2).astype(int)).size / (168 * 2)
        ax.text(170, y, f"{cov:.0f}% covered", va="center", fontsize=8.5,
                color=ORANGE, weight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f[5:-4] for f in order][::-1], fontsize=8.5)
    ax.set_xlim(0, 205); ax.set_xlabel("hours from first sample   (grey = the nominal 7-day window)")
    ax.grid(axis="y", visible=False)
    _tidy(ax)
    _save(fig, "f02_coverage", "No file actually spans seven days",
          "Each blue line is one sample. The grey bar is the 168 hours the brief says we get.")


def f03_duplicates():
    """The MEO files are the same block written twice."""
    rep = clean.dup_report()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    y = np.arange(len(rep))
    ax.barh(y, rep["rows"], height=0.6, color=FAINT, label="rows delivered")
    ax.barh(y, rep["unique"], height=0.6, color=BLUE, label="genuinely distinct")
    ax.set_yticks(y); ax.set_yticklabels(rep["file"], fontsize=8.5)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for i, r in rep.iterrows():
        ax.text(r["rows"] + 4, i, f"{r['unique']}/{r['rows']}", va="center",
                fontsize=8, color=INK2)
    ax.set_xlim(0, 290); ax.set_xlabel("row count")
    ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.set_title("Half of every MEO file is a duplicate", loc="left")

    ax = axes[1]
    d = clean.load("DATA_MEO_Train.csv", clean=False)
    dt = np.diff(d.utc_time.values).astype("timedelta64[s]").astype(float) / 3600.0
    ax.plot(dt, "-o", color=ORANGE, lw=1.1, ms=3, mec=SURF, mew=0.6)
    ax.axhline(0, color=MUTED, lw=0.9)
    i = int(np.argmin(dt))
    ax.annotate("one jump backwards,\nspanning the whole file",
                xy=(i, dt[i]), xytext=(i + 8, dt[i] * 0.55), fontsize=8.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.set_xlabel("row number"); ax.set_ylabel("time step to next row (hours)")
    ax.set_title("...because the block is simply repeated", loc="left")
    _tidy(*axes)
    _save(fig, "f03_duplicates", "The duplication is a copy-paste, so removing it is safe",
          "Exactly one backwards time step per file, spanning its full length. Nothing is lost by de-duplicating.")


def f04_shift():
    """Day 8 is not the same kind of day as days 1-7."""
    tr, te, _ = clean.load_pair("GEO")
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, c in zip(axes, COLS):
        parts = ax.violinplot([tr[c].to_numpy(), te[c].to_numpy()],
                              showextrema=False, widths=0.8)
        for pc, col in zip(parts["bodies"], [BLUE, ORANGE]):
            pc.set_facecolor(col); pc.set_alpha(0.75)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["days 1-7", "day 8"], fontsize=8.5)
        r = te[c].std() / tr[c].std()
        ax.set_title(f"{NICE[c]}   {r:.1f}x wider spread", loc="left", fontsize=9.5,
                     color=ORANGE if r > 2 else INK)
        if ax is axes[0]:
            ax.set_ylabel("error (m)")
    _tidy(*axes)
    _save(fig, "f04_distribution_shift", "The day we must predict is far wilder than the week we are shown",
          "GEO. Spread is 1.6x to 4.2x wider on day 8, i.e. 2.6x to 17x the variance, on every channel (Levene p < 0.003).")


def f05_ramp():
    """The volatility ramp, point 13, expanded."""
    g = E.exp_ramp(verbose=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    ax = axes[0]
    x = g.index.to_numpy()
    for c, col in zip(COLS, [BLUE, ORANGE, AQUA, INK]):
        ax.plot(x, g[c], "-o", color=col, lw=1.8, ms=5, mec=SURF, mew=0.9, label=NICE[c])
    ax.axvspan(7.5, 8.5, color=ORANGE, alpha=0.10)
    ax.text(8, ax.get_ylim()[1] * 0.94, "the day we\nmust predict", ha="center",
            va="top", fontsize=8.3, color=ORANGE)
    ax.set_xlabel("day of the window"); ax.set_ylabel("standard deviation (m)")
    ax.set_xticks(x)
    ax.legend(fontsize=8, ncol=4)
    ax.set_title("Volatility roughly doubles every day from day 5", loc="left")

    ax = axes[1]
    d = clean.load("DATA_GEO_Train.csv")
    t = (d.utc_time - d.utc_time.min()).dt.total_seconds() / DAY
    ax.plot(t, d.y_error, "-o", color=BLUE, lw=1.0, ms=3, mec=SURF, mew=0.5)
    te = clean.load("DATA_GEO_Test.csv")
    tt = 7 + (te.utc_time - te.utc_time.min()).dt.total_seconds() / DAY
    ax.plot(tt, te.y_error, "-o", color=ORANGE, lw=1.0, ms=3, mec=SURF, mew=0.5)
    ax.axvline(7, color=MUTED, lw=1.0, ls="--")
    ax.text(3.4, ax.get_ylim()[1] * 0.9, "training week", color=BLUE, fontsize=9, ha="center")
    ax.text(7.5, ax.get_ylim()[1] * 0.9, "day 8", color=ORANGE, fontsize=9, ha="center")
    ax.set_xlabel("days"); ax.set_ylabel("y error (m)")
    ax.set_title("The same thing, seen directly", loc="left")
    _tidy(*axes)
    _save(fig, "f05_volatility_ramp", "The GEO file gets steadily wilder through the week",
          "This is a ramp, not a glitch. Any model fitted on the quiet early days is being scored on the wild end.")


# ═══════════════════════════════════════════════════ ACT 2, THE PHYSICS
def f06_beat():
    """Why x and y split into two frequencies but z does not."""
    T = synth.ORBITAL_PERIOD["GPS"]
    t = np.linspace(0, 2 * DAY, 4000)
    orb = np.sin(2 * np.pi * t / T)
    earth = np.cos(2 * np.pi * t / synth.T_SID)
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.0), sharex=True)
    axes[0].plot(t / 3600, orb, color=BLUE, lw=1.6)
    axes[0].set_title(f"1. The error rotates with the satellite   ·   period T = {T/3600:.2f} h",
                      loc="left")
    axes[1].plot(t / 3600, earth, color=ORANGE, lw=1.6)
    axes[1].set_title("2. The coordinate frame rotates with the Earth   ·   "
                      f"period T_sid = {synth.T_SID/3600:.2f} h", loc="left")
    axes[2].plot(t / 3600, orb * earth, color=AQUA, lw=1.6)
    b = synth.beat_periods(T)
    axes[2].set_title("3. What we observe is their product, two new tones at "
                      f"{b[0]/3600:.2f} h and {b[1]/3600:.2f} h", loc="left")
    axes[2].set_xlabel("hours")
    for ax in axes:
        ax.set_yticks([])
    _tidy(*axes)
    _save(fig, "f06_beat_frequencies", "Why the x and y axes hide two rhythms, not one",
          "sin(A)·cos(B) = ½[sin(A−B) + sin(A+B)]. The z axis is the rotation axis, so it is untouched and stays at T.")


def f07_periodogram():
    """Does their data actually contain the frequencies physics predicts?"""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, key in zip(axes, ["GEO", "MEO-1", "MEO-2"]):
        tr, _, orbit = clean.load_pair(key)
        t = (tr.utc_time - tr.utc_time.min()).dt.total_seconds().to_numpy(float)
        P = np.linspace(20000, 190000, 700)
        y = tr["z_error"].to_numpy(float); y = y - y.mean()
        pw = []
        for p in P:
            w = 2 * np.pi / p
            c, s = np.cos(w * t), np.sin(w * t)
            pw.append((y @ c) ** 2 / max((c @ c), 1e-9) + (y @ s) ** 2 / max((s @ s), 1e-9))
        pw = np.array(pw); pw = pw / pw.max()
        ax.plot(P / 3600, pw, color=BLUE, lw=1.5)
        if orbit == "GEO":
            ax.axvline(synth.T_SID / 3600, color=ORANGE, lw=1.6, ls="--")
            ax.text(synth.T_SID / 3600, 1.02, " sidereal day", color=ORANGE, fontsize=8)
        else:
            for nm, TT in synth.ORBITAL_PERIOD.items():
                ax.axvline(TT / 3600, color=ORANGE, lw=1.3, ls="--", alpha=0.8)
            ax.text(12.9, 1.02, " MEO orbital periods", color=ORANGE, fontsize=8)
        ax.set_title(f"{key}   ·   z channel", loc="left")
        ax.set_xlabel("period (hours)")
        ax.set_ylim(0, 1.15)
        if ax is axes[0]:
            ax.set_ylabel("relative power")
    _tidy(*axes)
    _save(fig, "f07_periodogram", "The rhythms physics predicts are present in their own data",
          "Orange dashed lines are placed by orbital mechanics alone, before looking at the data.")


def f08_clock():
    """The clock is a random walk with resets, not a periodic signal."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    t = np.arange(0, 8 * DAY, 900.0)
    rng = np.random.default_rng(3)
    walk = synth._clock(t, rng, 0.05, 3, 0.15)
    axes[0].plot(t / DAY, walk, color=BLUE, lw=1.4)
    axes[0].set_title("A simulated atomic clock error", loc="left")
    axes[0].set_xlabel("days"); axes[0].set_ylabel("clock error (m)")

    tr = clean.load("DATA_MEO_Train2.csv")
    tt = (tr.utc_time - tr.utc_time.min()).dt.total_seconds() / DAY
    axes[1].plot(tt, tr.satclockerror, "-o", color=ORANGE, lw=1.0, ms=2.6)
    axes[1].set_title("Their MEO clock channel", loc="left")
    axes[1].set_xlabel("days")

    ax = axes[2]
    vals = tr.satclockerror.round(9)
    ax.hist(vals, bins=40, color=AQUA, edgecolor=SURF, linewidth=0.6)
    ax.set_title(f"only {vals.nunique()} distinct values in {len(vals)} rows", loc="left")
    ax.set_xlabel("clock error (m)"); ax.set_ylabel("count")
    _tidy(*axes)
    _save(fig, "f08_clock_random", "The clock has no rhythm to find, so we forbid the model from looking for one",
          "A clock's error is a random walk (temperature, ageing) plus step resets when the ground segment uploads. Neither is periodic.")


# ═══════════════════════════════════════════════ ACT 3, SYNTHESIS + MODEL
def f09_synth_build():
    """How the synthetic signal is assembled, term by term."""
    t = np.arange(0, 7 * DAY, 900.0)
    T = synth.ORBITAL_PERIOD["GPS"]
    per = synth.beat_periods(T)
    rng = np.random.default_rng(4)
    bias = np.full_like(t, 0.9)
    harm = 0.45 * np.sin(2 * np.pi * t / per[0]) + 0.30 * np.sin(2 * np.pi * t / per[1] + 1.1)
    drift = 0.25 * (t - t.min()) / (t.max() - t.min())
    noise = rng.normal(0, 0.05, len(t))
    parts = [("1. a fixed offset (harmless, it cancels)", bias, FAINT),
             ("2. orbital harmonics at the beat periods, the physics", harm, BLUE),
             ("3. slow drift between ground-segment uploads", drift, AQUA),
             ("4. measurement noise", noise, MUTED)]
    fig, axes = plt.subplots(5, 1, figsize=(11, 6.6), sharex=True)
    for ax, (lbl, y, col) in zip(axes, parts):
        ax.plot(t / DAY, y, color=col, lw=1.3)
        ax.set_title(lbl, loc="left", fontsize=9)
        ax.set_yticks([])
    axes[4].plot(t / DAY, bias + harm + drift + noise, color=ORANGE, lw=1.4)
    axes[4].set_title("= the x-error signal we generate", loc="left", fontsize=9.5)
    axes[4].set_yticks([]); axes[4].set_xlabel("days")
    _tidy(*axes)
    _save(fig, "f09_synthesis", "How we build a signal whose true answer we know",
          "Every term is a documented physical effect. Nothing here is fitted to the delivered data.")


def f10_recovery_example():
    """One synthetic window: hide day 8, ask for it, compare."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.0))
    for ax, (orbit, seed) in zip(axes.ravel(),
                                 [("GEO", 2), ("GEO", 5), ("MEO", 1), ("MEO", 7)]):
        tr, q, truth, meta = synth.make_window(orbit, seed=seed)
        P = predict(tr, q, orbit=orbit)
        tt = meta["t_train_s"] / DAY
        tq = meta["t_query_s"] / DAY
        ax.plot(tt, tr.x_error, "o", color=FAINT, ms=2.6, label="7 days we are given")
        ax.plot(tq, truth[:, 0], "-", color=INK, lw=2.4, label="day-8 truth (hidden)")
        ax.plot(tq, P.x_error, "-", color=BLUE, lw=2.0, label="model prediction")
        ax.axhline(tr.x_error.mean(), color=ORANGE, lw=1.6, ls="--", label="7-day average")
        ax.axvline(7, color=MUTED, lw=0.9, ls=":")
        r = np.sqrt(((P.x_error.to_numpy() - truth[:, 0]) ** 2).mean())
        r0 = np.sqrt(((tr.x_error.mean() - truth[:, 0]) ** 2).mean())
        ax.set_title(f"{orbit}  ·  model {r:.3f} m   vs   average {r0:.3f} m", loc="left")
        ax.set_xlim(5.2, 8.05)
        ax.set_ylabel("x error (m)")
    axes[0, 0].legend(fontsize=7.6, loc="upper left", ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("days")
    _tidy(*axes.ravel())
    _save(fig, "f10_recovery_examples", "The test: hide day 8, ask the model for it, compare to the truth we planted",
          "Only the last 1.8 days are shown. The model has never seen the black curve.")


def f11_recovery_dist():
    """Recovery across many windows."""
    R = E.exp_recovery(n=60, verbose=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for ax, o in zip(axes, ["GEO", "MEO"]):
        g = R[R.orbit == o]
        bins = np.linspace(0, max(g.zero.max(), g.model.max()) * 1.02, 34)
        ax.hist(g.zero, bins=bins, color=FAINT, label="do nothing", alpha=0.95)
        ax.hist(g.window_mean, bins=bins, color=ORANGE, label="7-day average", alpha=0.8)
        ax.hist(g.model, bins=bins, color=BLUE, label="model", alpha=0.9)
        imp = 100 * (1 - g.model.mean() / g.zero.mean())
        ax.set_title(f"{o}   ·   {imp:.0f}% better than doing nothing   "
                     f"({len(g)} windows)", loc="left")
        ax.set_xlabel("day-8 error, metres (lower is better)")
        ax.set_ylabel("number of windows" if o == "GEO" else "")
        ax.legend(fontsize=8)
    _tidy(*axes)
    _save(fig, "f11_recovery_distribution", "Not one lucky window, 60 of them, per orbit class",
          "Each window is an independent signal with a different period, phase, drift and noise draw.")


def f12_sensitivity():
    """Where does the model break?"""
    R = E.exp_sensitivity(n=30, verbose=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
    for ax, knob, lbl, col in [(axes[0], "noise", "measurement noise (m)", BLUE),
                               (axes[1], "artifact", "fraction of rows corrupted", ORANGE)]:
        g = R[R.knob == knob].groupby("value")[["zero", "model"]].mean()
        pct = 100 * (1 - g.model / g.zero)
        ax.plot(g.index, pct, "-o", color=col, lw=2, ms=6, mec=SURF, mew=1)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_xlabel(lbl); ax.set_ylabel("% better than doing nothing")
        ax.set_title({"noise": "Noise: the model degrades gracefully",
                      "artifact": "The sign artefact: catastrophic"}[knob], loc="left")
        if knob == "artifact":
            ax.text(0.25, -300, "worse than\ndoing nothing", color=ORANGE, fontsize=9,
                    ha="center")
    _tidy(*axes)
    _save(fig, "f12_sensitivity", "What the model survives, and what destroys it",
          "Left: quadrupling the noise costs a few points. Right: corrupting one row in ten sends it below useless.")


def f13_artifact_inject():
    """Inject their GEO artefact into a signal that was clean a moment ago."""
    tr, q, truth, meta = synth.make_window("GEO", seed=3)
    P0 = predict(tr, q, orbit="GEO")
    tr2 = tr.copy()
    V, idx = synth.inject_artifact(tr2[COLS].to_numpy(float), frac=0.35, seed=3)
    for j, c in enumerate(COLS):
        tr2[c] = V[:, j]
    P1 = predict(tr2, q, orbit="GEO")
    tt, tq = meta["t_train_s"] / DAY, meta["t_query_s"] / DAY
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for row, (d, P, lbl, col) in enumerate([(tr, P0, "clean synthetic signal", BLUE),
                                            (tr2, P1, "same signal + the GEO artefact", ORANGE)]):
        ax = axes[row, 0]
        ax.plot(tt, d.x_error, "-o", color=col, lw=0.8, ms=2.6, alpha=0.9)
        ax.set_title(lbl, loc="left")
        ax.set_ylabel("x error (m)")
        if row == 1:
            ax.set_xlabel("days")
        ax = axes[row, 1]
        ax.plot(tq, truth[:, 0], "-", color=INK, lw=2.4, label="day-8 truth")
        ax.plot(tq, P.x_error, "-", color=col, lw=2.0, label="prediction")
        r = np.sqrt(((P.x_error.to_numpy() - truth[:, 0]) ** 2).mean())
        ax.set_title(f"day-8 prediction   ·   RMS {r:.3f} m", loc="left")
        ax.legend(fontsize=8)
        if row == 1:
            ax.set_xlabel("days")
    _tidy(*axes.ravel())
    _save(fig, "f13_artifact_injection",
          "Proof by construction: the artefact alone is what breaks GEO",
          "Identical underlying physics in both rows. The only change is the alternating sign pattern copied from their GEO file.")


# ═══════════════════════════════════════════════════ ACT 4, RESULTS
def f14_sih_results():
    R = E.exp_sih(verbose=False)
    order = ["do nothing", "last value", "7-day average", "CHEAT: test mean", "SMART-HORIZON"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, key in zip(axes, ["GEO", "MEO-1", "MEO-2"]):
        g = R[R.pair == key].set_index("method").loc[order]
        cols = [FAINT, FAINT, FAINT, AQUA, BLUE]
        b = ax.barh(range(len(order)), g.RMS.to_numpy(), color=cols, height=0.6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8.2)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        for r, v in zip(b, g.RMS.to_numpy()):
            ax.text(v + g.RMS.max() * 0.03, r.get_y() + r.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, g.RMS.max() * 1.32)
        ax.set_xlabel("error, metres")
        gain = 100 * (1 - g.RMS.loc["SMART-HORIZON"] / g.RMS.loc["do nothing"])
        note = "nothing helps" if key == "GEO" else f"{gain:.0f}% better"
        ax.set_title(f"{key} ,  {note}", loc="left")
    _tidy(*axes)
    _save(fig, "f14_results", "On MEO the model wins. On GEO every method scores the same.",
          "'CHEAT' is shown the answer key's own average, no legitimate model can use it. On MEO we beat it anyway.")


def f15_meo_pred():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for ax, key, ch in [(axes[0], "MEO-1", "z_error"), (axes[1], "MEO-2", "x_error")]:
        tr, te, orb = clean.load_pair(key)
        P = predict(tr, te.utc_time, orbit=orb)
        t = (te.utc_time - te.utc_time.min()).dt.total_seconds() / 3600.0
        ax.plot(t, te[ch], "o", color=INK, ms=6.5, label="truth", zorder=3, mec=SURF, mew=1)
        ax.plot(t, P[ch], "-o", color=BLUE, lw=2, ms=4, mec=SURF, mew=0.7,
                label="model", zorder=2)
        ax.axhline(tr[ch].mean(), color=ORANGE, lw=2, ls="--", label="7-day average")
        ax.set_title(f"{key}  ·  {NICE[ch]} error", loc="left")
        ax.set_xlabel("hours into day 8"); ax.set_ylabel("error (m)")
        ax.legend(fontsize=8)
    _tidy(*axes)
    _save(fig, "f15_meo_predictions", "The model follows the shape of day 8, not just its level",
          "Left is a clean case, right a noisy one. Both are shown; neither is selected.")


def f16_geo_flip():
    d = clean.load("DATA_GEO_Test.csv")
    V = d[COLS].to_numpy(float)
    t = (d.utc_time - d.utc_time.min()).dt.total_seconds() / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6))
    ax = axes[0]
    for c, col in zip(COLS, [BLUE, ORANGE, AQUA, INK]):
        ax.plot(t, d[c], "-o", lw=1.2, ms=3.4, color=col, label=NICE[c])
    nrm = np.linalg.norm(V[:, :3], axis=1)
    for i in range(len(V) - 1):
        a, b = V[i], V[i + 1]
        if np.linalg.norm(a) > 3 and np.linalg.norm(b) > 3:
            if a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) < -0.7:
                ax.axvspan(t.iloc[i], t.iloc[i + 1], color=ORANGE, alpha=0.13, zorder=0)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_ylabel("error (m)"); ax.set_xlabel("hours")
    ax.set_title("Shaded = consecutive readings that are near-exact opposites "
                 "(worst cosine −0.9994)", loc="left")

    ax = axes[1]
    big = np.argsort(-nrm)[:24]
    big = np.sort(big)
    sg = np.sign(V[big, 0])
    ax.bar(range(len(big)), sg, color=[BLUE if s > 0 else ORANGE for s in sg], width=0.7)
    ax.set_xticks(range(len(big))); ax.set_xticklabels(big, fontsize=7)
    ax.set_yticks([-1, 1]); ax.set_yticklabels(["−", "+"])
    ax.set_xlabel("row number of the 24 largest excursions")
    ax.set_title("Their signs, in order: + − + − + − … 24 alternations out of 24  "
                 "(odds by chance ≈ 1 in 8 million)", loc="left")
    ax.grid(axis="x", visible=False)
    _tidy(*axes)
    _save(fig, "f16_geo_artifact", "The GEO file alternates sign with machine regularity",
          "All four channels flip together. Position and clock come from unrelated physical mechanisms and cannot do that.")


# ═══════════════════════════════════════════════════ ACT 5, THE METRIC
def f17_metric():
    R = E.exp_metric(verbose=False)
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    y = np.arange(len(R))[::-1]
    cols = []
    for lbl in R.prediction:
        cols.append(ORANGE if "noise" in lbl else (AQUA if "CHEAT" in lbl else FAINT))
    ax.barh(y, R.W, color=cols, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(R.prediction, fontsize=8.4)
    for yy, (w, r) in zip(y, zip(R.W, R.RMS)):
        ax.text(w + 0.006, yy, f"W={w:.4f}    (error {r:.2f} m)", va="center",
                fontsize=8, color=INK2)
    ax.set_xlim(0, 1.28); ax.set_xlabel("Shapiro-Wilk W ,  their Priority 1 score, higher is better")
    ax.grid(axis="y", visible=False)
    _tidy(ax)
    _save(fig, "f17_metric_broken",
          "Their score rewards noise and is blind to accuracy",
          "On their GEO file: a model that is 99% accurate scores exactly the same as doing nothing. Pure random noise scores 0.98.")


def f18_ceiling():
    rng = np.random.default_rng(1)
    ours = {6: 0.9067, 18: 0.7977, 69: 0.7715}
    labels = {6: "MEO-1  (n=6)", 18: "MEO-2  (n=18)", 69: "GEO  (n=69)"}
    ns = [6, 18, 69]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    for k, n in enumerate(ns):
        w = np.array([swtest(rng.normal(size=n))[0] for _ in range(3000)])
        lo, hi, med = np.percentile(w, 5), np.percentile(w, 95), np.median(w)
        y = len(ns) - 1 - k
        ax.plot([lo, hi], [y, y], color=AQUA, lw=8, solid_capstyle="round", zorder=1)
        ax.plot([med], [y], "o", color=INK, ms=7, zorder=3, mec=SURF, mew=1.2)
        ax.plot([ours[n]], [y], "D", color=ORANGE, ms=8, zorder=4, mec=SURF, mew=1.2)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([labels[n] for n in ns][::-1], fontsize=9)
    ax.set_ylim(-0.95, len(ns) - 0.3); ax.set_xlim(0.60, 1.03)
    ax.set_xlabel("Shapiro-Wilk W")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=INK, label="what a PERFECT answer scores (median)")
    ax.plot([], [], "-", color=AQUA, lw=8, label="range a perfect answer lands in (5–95%)")
    ax.plot([], [], "D", color=ORANGE, label="our score")
    ax.legend(fontsize=8, loc="lower left", ncol=1, bbox_to_anchor=(-0.005, -0.03))
    _tidy(ax)
    _save(fig, "f18_ceiling", "A flawless answer cannot score 1.0, and at n=6 it is mostly luck",
          "Green is where a model with perfectly random errors lands. With 6 test points that is anywhere from 0.79 to 0.98.")


def f19_qq():
    """Q-Q plots. Drawn by hand rather than via scipy.probplot, which stamps its
    own 'Probability Plot' title over ours."""
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.0))
    for row, (key, *_) in enumerate(clean.PAIRS):
        tr, te, orb = clean.load_pair(key)
        P = predict(tr, te.utc_time, orbit=orb)
        res = P[COLS].reset_index(drop=True) - te[COLS].reset_index(drop=True)
        for j, c in enumerate(COLS):
            ax = axes[row, j]
            r = np.sort(res[c].to_numpy(float))
            r = r[np.isfinite(r)]
            n = len(r)
            q = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
            sl, ic = np.polyfit(q, r, 1)
            ax.plot(q, sl * q + ic, "-", color=ORANGE, lw=1.3, zorder=1)
            ax.plot(q, r, "o", ms=3.6, mfc=BLUE, mec="none", zorder=2)
            W, p, H, which = swtest(r)
            ax.set_title(f"{key} · {NICE[c]}    W = {W:.3f}", fontsize=9, loc="left")
            if row == 2:
                ax.set_xlabel("theoretical quantiles", fontsize=8)
            if j == 0:
                ax.set_ylabel("residual (m)", fontsize=8)
    _tidy(*axes.ravel())
    _save(fig, "f19_qq", "Q-Q plots of our residuals  [their Priority 3]",
          "Points on the line mean the errors are the random bell-curve noise the brief asks for.")


def f20_benchmark():
    R = E.exp_benchmark(verbose=False)
    piv = R.pivot_table(index="method", columns="pair", values="rms")
    piv["MEO"] = piv[["MEO-1", "MEO-2"]].mean(axis=1)
    piv = piv.sort_values("MEO", ascending=False)
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    y = np.arange(len(piv))
    cols = [BLUE if m == "SMART-HORIZON" else FAINT for m in piv.index]
    b = ax.barh(y, piv["MEO"], color=cols, height=0.66)
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=8.4)
    for r, v in zip(b, piv["MEO"]):
        ax.text(v + 0.006, r.get_y() + r.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8, color=INK2)
    ax.set_xlabel("day-7 error on the MEO files, metres (lower is better)")
    ax.set_xlim(0, piv["MEO"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    _tidy(ax)
    _save(fig, "f20_benchmark", "Sixteen methods, ranked using only the delivered data",
          "Protocol: fit on days 1–6 of their own file, score on day 7. No outside data.")




def f21_ml_plateau():
    """Ordinary ML on the real task: it plateaus at the baseline."""
    R = E.exp_ml_fails(n_windows=400, verbose=False)
    R = R.sort_values("rms", ascending=False)
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    y = np.arange(len(R))
    cmap = {"baseline": FAINT, "learned": ORANGE, "physics": BLUE}
    b = ax.barh(y, R.rms, color=[cmap[k] for k in R.kind], height=0.66)
    ax.set_yticks(y); ax.set_yticklabels(R.model, fontsize=8.6)
    for r, v in zip(b, R.rms):
        ax.text(v + 0.014, r.get_y() + r.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8, color=INK2,
                bbox=dict(facecolor=SURF, edgecolor="none", pad=1.4))
    bl = R[R.kind == "baseline"].rms.min()
    ax.axvline(bl, color=INK, lw=1.2, ls="--")
    ax.text(bl + 0.01, len(R) - 0.4, "the trivial baseline\nevery learner must beat",
            fontsize=8.2, color=INK, va="top")
    ax.set_xlabel("day-8 error, metres (lower is better)")
    ax.set_xlim(0, R.rms.max() * 1.16)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=cmap[k], label=l) for k, l in
                       [("learned", "trained machine learning"),
                        ("baseline", "trivial baseline"),
                        ("physics", "physics model")]],
              fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, columnspacing=1.6)
    _tidy(ax)
    _save(fig, "f21_ml_plateau",
          "Six standard ML models, trained properly, and they stall at the baseline",
          "400 windows, 27,000 training rows, 19 engineered features. The learners are not weak; the five columns simply do not contain what decides day 8.")


def f22_how_it_works():
    """The model, as a diagram."""
    tr, q, truth, meta = synth.make_window("MEO", seed=11)
    t = meta["t_train_s"]; tq = meta["t_query_s"]
    y = tr["x_error"].to_numpy(float)
    cut = t.min() + (t.max() - t.min()) * 6.0 / 7.0
    a, b = t <= cut, t > cut

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ax = axes[0]
    ax.plot(t[a] / DAY, y[a], "o", color=BLUE, ms=3, label="days 1–6: fit on this")
    ax.plot(t[b] / DAY, y[b], "o", color=ORANGE, ms=3.4, label="day 7: held out")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("1. Split the window the caller gave us", loc="left")
    ax.set_xlabel("days"); ax.set_ylabel("x error (m)")

    ax = axes[1]
    P = _lomb(t, tr["z_error"].to_numpy(float), 38000.0, 54000.0) or 43082.0
    cands = {"flat (mean)": None, f"harmonic K=1": 1, "harmonic K=2": 2, "harmonic K=3": 3}
    errs, names = [], []
    for nm, K in cands.items():
        if K is None:
            p = np.full(b.sum(), y[a].mean())
        else:
            per = synth.beat_periods(P)
            c = _ridge(_design(t[a], per, K), y[a])
            p = _design(t[b], per, K) @ c
        errs.append(float(np.sqrt(np.mean((p - y[b]) ** 2)))); names.append(nm)
    order = np.argsort(errs)[::-1]
    cols = [BLUE if i == order[-1] else FAINT for i in range(len(errs))]
    cols = [BLUE if errs[i] == min(errs) else FAINT for i in range(len(errs))]
    ax.barh(range(len(errs)), [errs[i] for i in order], height=0.6,
            color=[cols[i] for i in order])
    ax.set_yticks(range(len(errs)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8.4)
    ax.set_xlabel("error on the held-out day 7 (m)")
    ax.set_title("2. Every candidate competes on day 7", loc="left")
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    Pr = predict(tr, q, orbit="MEO")
    ax.plot(t / DAY, y, "o", color=FAINT, ms=2.6, label="all 7 days")
    ax.plot(tq / DAY, truth[:, 0], "-", color=INK, lw=2.2, label="day-8 truth")
    ax.plot(tq / DAY, Pr.x_error, "-", color=BLUE, lw=2, label="prediction")
    ax.axvline(7, color=MUTED, lw=0.9, ls=":")
    ax.set_xlim(5.5, 8.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("3. Refit the winner on all 7 days, extrapolate", loc="left")
    ax.set_xlabel("days")
    _tidy(*axes)
    _save(fig, "f22_how_it_works", "How the model works, the window grades itself",
          "No stored training set is consulted at any point. The only model-selection signal is the caller's own day 7.")


ALL = {n: f for n, f in sorted(globals().items()) if n[0] == "f" and n[1].isdigit()}

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    want = sys.argv[1:] or list(ALL)
    for n in want:
        fn = ALL.get(n) or ALL.get([k for k in ALL if k.startswith(n)][0])
        print(f"[{fn.__name__}]")
        fn()
