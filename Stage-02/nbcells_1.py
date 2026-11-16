"""Stage-02 notebook, cells part 1: the corpus, and what the error looks like."""


def add(md, co):

    md(r"""
# Stage 2 — Task A: correcting the broadcast, and asking whether it matters

**Stage 1 ended with a verdict: the delivered PS-08 files cannot support the stated
task.** Half the MEO rows are duplicates, no file spans seven days, the GEO file
contains an artefact that appears zero times in two million real epochs, and even a
*perfect* seven-day file is missing the quantities that drive the error.

This notebook is what we did next. It has two halves.

**Half one — build something real, and find out what the error actually is.**
We stop using the delivered files and build a 244-day corpus from primary sources.
Then we ask, in order: what does the error look like, when does it repeat, how much
of it is even the satellite's fault, and what is the physical explanation for the
rest.

**Half two — the honest test.** Task A produces a model that cuts the error a long
way. So we ask the only question that really matters: *does it move a receiver's
position fix?* We build a single-point-positioning simulator, feed it the real
broadcast errors, and read the answer off in metres on the ground.

Every number here is computed in this notebook from `../full2.parquet` and the
artefacts in this folder. Where a number differs from the write-up, both are shown.
""")

    md("## Setup")
    co(r"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats

CORPUS = Path("../full2.parquet")
CL     = 299792458.0                      # m/s
NAME   = {"G":"GPS","E":"Galileo","C":"BeiDou","J":"QZSS"}
SCOL   = {"G":"#1f77b4","E":"#d62728","C":"#2ca02c","J":"#9467bd"}

mpl.rcParams.update({
    "figure.dpi":110, "savefig.dpi":110,
    "axes.grid":True, "grid.alpha":0.25, "grid.linewidth":0.6,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.titlesize":10, "axes.labelsize":9,
    "xtick.labelsize":8, "ytick.labelsize":8, "legend.fontsize":8,
    "figure.titlesize":11})

def show(df, caption=None):
    if caption: print(caption)
    display(df)

d = pd.read_parquet(CORPUS, columns=[
    "epoch","sat","sys","orbit_class","dR","dA","dC","dClk_ns","dClk_debias_ns",
    "dx","dy","dz","px","py","pz","age_s","arc_toe","u_arg"])
d["clk"]  = d.dClk_ns*1e-9*CL             # broadcast clock - precise clock, in metres
d["clkd"] = d.dClk_debias_ns*1e-9*CL      # per-satellite-per-day median removed

print(f"corpus : {len(d):,} epochs   {d.sat.nunique()} spacecraft   "
      f"{d.epoch.min().date()} -> {d.epoch.max().date()}")
print(f"         {(d.epoch.max()-d.epoch.min()).days+1} days on a uniform 900 s grid")
print("         classes: " + ", ".join(f"{k} {v:,}" for k,v in d.orbit_class.value_counts().items()))
""")

    md(r"""
---
# Part 1 — Building the corpus

Stage 1 said the delivered data was unusable. That leaves two options: give up, or
measure the same physical quantity ourselves. The quantity is not exotic —
**error = broadcast ephemeris − precise ephemeris** — and both halves are public.

| stage | script | what it does |
|---|---|---|
| 1 | `fetch_all.py` | downloads `BRDM00DLR` broadcast nav + `JAX0MGXFIN` precise orbits from NASA CDDIS |
| 2 | `build_errors.py` + `gnss.py` | evaluates the broadcast message per each system's ICD, differences it against SP3 |
| 3 | `windows.py` | memory-safe sliding-window access |

**Why JAXA specifically.** It is the only final MGEX product that contains GEO
satellites. CODE, GRG, IAC, WUM and GFZ all start BeiDou at C06 and carry no GEO at
all. No JAXA means no J07/J08 and no GEO data — and GEO is half the problem
statement.

**No clock files are downloaded.** The clock column inside SP3 agrees with the
separate 30-second CLK product to **1.6 picoseconds**, roughly 1000x below the signal
being measured, so the extra download would buy nothing.

The corpus is 14,000x larger than the delivered files, of the identical physical
quantity, on a uniform grid, with every satellite labelled. But it is not clean out
of the box, and the next three cells are the problems we had to find first.
""")

    md("### What we got — and the first thing wrong with it")
    co(r"""
n_month = d.groupby([d.epoch.dt.to_period("M"), "sat"]).size().reset_index()
per_month = n_month.groupby("epoch").sat.nunique()
months  = len(per_month)
allmon  = n_month.groupby("sat").epoch.nunique()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
ax = axes[0]
ax.bar([str(p) for p in per_month.index], per_month.values, color="#4c72b0", alpha=0.85)
ax.set_ylabel("spacecraft present"); ax.set_title("Constellation composition drifts")
ax.tick_params(axis="x", rotation=45)

ax = axes[1]
ax.hist(allmon, bins=np.arange(0.5, months+1.5), color="#c44e52", alpha=0.85)
ax.axvline(months, color="black", ls="--", lw=1)
ax.set_xlabel("months in which the spacecraft appears"); ax.set_ylabel("spacecraft")
ax.set_title(f"only {int((allmon==months).sum())} of {len(allmon)} appear in all {months}")
fig.tight_layout(); plt.show()

print(f"spacecraft present in every month : {int((allmon==months).sum())} of {len(allmon)}")
""")

    md(r"""
**What the plot shows.** The left panel is not a data-loading bug — the constellation
genuinely changes month to month. Satellites are launched, retired and taken out of
service, so any month-over-month comparison is partly measuring *which satellites
were up*, not how they behaved.

**What we did about it.** Every comparison across time in this notebook is made
either within a single satellite or on a fixed satellite set. The 79-of-103 figure is
also the reason "just fit one model per satellite" is weaker than it sounds: a
quarter of the spacecraft are not present for the whole period.
""")

    md("### Problem 2 — three PRNs are each two different spacecraft")
    co(r"""
r_km = np.sqrt(d.px**2 + d.py**2 + d.pz**2)/1e3
sub  = d[d.sat.isin(["C13","C38","C39"])].copy()
sub["r_km"] = r_km[sub.index]
daily = sub.groupby([sub.epoch.dt.date, "sat"]).r_km.median().unstack()

fig, ax = plt.subplots(figsize=(9, 3.0))
for s in daily.columns:
    ax.plot(pd.to_datetime(daily.index), daily[s], lw=1.2, label=s)
ax.axhline(35000, color="black", ls=":", lw=1)
ax.set_ylabel("daily median orbital radius (km)")
ax.set_title("One PRN, two spacecraft: the radius steps 42,164 km -> 27,906 km")
ax.legend(); fig.tight_layout(); plt.show()

for s in ["C13","C38","C39"]:
    v  = daily[s].dropna()
    hi, lo = v[v > 35000], v[v < 35000]
    if len(hi) and len(lo):
        print(f"{s}: GSO era {hi.mean():,.0f} km (through {hi.index.max()})  ->  "
              f"MEO era {lo.mean():,.0f} km (from {lo.index.min()})")
""")

    md(r"""
**What the plot shows.** A satellite's orbital radius is fixed by its orbit. It
cannot step by 14,000 km. **All three PRNs flip on the same date**, which is the
giveaway that this is a constellation change — a retired GSO spacecraft's PRN being
reissued to a new MEO one — and not physics.

**What we did about it.** Every downstream script splits these into `C13_a`/`C13_b`
and so on, at the radius threshold drawn above. Grouping by PRN would silently splice
two unrelated spacecraft into one series, and every per-satellite constant we later
depend on would be an average of two different numbers.
""")

    md("### Problem 3 — a handful of rows are catastrophically wrong")
    co(r"""
worst = d.reindex(d.clk.abs().sort_values(ascending=False).index[:2000])
tbl = (worst.groupby("sat")
            .agg(n=("clk","size"), worst_m=("clk", lambda v: v.iloc[np.argmax(np.abs(v.values))]))
            .sort_values("worst_m", key=abs, ascending=False).head(5))
show(tbl.round(1), "Largest clock outliers")

g  = d[d.sat=="C10"]
gc = g[g.clk.abs() < 100]
print(f"\nC10 clock RMS with the outliers  : {np.sqrt((g.clk**2).mean()):>12,.1f} m")
print(f"C10 clock RMS after |clk| < 100 m: {np.sqrt((gc.clk**2).mean()):>12,.2f} m")
n_cut = int((d.clk.abs() >= 100).sum())
print(f"rows removed corpus-wide by the cut: {n_cut} of {len(d):,} ({100*n_cut/len(d):.4f}%)")

fig, ax = plt.subplots(figsize=(9, 2.8))
ax.plot(g.epoch, g.clk, lw=0.5, color="#c44e52")
ax.set_ylabel("C10 clock error (m)")
ax.set_title("Nine epochs move one satellite's RMS by four orders of magnitude")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** A few isolated epochs reach tens of thousands of metres.
These are stale navigation records — orbital elements not advanced while the
time-of-ephemeris was bumped. `build_errors.py` already rejects most of them by a
consistency check between consecutive messages; these are the survivors.

**What we did about it.** A flat `|clock| < 100 m` cut, applied everywhere below.
It removes **18 rows in two million** and takes C10 from 6,639 m RMS to 14.6 m. This
is worth stating plainly, because an unnoticed outlier of that size does not
*degrade* a result — it invents one. (C10 is a poorly-behaved satellite even after
the cut; the point is that nine epochs were setting its RMS, not its behaviour.)
""")

    co(r"""
# the working corpus, cleaned, used from here on
d = d[d.clk.abs() < 100].dropna(subset=["dR","dA","dC","clk"]).copy()

# PRN reuse -> distinct spacecraft id
rr = np.sqrt(d.px**2 + d.py**2 + d.pz**2)
sid = d.sat.to_numpy().astype(object)
m = d.sat.isin(["C13","C38","C39"]).to_numpy()
sid[m] = np.where(rr.to_numpy()[m] > 3.5e7, d.sat.to_numpy()[m]+"_a", d.sat.to_numpy()[m]+"_b")
d["scid"] = sid

MEO = d[d.orbit_class=="MEO"]
print(f"working corpus: {len(d):,} rows, {d.scid.nunique()} spacecraft "
      f"({d.sat.nunique()} PRNs)")
""")

    md(r"""
---
# Part 2 — What does the error actually look like?

Before any modelling. Four channels: radial, along-track, cross-track (the error
vector rotated into the satellite's own frame) and clock, all in metres.
""")

    co(r"""
ex = {"G05":"GPS MEO", "E11":"Galileo MEO", "C06":"BeiDou GSO"}
ex = {k:v for k,v in ex.items() if k in set(d.sat)}
fig, axes = plt.subplots(len(ex), 1, figsize=(11, 2.0*len(ex)), sharex=True)
for ax,(s,lab) in zip(np.atleast_1d(axes), ex.items()):
    g = d[(d.sat==s) & (d.epoch < d.epoch.min()+pd.Timedelta("21D"))]
    ax.plot(g.epoch, g.dR,  lw=0.7, label="radial")
    ax.plot(g.epoch, g.clk, lw=0.7, label="clock", alpha=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("m"); ax.set_title(f"{s} — {lab}", loc="left")
np.atleast_1d(axes)[0].legend(ncol=2)
fig.suptitle("Three weeks of raw broadcast error", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** The series do not wander around zero. Each one sits on a
**offset of its own** and wiggles around it. That offset is far larger than the
wiggle — and it is the single most important fact about this dataset.
""")

    md("### The trap: almost all of the raw error is a static per-satellite bias")
    co(r"""
rows = []
for (sy, cls), g in d.groupby(["sys","orbit_class"]):
    if len(g) < 20000: continue
    # sd of the error about a rolling 7-day mean, per satellite -> how much it MOVES
    sd = []
    for s, gg in g.groupby("scid"):
        v = gg.sort_values("epoch").dR.to_numpy()
        if len(v) < 700: continue
        k = 672
        base = pd.Series(v).rolling(k, min_periods=k//2).mean().to_numpy()
        sd.append(np.nanstd(v - base))
    if not sd: continue
    rows.append({"group": f"{NAME[sy]} {cls}", "n_sat": g.scid.nunique(),
                 "dR RMS": np.sqrt((g.dR**2).mean()),
                 "dR sd about 7-day mean": float(np.nanmedian(sd))})
T = pd.DataFrame(rows).set_index("group").sort_values("dR RMS", ascending=False)
T["static share of RMS"] = (1 - T["dR sd about 7-day mean"]/T["dR RMS"]).map("{:.0%}".format)
show(T.round(3))

fig, ax = plt.subplots(figsize=(8, 3.0))
y = np.arange(len(T))
ax.barh(y, T["dR RMS"], color="#c44e52", alpha=0.85, label="total RMS")
ax.barh(y, T["dR sd about 7-day mean"], color="#4c72b0", label="sd about a 7-day mean")
ax.set_yticks(y); ax.set_yticklabels(T.index); ax.invert_yaxis()
ax.set_xlabel("radial error (m)"); ax.legend()
ax.set_title("Red is what you would 'predict'. Blue is all there is to learn.")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** For every group, the total radial RMS (red) is dominated by
a constant. The part that actually varies within a week (blue) is a small fraction of
it.

**Why this is a trap.** Any skill number computed against "predict zero" is measuring
that constant. **Our first pass reported 0.998 "skill" for GPS radial purely from
having learned a fixed offset.**

**What we did about it.** Every number from here on is scored against a
**per-satellite baseline** — usually the 7-day window mean — never against zero. This
one decision changes the reported performance of the entire project by an order of
magnitude, which is exactly why it is stated this early.
""")

    md("### When does the error repeat? (it is not daily)")
    co(r"""
def acf_peak(v, max_lag):
    v = v - np.nanmean(v)
    v = np.nan_to_num(v)
    out = np.array([np.corrcoef(v[:-L], v[L:])[0,1] for L in range(1, max_lag)])
    return out

rows, curves = [], {}
for (sy, cls), g in d.groupby(["sys","orbit_class"]):
    if g.scid.nunique() < 2 or len(g) < 50000: continue
    s = g.scid.value_counts().index[0]
    v = (g[g.scid==s].sort_values("epoch")
           .set_index("epoch").dR.resample("900s").mean().to_numpy())
    if len(v) < 1200: continue
    a = acf_peak(v[:23000], 1150)                       # up to ~12 days
    curves[f"{NAME[sy]} {cls}"] = a
    # search from 6 h out: below that the curve is still on its initial decay
    lo = int(6*3600/900)
    lag = int(np.nanargmax(a[lo:]) + lo + 1)
    rows.append({"group": f"{NAME[sy]} {cls}", "example sat": s,
                 "ACF peak lag (h)": lag*900/3600, "r at peak": a[lag-1]})
show(pd.DataFrame(rows).set_index("group").round(2))

fig, ax = plt.subplots(figsize=(10, 3.2))
for k, a in curves.items():
    ax.plot(np.arange(1, len(a)+1)*900/3600, a, lw=1.0, label=k)
ax.axvline(24, color="black", ls=":", lw=1); ax.text(24, 0.9, " 24 h", fontsize=8)
ax.axvline(239.5, color="grey", ls=":", lw=1); ax.text(239.5, 0.9, " 9.98 d", fontsize=8)
ax.axhline(0, color="black", lw=0.6)
ax.set_xscale("log"); ax.set_xlabel("lag (hours, log scale)"); ax.set_ylabel("autocorrelation of radial error")
ax.legend(fontsize=7.5, ncol=2); ax.set_title("The error repeats on the ORBIT, not on the day")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** Each group peaks at its own orbital period, not at 24 hours.
GPS peaks near 12 h, BeiDou MEO near 12.75 h, the geosynchronous classes at 24 h
(because for them the orbit *is* the day), and Galileo has its strongest structure at
**239.5 h = 9.98 days** — its 17-revolution/10-day ground-track repeat.

**Why this matters, and it matters a lot.** A seven-day window contains 14 GPS
cycles, 13 BeiDou MEO cycles, 7 geosynchronous cycles — and **0.7 Galileo repeat
cycles**. You cannot estimate an amplitude and a phase from seven-tenths of a cycle.
That is an architectural fact about the problem, not a tuning knob, and it is the
reason the geosynchronous classes are easier to predict than the MEO ones all the way
through this project.

**What we did about it.** Any periodic basis is built on the *orbital* period of the
class in question, never on a 24-hour day.
""")
