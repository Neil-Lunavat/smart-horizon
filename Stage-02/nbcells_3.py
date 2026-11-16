"""Stage-02 notebook, cells part 3: the reframing, the model, the validation, and
the end-to-end test of whether any of it is useful."""


def add(md, co):

    md(r"""
---
# Part 6 — The reframing: nowcast, not forecast

The competition asks for a **forecast**: seven days of error in, day eight out. There
is a problem with that which has nothing to do with modelling.

**Error = broadcast − precise.** So to compute your own *input* you need the precise
product. We initially believed the precise orbits lagged by two weeks, which would
mean a "prediction" for day 8 could only be made a fortnight after day 8 had
happened. There is no user for that.

**So Task A asks a different question:**

> Given the broadcast message that is live *right now*, plus whatever truth has
> already been published, **how wrong is that message?**

Not a forecast — a **correction**. At epoch *t* the model gets broadcast continuously
right up to *t*, and SP3-derived truth only up to *t* − 3 days. It has to bridge the
gap using broadcast alone. This is the "broadcast ephemeris improvement" problem from
the literature, and it is a system somebody could actually run.
""")

    co(r"""
fig, ax = plt.subplots(figsize=(10, 2.3))
ax.axhline(1.0, xmin=0.02, xmax=0.86, color="#4c72b0", lw=8, solid_capstyle="butt")
ax.axhline(0.5, xmin=0.02, xmax=0.62, color="#c44e52", lw=8, solid_capstyle="butt")
ax.axvline(0.86, color="black", ls="--", lw=1.2)
ax.text(0.87, 1.28, "now  (t)", fontsize=9, fontweight="bold")
ax.text(0.02, 1.14, "broadcast navigation message — available in real time", fontsize=8.5, color="#4c72b0")
ax.text(0.02, 0.64, "SP3 precise truth — 3 days behind", fontsize=8.5, color="#c44e52")
ax.annotate("", xy=(0.86, 0.5), xytext=(0.62, 0.5),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
ax.text(0.74, 0.30, "the 3-day gap the\nmodel must bridge", ha="center", fontsize=8)
ax.set_xlim(0, 1.05); ax.set_ylim(0.1, 1.5); ax.axis("off")
ax.set_title("What the model is allowed to see", loc="left", fontweight="bold")
plt.show()
""")

    md("### How big is the lag, actually? (we had assumed wrong)")
    co(r"""
lat = pd.DataFrame({
    "SP3 latency": ["raw (none)","0.25 d","1 d","2 d","3 d","5 d","7 d","14 d","28 d"],
    "radial": [1.295, 0.103, 0.096, 0.102, 0.105, 0.111, 0.112, 0.124, 0.137],
    "clock":  [7.046, 0.473, 0.555, 0.714, 0.818, 0.953, 1.131, 1.022, 1.803]})
show(lat.set_index("SP3 latency"), "measured on the corpus — logs/latency.log")

fig, ax = plt.subplots(figsize=(7.5, 3.0))
x = [0.25,1,2,3,5,7,14,28]
ax.plot(x, lat.clock[1:], "o-", color="#c44e52", label="clock")
ax.plot(x, lat.radial[1:], "o-", color="#4c72b0", label="radial")
ax.axvline(3, color="black", ls=":", lw=1.2); ax.text(3.2, 1.5, "the real lag\n(3 days)", fontsize=8)
ax.axvline(14, color="grey", ls=":", lw=1.2); ax.text(14.3, 1.5, "what we\nassumed", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("SP3 latency (days)"); ax.set_ylabel("error after correction (m)")
ax.legend(); ax.set_title("Latency costs more than any modelling choice we made")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** We probed NASA CDDIS directly rather than trusting the
assumption, and the final JAXA product is available at a **3-day lag, not two weeks**
(the rapid products are at 2 days). The radial channel barely cares. The clock cares
a lot: 0.818 m at 3 days against 1.022 m at 14 days, and 0.473 m at 6 hours.

**Why this matters more than it looks.** The clock is the dominant term in what a
user feels. **Moving from the assumed operating point to the real one is worth more
than any architecture change in this entire project.** It cost nothing but a
measurement, and we nearly did not make it.

**What we did about it.** Re-fetched the corpus to the present day (236 → 244 days)
and fixed the operating point at a 3-day lag everywhere.
""")

    md(r"""
---
# Part 7 — The one feature that had to be built

Everything the model can use at prediction time must come from the broadcast, since
the truth is 3 days stale. One useful quantity is not in the raw data and had to be
constructed.

**The handover jump.** When a new navigation message takes over from the old one,
evaluate **both** records at the same instant. The old one is being extrapolated
slightly past its intended window; the new one is fresh. Their disagreement is the
ground segment publicly correcting itself. It needs no SP3, and it is known the
moment the new message arrives.

**Our prediction, stated in advance:** the jumps will be tiny and useless.
""")

    co(r"""
ho = pd.read_parquet("handover.parquet")
print(f"{len(ho):,} handover events, {ho.sat.nunique()} satellites")
q = ho[["ho_R","ho_A","ho_C","ho_clk_m"]].abs().quantile([0.5, 0.9, 0.99]).round(4)
show(q, "|jump| quantiles (metres)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
ax = axes[0]
for c, col in [("ho_R","#1f77b4"), ("ho_A","#d62728"), ("ho_C","#2ca02c")]:
    v = ho[c].clip(-1, 1)
    ax.hist(v, bins=120, histtype="step", lw=1.2, color=col, label=c)
ax.set_yscale("log"); ax.set_xlabel("handover jump (m)"); ax.set_ylabel("events")
ax.legend(); ax.set_title(f"Median |jump| {ho.ho_R.abs().median()*1000:.0f} mm, signed median "
             f"{ho.ho_R.median()*1000:+.1f} mm — looks useless")

# the feature that actually worked: the 7-day rolling SD of the jumps, per satellite
ho2 = ho.sort_values(["sat","arc_toe"])
sd = ho2.groupby("sat").ho_R.apply(lambda v: v.clip(-20,20).rolling(200, min_periods=50).std())
ho2 = ho2.assign(ho_R_s7=sd.reset_index(level=0, drop=True))
per = ho2.groupby("sat").ho_R_s7.median().dropna().sort_values()
ax = axes[1]
ax.barh(range(len(per)), per.values, color="#4c72b0", alpha=0.85)
ax.set_yticks([]); ax.set_xlabel("rolling sd of the radial handover jump (m)")
ax.set_title("...but its SPREAD differs 10x between satellites")
for nm in ["G27","G04"]:
    if nm in per.index:
        i = list(per.index).index(nm)
        ax.annotate(nm, (per.iloc[i], i), fontsize=8, xytext=(6,-3), textcoords="offset points")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows, and why the prediction was wrong.** The typical jump really
is tiny — a signed median of a couple of millimetres, a median absolute value of a few
centimetres — and as a *value* it is useless. That part of the prediction was right.
But the **spread** of the jumps varies by an order of magnitude between satellites,
and it is that spread which carries information.

**The model's top features turned out to be `ho_R_s7`, `ho_C_s7`, `ho_A_s7` — the
7-day rolling standard deviation of the jumps, not the jumps themselves.** It is
reading how *jittery* the ground segment has been for this satellite lately, i.e. how
much to distrust the current message. That was worth **11 points of radial skill
(71% → 82%)**.

**What we did about it.** Kept the feature, and recorded that we had predicted the
wrong thing for the right reason. It also matters for Part 9: this same quantity
predicts the error's *scale*, which is what makes a calibrated uncertainty possible
from broadcast alone.
""")

    md(r"""
---
# Part 8 — What to optimise, and the model

Part 5 established the principle physically. This is its standard form.

**SISRE** — signal-in-space range error — is the accepted way to weight the four
channels by what a user actually feels. Radial and clock enter almost at full
strength; along- and cross-track are down-weighted about 7x because they are nearly
perpendicular to the line of sight.
""")

    co(r"""
W_R = {"MEO":0.98, "GSO":0.99, "GEO":0.99}
W_A = {"MEO":0.139,"GSO":0.123,"GEO":0.123}

# de-bias each component per satellite first: those constants are Part 4's antenna
# offset and Part 3's datum, and neither is what we are trying to optimise here
rows = []
for s in "GEC":
    g = MEO[MEO.sys == s].copy()
    for c in ["dR","dA","dC","clk"]:
        g[c] = g[c] - g.groupby("scid")[c].transform("median")
    wR, wA = W_R["MEO"], W_A["MEO"]
    rc = np.mean((wR*g.dR - g.clk)**2)
    al = wA**2*np.mean(g.dA**2)
    cr = wA**2*np.mean(g.dC**2)
    tot = rc + al + cr
    rows.append({"constellation":NAME[s], "radial+clock":rc/tot, "along":al/tot, "cross":cr/tot})
V = pd.DataFrame(rows).set_index("constellation")
show((V*100).round(1), "Share of SISRE^2 (%)")

fig, ax = plt.subplots(figsize=(7.5, 2.6))
b = np.zeros(len(V))
for c, col in [("radial+clock","#4c72b0"), ("along","#c44e52"), ("cross","#55a868")]:
    ax.barh(V.index, V[c], left=b, color=col, label=c); b = b + V[c].values
ax.set_xlabel("share of SISRE$^2$"); ax.legend(ncol=3, loc="lower right")
ax.set_title("Along-track and cross-track are almost all of the effort and none of the error")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** Radial and clock are ~92% of GPS SISRE² and essentially all
of it for Galileo and BeiDou. Along-track is about 6% for GPS and 0.1% for the
others.

**Why this mattered to us.** We had already spent **two falsified physics hypotheses**
chasing along-track — a sun-geometry/eclipse hypothesis (predicted along-track spikes
at low beta angle: result flat, R² = 0.008) and an upload-age hypothesis (predicted
error growth since a new upload: result flat across 0→12 h). Zeroing GPS along-track
*perfectly* moves SISRE only from 0.557 to 0.540 m.

**What we did about it.** Dropped along-track as a target and reported SISRE
throughout, rather than per-component RMS which had been quietly flattering us.

### The model ladder

Every model below sees identical information: broadcast up to *t*, truth up to
*t* − 3 days. The numbers are the single held-out window used during development;
Part 10 is the honest cross-validated version.

| model | SISRE | note |
|---|---|---|
| raw broadcast | 7.393 | do nothing |
| 7-day window mean | 0.417 | **the baseline to beat** |
| XGBoost | 0.453 | **lost to the baseline** |
| MLP (flattened) | 0.358 | |
| GRU (2-layer) | 0.380 | **lost to the MLP** |
| TCN (dilated conv) | 0.337 | |
| v2 (per-system scaling) | 0.353 | |
| **v3 (heteroscedastic)** | **0.337** | best RMS |
| v4 (Student-t) | 0.345 | best calibration |

**Two negative results worth keeping.** Gradient boosting lost to a 7-day average,
and a recurrent network lost to a plain MLP given the same data flattened. When an
architecture that models sequence structure cannot beat one that ignores it, the
signal is not sequence-shaped — it is constants plus slow drift.

**The bug that produced v2.** v1 pooled all constellations under plain mean-squared
error. BeiDou clock error is ~12 m and Galileo's is ~0.38 m, so BeiDou contributed
about 1000x the squared error and Galileo was optimised as if it were noise — **v1
made Galileo clock worse than doing nothing** (0.383 → 0.459 m). The fix is to divide
each target by its per-system residual sd inside the loss. Galileo went 0.455 → 0.281.
""")

    md("### v3 — the model we kept")
    co(r"""
z = np.load("taskA.npz", allow_pickle=True)
sats, t0 = list(z["sats"]), pd.Timestamp(str(z["t0"]))
p = np.load("v3_pred.npz")
Yt, Pf, Pt, St, Bt = p["Yt"], p["Pf"], p["Pt"], p["St"], p["Bt"]
si, tt, SYSI = p["si"], p["tt"], p["SYSI"]
wR, wA = p["wR"], p["wA"]
TG = ["dR","dA","dC","dClk_m"]

def sisre(r, m=None):
    m = np.ones(len(r), bool) if m is None else m
    return np.sqrt(np.mean((wR[m]*r[m,0] - r[m,3])**2 + wA[m]**2*(r[m,1]**2 + r[m,2]**2)))

rows = []
for tag, P in [("raw broadcast", np.zeros_like(Yt)), ("7-day window mean", Bt), ("v3", Pf)]:
    r = Yt - P
    rows.append({"model": tag, **{c: np.sqrt((r[:,j]**2).mean()) for j,c in enumerate(TG)},
                 "SISRE": sisre(r)})
show(pd.DataFrame(rows).set_index("model").round(3),
     f"held-out window {t0 + pd.Timedelta(seconds=int(tt.min())*900):%Y-%m-%d} "
     f"to {t0 + pd.Timedelta(seconds=int(tt.max())*900):%Y-%m-%d}   (n = {len(Yt):,})")

rows = []
for k, s in enumerate("GECJ"):
    m = SYSI == k
    if m.sum() < 200: continue
    rows.append({"system": NAME[s], "n": int(m.sum()), "raw": sisre(Yt, m),
                 "baseline": sisre(Yt-Bt, m), "v3": sisre(Yt-Pf, m),
                 "cut vs raw": 1 - sisre(Yt-Pf, m)/sisre(Yt, m)})
PS = pd.DataFrame(rows).set_index("system")
PS["cut vs raw"] = PS["cut vs raw"].map("{:.1%}".format)
show(PS.round(3), "per system")

fig, ax = plt.subplots(figsize=(8, 3.0))
x = np.arange(len(PS)); w = 0.27
ax.bar(x-w, PS.raw, w, label="raw broadcast", color="#c44e52")
ax.bar(x,   PS.baseline, w, label="7-day mean baseline", color="#dd8452")
ax.bar(x+w, PS.v3, w, label="v3 nowcast", color="#4c72b0")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(PS.index)
ax.set_ylabel("SISRE (m, log scale)"); ax.legend()
ax.set_title("The model beats the baseline everywhere — but look at the gap size")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** Against raw broadcast the reduction is enormous and mostly
BeiDou's constant. Against the **7-day mean baseline** — the honest comparison for a
*model* — the bars are close. That gap is the entire contribution of the machine
learning, and Part 10 measures it properly.
""")

    md(r"""
---
# Part 9 — The residuals: a scale mixture, not a heavy tail

This is the finding that transferred to everything afterwards, and it started as an
apparent disaster.

Pooling the model's clock residuals across all satellites gave **kurtosis 438 and a
Shapiro-Wilk W of 0.507**. For a competition scored on Shapiro-Wilk, that looks
fatal. It was a measurement error — ours.
""")

    co(r"""
res = Yt - Pt
j = 0                                     # radial
ws, wp = [], []
for s in np.unique(si):
    m = si == s
    if m.sum() < 200: continue
    x = res[m, j]
    ws.append(stats.shapiro(x[np.random.default_rng(0).choice(len(x), min(400,len(x)), replace=False)]).statistic)
print(f"Shapiro-W WITHIN a single satellite (median over {len(ws)} sats) : {np.median(ws):.4f}")
sub = res[np.random.default_rng(0).choice(len(res), 4000, replace=False), j]
print(f"Shapiro-W POOLED across satellites                             : {stats.shapiro(sub).statistic:.4f}")

sd_per = np.array([res[si==s, j].std() for s in np.unique(si) if (si==s).sum() > 200])
print(f"\nper-satellite residual sd spans {sd_per.min():.3f} to {sd_per.max():.3f} m "
      f"({sd_per.max()/sd_per.min():.0f}x)")

# standardise each satellite by its own sd, then pool again
zsc = np.empty_like(res[:, j])
for s in np.unique(si):
    m = si == s
    v = res[m, j]
    zsc[m] = (v - v.mean())/(v.std() + 1e-9)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
ax = axes[0]
ax.hist(res[:, j], bins=200, range=(-0.6, 0.6), color="#c44e52", alpha=0.85)
ax.set_yscale("log"); ax.set_title(f"pooled raw residuals\nW = {stats.shapiro(sub).statistic:.3f}")
ax.set_xlabel("radial residual (m)")

ax = axes[1]
ax.hist(sd_per, bins=25, color="#dd8452", alpha=0.85)
ax.set_xlabel("that satellite's own residual sd (m)"); ax.set_ylabel("spacecraft")
ax.set_title(f"...because the scales differ {sd_per.max()/sd_per.min():.0f}x")

ax = axes[2]
sub2 = zsc[np.random.default_rng(0).choice(len(zsc), 4000, replace=False)]
w2 = stats.shapiro(sub2).statistic
ax.hist(zsc, bins=200, range=(-5, 5), density=True, color="#4c72b0", alpha=0.85)
xx = np.linspace(-5, 5, 200); ax.plot(xx, stats.norm.pdf(xx), "k--", lw=1.2)
ax.set_yscale("log"); ax.set_xlabel("residual / that satellite's own sd")
ax.set_title(f"standardised, then pooled\nW = {w2:.4f}")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the three panels show, left to right.**

1. Pooled, the residuals look badly non-Gaussian — a sharp peak with long tails.
2. The cause: satellites differ in residual **scale** by an order of magnitude.
3. Standardise each satellite by its own sd and pool again, and the distribution is
   Gaussian to three decimal places.

**The mechanism, stated plainly: pooling Gaussians of different variance produces a
scale mixture, and a scale mixture is heavy-tailed by construction.** Every component
was normal. The histogram was not. Nothing was wrong with the model.

**What we did about it — and this is the important part.** The fix is *not* a robust
loss on the mean. Huber was exactly the wrong instinct: it tells the model to ignore
erratic satellites rather than to represent them. The fix is to **model the scale**.
So v3 emits a mean *and* a log-sigma per target and trains under Gaussian negative
log-likelihood, which makes the standardised residuals genuinely N(0,1) and produces
a calibrated error bar for free.

And the scale is predictable from broadcast alone — the handover-jump spread of
Part 7 correlates with residual sd at r = 0.69 radial — so no SP3 is needed at
prediction time to know how much to trust the answer.
""")

    co(r"""
zt = (Yt - Pt)/St
rows = []
for j, c in enumerate(TG):
    x = zt[:, j]
    sub = x[np.random.default_rng(0).choice(len(x), 4000, replace=False)]
    w, pv = stats.shapiro(sub)
    rows.append({"target": c, "mean": x.mean(), "sd": x.std(), "skew": stats.skew(x),
                 "kurtosis": stats.kurtosis(x), "Shapiro W": w})
show(pd.DataFrame(rows).set_index("target").round(4), "standardised residuals (y-mu)/sigma")

cov = []
for j, c in enumerate(TG):
    a = np.abs(Yt[:,j]-Pt[:,j])/St[:,j]
    cov.append({"target": c, **{f"{k}s": (a<k).mean() for k in (1,2,3)}})
CV = pd.DataFrame(cov).set_index("target")
show((CV*100).round(1), "coverage (%)   nominal: 68.3 / 95.4 / 99.7")

fig, ax = plt.subplots(1, 4, figsize=(12, 2.9))
for j, c in enumerate(TG):
    x = zt[:, j]
    x = x[np.random.default_rng(1).choice(len(x), 3000, replace=False)]
    stats.probplot(x, dist="norm", plot=ax[j])
    ax[j].set_title(c, fontsize=9); ax[j].get_lines()[0].set_markersize(2)
    ax[j].get_lines()[0].set_alpha(0.4); ax[j].set_xlabel(""); ax[j].set_ylabel("")
fig.suptitle("Q-Q plots of the standardised residuals against a normal", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the Q-Q plots show.** Straight lines through the body of the distribution for
all four channels, with the clock bending at the extreme tails. That bend is real —
some satellites genuinely jump — and it is why v4 was built with a Student-t
predictive law, which independently chose the heaviest tails (nu = 7.8) for the clock
exactly as the diagnosis predicted.

**This is the direct answer to a Shapiro-Wilk-scored competition:** the residuals are
Gaussian, provided you test them the right way — per satellite, or standardised
before pooling. Testing them pooled and raw will report catastrophic non-normality no
matter how good the model is.
""")

    md(r"""
---
# Part 10 — The validation reckoning

Everything up to here was reported on **one held-out window**. That was wrong in two
ways: the window is thin, and four model versions had been designed while looking at
its score. That is model selection on the test set, and we did it without noticing.

The honest protocol: **24 random contiguous 2-week test blocks**, model retrained
from scratch for each, training data strictly before the block with an 8-day gap so
that no window straddles the boundary.
""")

    co(r"""
import json
mc = pd.DataFrame(json.load(open("logs/pod_results/mccv.json"))).sort_values("start")
mc["gain"] = 1 - mc.test/mc.base

print(f"raw broadcast SISRE   {mc.raw.mean():.3f} m")
print(f"baseline SISRE        {mc.base.mean():.3f} +/- {mc.base.std():.3f} m")
print(f"MODEL SISRE           {mc.test.mean():.3f} +/- {mc.test.std():.3f} m   "
      f"(median {mc.test.median():.3f}, range [{mc.test.min():.3f}, {mc.test.max():.3f}])")
print(f"vs raw broadcast      {100*(1-mc.test.mean()/mc.raw.mean()):.1f}% reduction")
print(f"vs baseline           {100*mc.gain.mean():.1f}% mean improvement")
print(f"beats baseline in     {int((mc.test < mc.base).sum())} / {len(mc)} folds")
dif = mc.base - mc.test
print(f"paired t              {dif.mean()/(dif.std(ddof=1)/np.sqrt(len(dif))):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
ax = axes[0]
x = np.arange(len(mc))
ax.bar(x-0.2, mc.base, 0.4, label="7-day mean baseline", color="#dd8452")
ax.bar(x+0.2, mc.test, 0.4, label="v3 model", color="#4c72b0")
ax.set_xlabel("cross-validation fold (ordered in time)"); ax.set_ylabel("SISRE (m)")
ax.legend(); ax.set_title("24 blocked folds — the model wins every one, narrowly")

ax = axes[1]
ax.plot(x, 100*mc.gain, "o-", color="#4c72b0")
ax.axhline(100*mc.gain.mean(), color="black", ls="--", lw=1,
           label=f"mean {100*mc.gain.mean():.1f}%")
ax.axhline(0, color="#c44e52", lw=1)
ax.set_xlabel("fold (ordered in time)"); ax.set_ylabel("improvement over baseline (%)")
ax.legend(); ax.set_title("...and the margin grows with how much history it has")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plots show — read the two halves separately, because they say different
things.**

**Against raw broadcast: about a 90% reduction, and it is not fragile.** Even the
worst of 24 folds is a 75% cut.

**Against the 7-day-mean baseline: around 12%, small but utterly consistent.** The
model wins **24 out of 24 folds** at a paired t of about 8 — the improvement is real
and highly significant — but it is a tenth, not a doubling. The right panel shows why
the single-window number was so flattering: the margin is near zero on the earliest
blocks, where there is little training history, and 11–23% on later ones.

**And the fold-to-fold spread is the honest headline.** The standard deviation is
0.43 on a mean of 0.71, a 5x range across folds. Any single fortnight can be sold as
0.36 m or as 1.81 m depending on which one you pick. **The window we had quoted all
session was the easiest in the corpus**, and the true performance was roughly double
what we had been claiming.

**What we did about it.** Corrected the numbers in the write-up rather than replacing
them silently, and adopted a rule for the rest of the project: cross-validate before
quoting anything, and always show per-fold values so that no single block can carry a
number.
""")

    md(r"""
---
# Part 11 — So is it provably useful?

Now we can close the loop. Part 5 built a receiver and showed that most of the
measured "error" is invisible to it. Part 8 built a model that reduces the error.

**The question: does the model move the fix?** Same simulator, same receiver, same
elevation mask. The only change is which correction we apply to each satellite's
range before solving.
""")

    co(r"""
scid = np.array(sats)[si]
M = pd.DataFrame({"epoch": t0 + pd.to_timedelta(tt*900, unit="s"),
                  "sat": [s.split("_")[0] for s in scid]})
for k, c in enumerate(TG):
    M["y_"+c] = Yt[:,k]; M["m_"+c] = Pf[:,k]; M["b_"+c] = Bt[:,k]

geo = d[d.epoch >= M.epoch.min()][["epoch","sat","sys","px","py","pz"]]
J = geometry(geo.merge(M, on=["sat","epoch"], how="inner"))
print(f"{len(J):,} satellite-epochs, {J.sat.nunique()} satellites, {J.epoch.nunique()} epochs")

def err_after(pre):
    if pre == "":
        R, A, Cc, K = J.y_dR, J.y_dA, J.y_dC, J.y_dClk_m
    else:
        R, A, Cc, K = (J["y_"+c] - J[pre+"_"+c] for c in TG)
    return R*J.uR + A*J.uA + Cc*J.uC - K

J["e_raw"]   = err_after("")
J["e_base"]  = err_after("b")
J["e_model"] = err_after("m")
V2 = J[J.elev > 10].copy()

print(f"\n=== position error at New Delhi, {V2.epoch.nunique()} epochs ===")
print("all constellations:")
for c, tag in [("e_raw","broadcast, uncorrected"), ("e_base","+ 7-day mean correction"),
               ("e_model","+ v3 nowcast correction")]:
    _ = run_fix(V2, c, tag)
print("GPS only:")
gonly = V2[V2.sys == "G"]
out = {}
for c, tag in [("e_raw","broadcast, uncorrected"), ("e_base","+ 7-day mean correction"),
               ("e_model","+ v3 nowcast correction")]:
    out[tag] = run_fix(gonly, c, tag)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
ax = axes[0]
for (tag, O), col in zip(out.items(), ["#c44e52","#dd8452","#4c72b0"]):
    h = np.sort(np.hypot(O[:,0], O[:,1]))
    ax.plot(h, np.linspace(0, 1, len(h)), lw=1.6, color=col, label=tag)
ax.set_xlabel("horizontal position error (m)"); ax.set_ylabel("fraction of epochs")
ax.set_xlim(0, 2.5); ax.legend(fontsize=7.5); ax.set_title("GPS-only fix, cumulative")

ax = axes[1]
budget = {"ionosphere (uncorrected)":5.0, "ionosphere (dual-freq residual)":0.3,
          "troposphere (modelled residual)":0.3, "multipath, open sky":0.5,
          "multipath, urban":3.0, "receiver noise (phone)":1.0,
          "broadcast SISRE, uncorrected":np.sqrt((gonly.e_raw**2).mean()),
          "broadcast SISRE, our model":np.sqrt((gonly.e_model**2).mean())}
k = list(budget); vv = [budget[x] for x in k]
cols = ["#bbb"]*6 + ["#c44e52","#4c72b0"]
ax.barh(range(len(k)), vv, color=cols, alpha=0.9)
ax.set_yticks(range(len(k))); ax.set_yticklabels(k, fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("typical 1-sigma range error (m)")
ax.set_title("What we fixed, against the rest of a phone's error budget")
fig.tight_layout(); plt.show()
print("\n(grey bars are typical published values for context, not measured here)")
""")

    md(r"""
**What the left panel shows — the answer, in metres.**

The correction is **real and it works**. Applying it to a GPS-only fix cuts the
horizontal error meaningfully and the vertical error more, and the improvement holds
across the whole test period rather than on selected epochs.

But read the three curves carefully. **Most of the gap is between "uncorrected" and
"7-day mean" — a per-satellite constant, no learning involved.** The v3 model adds a
further slice on top, consistent with the 12% figure from Part 10. That is exactly
the same story as the radial channel in Part 4: the physics and the constants do most
of the work, and the machine learning earns a real but thin margin.

**What the right panel shows — the answer to whether you would ship it.**

Broadcast error is **not** the dominant term in a phone's position error. The
ionosphere alone is several metres uncorrected, and urban multipath can be worse. Our
correction removes a few tenths of a metre from a budget of several metres.
**In a phone, it would be invisible.**

**Where it would genuinely matter** is any application that has already eliminated
the other terms — precise point positioning, survey, agriculture, timing, aviation
integrity monitoring. There, orbit and clock error is a leading term, and a few
tenths of a metre is worth having. That is also exactly the niche that real
correction services occupy: **SBAS (WAAS, EGNOS), Galileo HAS and QZSS CLAS broadcast
precisely this kind of orbit-and-clock correction.** What Task A built is a
re-derivation of that idea from data, which is a good sanity check on the design and
a firm ceiling on the novelty.

**So: provably useful, yes — demonstrably, in metres, on a real geometry. Useful
enough to change your phone, no.** Both halves of that sentence are the finding.
""")

    md(r"""
---
# Part 12 — Why we left Task A anyway

Task A works as a system. It fails as a *submission*, for one decisive reason.

**At evaluation, participants are handed seven days of error values and a list of
day-8 timestamps. That is the entire input.** The Task A model requires, at inference
time:

* the live broadcast navigation message,
* SP3-derived truth up to 3 days before the epoch,
* and the **satellite's identity**, in order to fetch either of the above.

**None of these are supplied, and none can be reconstructed from five columns of
error values with no satellite identifier.** So the model cannot be scored — not
"would score poorly", but *cannot be run*. A submission consisting only of Task A is
a submission the harness cannot execute.

Two supporting reasons, both already visible above: the margin over a trivial
baseline is 12%, not a doubling; and the fold-to-fold variance is a 5x spread,
because a trained model can be lucky or unlucky about its training period.

**But nothing here was wasted.** Every finding transferred to the deliverable:

| Task A finding | how it is used in the delivered model |
|---|---|
| residuals are Gaussian per satellite, scale-mixed when pooled | sigma reported per window, normality never tested pooled |
| heteroscedastic modelling with a calibrated sigma | the deliverable returns a 1-sigma per prediction |
| the clock has no 24-hour memory | clock restricted to level estimators; harmonics forbidden |
| periodicity is orbital, not daily | sidereal basis for GEO, beat frequencies for MEO |
| geosynchronous is the most predictable class | separate GEO and MEO models |
| **cross-validate before quoting anything** | blocked CV from the outset, per-fold values always shown |

---

# Summary of Stage 2

**What we did.** Replaced the unusable delivered files with a 244-day, 2.07-million-row
corpus built from primary sources, then reframed the task from a forecast nobody
could run into a nowcast somebody could.

**What we found, in order of how much it changed our minds:**

1. **Most of the raw error is a static per-satellite bias.** Scoring against zero
   reports ~0.99 "skill" for having learned a constant. Everything is scored against
   a per-satellite baseline instead.
2. **65–86% of the clock signal is the reference product's datum, not the
   satellites.** Four independent ground segments cannot correlate at r = 0.85–0.96.
3. **The largest orbit offset is the antenna phase centre**, matching published IGS
   values to 4 mm on Block IIF — a physical constant, not something we discovered,
   and about 86% of the radial "win" is that table lookup.
4. **A receiver absorbs almost all of it anyway.** A common error contributes exactly
   zero to a position fix; a shared radial offset is ~96% absorbed. Only
   satellite-to-satellite differences survive.
5. **The apparent non-normality was a scale mixture**, not a model failure — which is
   the direct answer to a Shapiro-Wilk-scored problem.
6. **The SP3 latency is 3 days, not two weeks**, and that measurement was worth more
   than any architecture change we made.
7. **The honest model margin is ~12% over a trivial baseline**, 24/24 folds, after we
   caught ourselves quoting the easiest window in the corpus.

**And the useful-in-the-real-world answer:** the correction demonstrably improves a
simulated position fix, by a few tenths of a metre — real, reproducible, and far
below the ionosphere and multipath terms that dominate a phone. It belongs in the
class of correction service that SBAS and Galileo HAS already provide, not in a
consumer device.
""")
