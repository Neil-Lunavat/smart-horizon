"""Stage-02 notebook, cells part 2: the clock common mode, the antenna, and the
receiver test that asks whether either of them matters."""


def add(md, co):

    md(r"""
---
# Part 3 — Most of the "clock error" is not the satellite's

This is the least intuitive finding in the project, so it is worth setting up
carefully.

**The question.** The clock channel is by far the largest error we measure — around
7 m RMS raw, against roughly 1 m for the orbit. Since the clock is also ~92% of what
a user feels (Part 6), everything depends on what that 7 m actually *is*.

**The setup.** Take the clock error of every satellite in one constellation and, at
each epoch, split it into two pieces:

    clock error of satellite i at time t  =  c(t)  +  r_i(t)
                                             ^^^^     ^^^^^^
                                    shared by all    that satellite's own

`c(t)` is just the average across the satellites of that constellation at that epoch.
If satellite clocks drift independently — which is what "independent atomic clocks"
means — then `c(t)` should be small noise and explain almost nothing.

**The measurement.** For each satellite, what fraction of its variance does `c(t)`
account for: `1 − var(r_i)/var(clock_i)`.
""")

    co(r"""
def cm_share(g, col):
    # fraction of each satellite's variance explained by the epoch-wise group mean
    cm = g.groupby("epoch")[col].transform("mean")
    res = g[col] - cm
    out = []
    for _, t in g.assign(_r=res).groupby("scid"):
        vt = np.var(t[col])
        if vt > 0 and len(t) > 500:
            out.append(1 - np.var(t._r)/vt)
    return float(np.median(out)) if out else np.nan

# the per-satellite-per-day median is removed first, so we are asking about the
# WITHIN-DAY variation -- the part a nowcast would have to predict
rows = []
for s in "GEC":
    g = MEO[MEO.sys == s]
    rows.append({"constellation": NAME[s], "sats": g.scid.nunique(),
                 "clock": cm_share(g, "clkd"),
                 "radial": cm_share(g.assign(v=g.dR - g.groupby("scid").dR.transform("median")), "v"),
                 "along":  cm_share(g.assign(v=g.dA - g.groupby("scid").dA.transform("median")), "v"),
                 "cross":  cm_share(g.assign(v=g.dC - g.groupby("scid").dC.transform("median")), "v")})
S = pd.DataFrame(rows).set_index("constellation")
show((S[["clock","radial","along","cross"]]*100).round(1),
     "Share of each satellite's variance explained by its constellation's common mode (%)")
""")

    md(r"""
**What the table shows.** For the **clock**, the shared component explains roughly
**two-thirds to six-sevenths** of every satellite's variance. For the **orbit**
components — same satellites, same epochs, same arithmetic — it explains essentially
nothing (a few percent, which is what you would get from chance alone).

So there is something large, shared, and specific to the clock channel. The obvious
reading is "the ground segment's clock ensemble moves together". The next test kills
that reading.
""")

    md("### The decisive test: do *different* constellations share it?")
    co(r"""
cms = {}
for s in "GECJ":
    g = (MEO if s != "J" else d)
    g = g[g.sys == s]
    if g.scid.nunique() < 2: continue
    cms[NAME[s]] = g.groupby("epoch").clkd.mean()
CM = pd.DataFrame(cms).dropna()
R  = CM.corr()
show(R.round(3), f"Correlation between constellations' common-mode clock series "
                 f"(n = {len(CM):,} epochs)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
ax = axes[0]
w = CM.loc[CM.index[:288]]                      # three days
for c in w.columns:
    ax.plot(w.index, w[c], lw=1.1, label=c)
ax.set_ylabel("constellation mean clock error (m)")
ax.set_title("Four independent ground segments, three days")
ax.legend(fontsize=7.5); ax.tick_params(axis="x", rotation=30)

ax = axes[1]
im = ax.imshow(R.values, vmin=0, vmax=1, cmap="RdYlBu_r")
ax.set_xticks(range(len(R))); ax.set_xticklabels(R.columns, rotation=45)
ax.set_yticks(range(len(R))); ax.set_yticklabels(R.index)
for i in range(len(R)):
    for j in range(len(R)):
        ax.text(j, i, f"{R.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.grid(False); ax.set_title("...are 0.85-0.96 correlated")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows, and why it settles it.** GPS, Galileo, BeiDou and QZSS are
operated by four different countries, with four physically separate ground segments,
four separate clock ensembles, and four separate time scales. **There is no mechanism
by which their satellite clocks can move together at r = 0.85–0.96.** The left panel
makes it visual: the four curves are the same curve.

So the shared component is not coming from the satellites. It is coming from the one
thing all four measurements have in common: **the precise product we differenced
against.** Every SP3 file has an arbitrary per-epoch clock datum — the analysis
centre's choice of reference — and JAXA produces all four constellations in a single
file, with a single datum. That datum is imprinted identically on every satellite in
it, and it lands in our "error" because we subtracted it.

**So: 65–86% of the thing we were calling satellite clock error is a bookkeeping
convention of the reference product, not a property of any satellite.**
""")

    co(r"""
glob = d.groupby("epoch").clkd.transform("mean")
before = d.groupby("scid").clkd.std().median()
after  = (d.clkd - glob).groupby(d.scid).std().median()
print(f"median per-satellite clock sd   before {before:.3f} m   after {after:.3f} m "
      f"({100*(1-after/before):.0f}% removed by one global series)")

v = d.groupby("epoch").clkd.mean().sort_index()
v = v.reindex(pd.date_range(v.index.min(), v.index.max(), freq="900s")).to_numpy()
print("\nautocorrelation of the common mode:")
for lag, lab in [(1,"15 min"), (4,"1 h"), (96,"24 h"), (672,"7 d")]:
    a, b = v[:-lag], v[lag:]; m = np.isfinite(a) & np.isfinite(b)
    print(f"   lag {lab:7s}  r = {np.corrcoef(a[m], b[m])[0,1]:+.3f}")
""")

    md(r"""
**What this adds.** The common mode is extremely smooth at short lag (r = 0.99 at
15 minutes) and has largely forgotten itself by one day. That shape is exactly what a
per-epoch alignment convention looks like, and it is *not* what an atomic clock looks
like.

**What we did about it.** Two things. First, it explains why a **nowcast** is a
sensible target and a **24-hour forecast** is not: a quantity that is smooth at
15 minutes and memoryless at 24 hours can be tracked but not forecast. Second, it
tells us the headline "7 m of clock error" is not 7 m of anything a user experiences
— which Part 6 then confirms directly.
""")

    md(r"""
---
# Part 4 — The first physical finding: it is the antenna

Back to the orbit. Part 2 showed each satellite sits on a large fixed radial offset.
Plotting those offsets, they clustered by **hardware block** — all the Block IIF
satellites together, all the Block IIR-B together. Hardware clustering suggests a
hardware explanation, so we tested the obvious one.

**The hypothesis.** Broadcast ephemeris tells you where the satellite's **antenna
phase centre** is, because that is where the signal comes from. Precise SP3 orbits
tell you where its **centre of mass** is, because that is what orbit dynamics
integrates. These are different points on the spacecraft, separated by a metre or so
along the body Z axis — which points at nadir.

**The prediction, made before looking anything up.** If that is the cause, the offset
must be almost entirely **radial** (because body Z is radial), essentially zero in
along- and cross-track, and constant per satellite.
""")

    co(r"""
per_sat = MEO.groupby("scid")[["dR","dA","dC"]].median()
print(f"per-satellite median radial : {per_sat.dR.median():+.3f} m   "
      f"(IQR {per_sat.dR.quantile(.25):+.2f} to {per_sat.dR.quantile(.75):+.2f})")
print(f"                along-track : {per_sat.dA.median():+.3f} m")
print(f"                cross-track : {per_sat.dC.median():+.3f} m")

fig, ax = plt.subplots(figsize=(7.5, 3.0))
for c, col in [("dR","#1f77b4"), ("dA","#d62728"), ("dC","#2ca02c")]:
    ax.hist(per_sat[c], bins=28, alpha=0.6, color=col, label=c)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("per-satellite median error (m)"); ax.set_ylabel("spacecraft")
ax.legend(); ax.set_title("The offset is almost purely radial — as predicted")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** Radial is displaced by about −1.1 m; along- and cross-track
sit on zero. The prediction holds.

That is suggestive but not proof — so we downloaded the IGS `igs20.atx` antenna file,
which publishes these offsets, **after** doing the measurement, and compared.
""")

    co(r"""
tab = pd.read_csv("satellite_offset_table.csv")
tab = tab.dropna(subset=["antex_z_m","meas_radial_m"])
gps = tab[tab.sys == "G"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
for blk, g in gps.groupby("block"):
    ax.scatter(g.antex_z_m, -g.meas_radial_m, s=34, alpha=0.85, label=blk)
lim = [0.5, 2.1]
ax.plot(lim, lim, color="black", ls="--", lw=1)
ax.set_xlabel("published ANTEX z offset (m)"); ax.set_ylabel("measured, from the data (m)")
ax.legend(fontsize=7); ax.set_title("GPS: measured vs published, nothing fitted")

r2 = np.corrcoef(gps.antex_z_m, -gps.meas_radial_m)[0,1]**2
print(f"GPS  R^2 = {r2:.3f}   n = {len(gps)} satellites")

blk = (tab.groupby("block")
          .agg(n=("prn","size"), published=("antex_z_m","median"),
               measured=("meas_radial_m", lambda v: -np.median(v)))
          .assign(residual=lambda t: t.measured - t.published))
blk = blk[blk.n >= 3].sort_values("residual", key=abs)
show(blk.round(3), "By hardware block (metres)")

ax = axes[1]
y = np.arange(len(blk))
ax.barh(y, blk.residual, color=np.where(blk.residual.abs() < 0.1, "#55a868", "#c44e52"), alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(blk.index, fontsize=7); ax.invert_yaxis()
ax.set_xlabel("measured − published (m)"); ax.set_title("Residual per block")
fig.tight_layout(); plt.show()
""")

    md(r"""
**What the plot shows.** The points sit on the 1:1 line. Block IIF agrees with the
published value to **4 mm across 11 satellites**, GPS as a whole gives **R² ≈ 0.95**,
and the ordering across all five GPS blocks is reproduced exactly. Nothing was
fitted — this is a measurement against an independent published table.

**Two honest notes.** First, **this is not our discovery.** It is a standard,
documented correction (Montenbruck et al., *Broadcast versus precise ephemerides: a
multi-GNSS perspective*, GPS Solutions 2015). We re-derived a published table from
data. That should *raise* confidence in the pipeline and *lower* any claim of
novelty, and both are worth saying. Second, the two blocks with large residuals
(BeiDou-2 IGSO, QZSS-2I) are a known bug in our table — they use the wrong frequency
convention. Not fixed.

**The uncomfortable consequence.**

| radial correction | RMS |
|---|---|
| raw broadcast − precise | 1.295 m |
| minus the published ANTEX offset — a lookup table, no training | **0.224 m** |
| minus a per-satellite constant learned from history | 0.184 m |
| full learned model | 0.144 m |

**About 86% of the radial "win" is a table lookup** that a GNSS practitioner would
apply *before* calling the difference an error at all. The machine learning earns
0.184 → 0.144.

**What we did about it.** Reported it exactly like this, rather than quoting the
1.295 → 0.144 number. And then asked the question in Part 5, which is the one that
actually decides how much of any of this matters.
""")

    md(r"""
---
# Part 5 — Does any of this matter to a receiver?

Everything so far is about the difference between two *products*. But nobody uses
that difference. People use GPS to find out where they are.

So here is the question this section answers, end to end:

> A phone trilaterates its position from several satellites' broadcast messages.
> Does it care that the broadcast refers to the antenna phase centre rather than the
> centre of mass? Does it care about a 7 m clock error that every satellite shares?
> **Which parts of what we just measured actually move a position fix, and which
> parts are invisible?**

**How the simulator works.** A receiver at a known location has four unknowns:
three coordinates and its own clock offset. It forms one equation per visible
satellite. We put the *real* broadcast error of each satellite into its equation as a
range error, solve the least-squares problem, and read off how far the solution moves
from the truth. That is the whole thing — no approximations about what "matters",
just the geometry a receiver actually solves.

One extra detail that matters: a multi-constellation receiver estimates **one clock
unknown per constellation** (an inter-system bias), because the four systems keep
different time scales. We do the same.
""")

    co(r"""
A_WGS, F_WGS = 6378137.0, 1/298.257223563
E2 = F_WGS*(2 - F_WGS)

def llh2ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    N = A_WGS/np.sqrt(1 - E2*np.sin(la)**2)
    return np.array([(N+h)*np.cos(la)*np.cos(lo), (N+h)*np.cos(la)*np.sin(lo),
                     (N*(1-E2)+h)*np.sin(la)])

def enu_basis(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return (np.array([-np.sin(lo), np.cos(lo), 0.0]),
            np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)]),
            np.array([ np.cos(la)*np.cos(lo),  np.cos(la)*np.sin(lo), np.sin(la)]))

LAT, LON, HGT = 28.6139, 77.2090, 216.0            # New Delhi
RX = llh2ecef(LAT, LON, HGT)
EE, NN, UU = enu_basis(LAT, LON)
SYSL = ["G","E","C","J"]

def geometry(df):
    # attach line-of-sight direction, elevation, and the LOS projected into RTN
    df = df.sort_values(["sat","epoch"]).reset_index(drop=True)
    P = df[["px","py","pz"]].to_numpy()
    t = df.epoch.astype("int64").to_numpy()/1e9
    V = np.zeros_like(P)
    for _, g in df.groupby("sat", sort=False):
        i = g.index.to_numpy()
        V[i] = np.gradient(P[i], t[i], axis=0)
    eR = P/np.linalg.norm(P, axis=1, keepdims=True)
    hh = np.cross(P, V); eC = hh/np.linalg.norm(hh, axis=1, keepdims=True)
    eA = np.cross(eC, eR)
    LOS = P - RX
    U = LOS/np.linalg.norm(LOS, axis=1, keepdims=True)
    df["elev"] = np.degrees(np.arcsin(U @ UU))
    df["ux"], df["uy"], df["uz"] = U[:,0], U[:,1], U[:,2]
    df["uR"] = np.einsum("ij,ij->i", U, eR)
    df["uA"] = np.einsum("ij,ij->i", U, eA)
    df["uC"] = np.einsum("ij,ij->i", U, eC)
    return df

def fix_error(g, ecol):
    # one epoch: least-squares position error, with one clock unknown per system
    n = len(g); pres = [s for s in SYSL if (g.sys == s).any()]
    if n < 4 + len(pres): return None
    Cm = np.zeros((n, len(pres)))
    for k, s in enumerate(pres): Cm[(g.sys == s).to_numpy(), k] = -1.0
    G = np.column_stack([g.ux, g.uy, g.uz, Cm])
    x, *_ = np.linalg.lstsq(G, g[ecol].to_numpy(), rcond=None)
    dr = x[:3]
    return np.array([dr @ EE, dr @ NN, dr @ UU])

def run_fix(sel, ecol, tag, quiet=False):
    O = np.array([r for _, g in sel.groupby("epoch", sort=True)
                  if (r := fix_error(g, ecol)) is not None])
    hor = np.hypot(O[:,0], O[:,1])
    if not quiet:
        print(f"  {tag:36s} H {np.sqrt((hor**2).mean()):6.3f} m   "
              f"V {np.sqrt((O[:,2]**2).mean()):6.3f} m   "
              f"3D {np.sqrt((O**2).sum(1).mean()):6.3f} m")
    return O

SPP = geometry(d[(d.epoch >= "2026-08-01") & (d.epoch < "2026-09-01")].copy())
VIS = SPP[SPP.elev > 10].copy()
print(f"simulated receiver at New Delhi: {VIS.epoch.nunique():,} epochs, "
      f"{len(VIS)/VIS.epoch.nunique():.1f} satellites above 10 deg on average")
""")

    md("### First, three controls — to show the simulator behaves like a receiver")
    co(r"""
ctl = VIS[VIS.sys == "G"].copy()
rng = np.random.default_rng(0)

print("=== controls (GPS only) ===")
ctl["c1"] = 10.0
_ = run_fix(ctl, "c1", "a constant 10 m on EVERY satellite")
ctl["c2"] = 10.0*ctl.uR
_ = run_fix(ctl, "c2", "a 10 m RADIAL offset on every satellite")
ctl["c3"] = rng.normal(0, 1.0, len(ctl))
_ = run_fix(ctl, "c3", "white 1 m, independent per satellite")
""")

    md(r"""
**What the controls show — and this is the heart of the answer.**

1. **A constant error on every satellite produces *exactly zero* position error.**
   The receiver cannot tell a common range error from its own clock being wrong, so
   it absorbs the whole thing into its clock unknown. This is not an approximation;
   it is 0.000 m.

2. **A 10 m *radial* offset on every satellite produces 0.014 m horizontal and
   0.354 m vertical.** Roughly 96% of it is absorbed too, because from the ground all
   the satellites are broadly "up", so a shared radial error looks nearly like a
   shared range error. What leaks is a small vertical bias.

3. **White noise of 1 m per satellite produces about 1 m horizontal and 1.7 m
   vertical.** Errors that differ *between* satellites are the ones that survive, and
   they come through roughly one-for-one, amplified by geometry.

**So the rule is: only the satellite-to-satellite *differences* matter.** Anything
shared is free. Now apply that to the two things we just spent Parts 3 and 4 finding.
""")

    md("### Now the real thing")
    co(r"""
ax_off = pd.read_csv("antex_offsets.csv").set_index("prn").z_if_m
VIS["z_off"] = VIS.sat.map(ax_off)
have = VIS.dropna(subset=["z_off"]).copy()
print(f"ANTEX offset available for {have.sat.nunique()} of {VIS.sat.nunique()} PRNs\n")

# per-satellite constant clock offset: broadcast group delay / datum convention.
# It is NOT a time-varying broadcast error, and a real receiver never sees it.
have["clk_const"] = have.groupby("sat").clk.transform("median")

def los(dR, dA, dC, clk, g):
    return dR*g.uR + dA*g.uA + dC*g.uC - clk

# (a) the difference we measured, taken literally: broadcast(APC) - SP3(CoM)
have["e_literal"] = los(have.dR, have.dA, have.dC, have.clk, have)
# (b) ANTEX applied, so both sides refer to the same point on the spacecraft
have["e_antex"]   = los(have.dR + have.z_off, have.dA, have.dC, have.clk, have)
# (c) also drop the per-satellite constant clock offset (group delay convention)
have["e_user"]    = los(have.dR + have.z_off, have.dA, have.dC, have.clk - have.clk_const, have)

print("=== position error at New Delhi, all constellations ===")
_ = run_fix(have, "e_literal", "broadcast - SP3, taken literally")
_ = run_fix(have, "e_antex",   "+ ANTEX applied")
_ = run_fix(have, "e_user",    "+ per-satellite clock constant removed")

print("\n=== GPS only ===")
gg = have[have.sys == "G"]
_ = run_fix(gg, "e_literal", "broadcast - SP3, taken literally")
_ = run_fix(gg, "e_antex",   "+ ANTEX applied")
_ = run_fix(gg, "e_user",    "+ per-satellite clock constant removed")

print("\n=== how big is the ANTEX correction in the range domain? ===")
for s in "GEC":
    m = have[have.sys == s]
    print(f"  {NAME[s]:8s} range-error RMS  literal {np.sqrt((m.e_literal**2).mean()):7.3f}  "
          f"ANTEX {np.sqrt((m.e_antex**2).mean()):7.3f}  "
          f"+clk const {np.sqrt((m.e_user**2).mean()):7.3f}  (metres)")
""")

    md(r"""
**What this shows — the direct answer to the question.**

**The antenna phase centre offset is not a user error.** Broadcast is *supposed* to
give the antenna phase centre, because that is where the signal is transmitted from,
and that is exactly what the receiver wants. The −1.1 m we measured is the difference
between two products describing two different points on the spacecraft — a
reference-frame difference, not something anyone experiences. And even if you got it
wrong, control 2 says the receiver would absorb most of it anyway: applying ANTEX
moves the GPS-only fix from 1.07 m to 0.94 m horizontal — **about 13 cm, from a
per-satellite offset of 1.1 m that is larger still in the range domain.**

**The shared clock signal of Part 3 is also not a user error.** It is common to every
satellite in the product, and control 1 says a common error contributes exactly zero
to position. The receiver's clock unknown eats it.

**The per-satellite clock constant is a bookkeeping convention too.** BeiDou's
range-error RMS drops from about 12.6 m to about 0.9 m once a per-satellite constant
is removed. A real receiver applies the broadcast group delay and never sees that
12 m; it is an artefact of our differencing two products with different frequency
conventions.

**So of the headline "7.1 m of broadcast error", the overwhelming majority is
invisible to a positioning user.** What is left — and it is the only thing that was
ever worth modelling — is the part that differs between satellites and changes with
time. That residual is what Part 8 measures, and what Part 10 tests.

This is a deflationary result and it is the most useful thing in this notebook. It
means the project's real headline is not "90% error reduction"; it is whatever the
model does to the numbers in the *third* row above.
""")
