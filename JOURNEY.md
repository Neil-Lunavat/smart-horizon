# SMART-HORIZON — GNSS broadcast error correction

**An exploration log: what we tried, what broke, what survived, and why.**

Written 2026-09-04. Every number here came from a run on this machine or the
rented GPU; where a number was later found to be misleading, both the original
and the correction are shown rather than the original being quietly replaced.

---

## 0. The problem as handed to us, and why we abandoned it

SIH problem statement PS-08 supplies six CSVs in `Data_PS-08/` — GEO and MEO
satellite error series for 1–10 Sept 2025, a few hundred rows each — and asks for
a model that takes 7 days of error and predicts the 8th day at arbitrary
timestamps. Scoring is the **Shapiro-Wilk W statistic** on the residuals, with
mean/std as a tiebreak.

Two problems killed this framing.

**The supplied data is far too small.** After deduplication (the MEO files contain
literal duplicate rows — `DATA_MEO_Train.csv` has 90 rows but only 46 unique
timestamps) there are 46 to 143 usable points per file. Nothing that deserves the
name "model" can be fitted to that.

**The task is operationally impossible as specified.** Error = broadcast − precise.
Computing the *input* therefore requires the precise (SP3) product. If SP3 lagged
by two weeks, a "prediction" for day 8 could only be made two weeks after day 8.
There is no user for that.

So we reframed. Rather than predicting error from error, learn the mapping from
**broadcast** (available in real time) to **error**, using a large corpus where
both are known. Then at deployment, feed today's broadcast and get today's
correction. The SP3 lag then only affects offline training, where it costs nothing.

> **Later correction (§9):** the two-week lag premise was wrong. JAXA final SP3 is
> available at a **3-day** lag. Measured, not assumed. This did not invalidate the
> reframing but did change the operating point substantially.

---

## 1. The corpus

`fetch_all.py` → `build_errors.py` → `windows.py` was already built and produced
`full.parquet`. We extended it from 236 to **244 days** (1 Jan – 1 Sep 2026),
**2,067,234 rows**, 100 satellites, on a uniform 900 s grid.

| stage | file | what it does |
|---|---|---|
| 1 | `fetch_all.py` | downloads `BRDM00DLR` broadcast nav + `JAX0MGXFIN` precise orbits from CDDIS |
| 2 | `build_errors.py` + `gnss.py` | evaluates broadcast per ICD, differences against SP3 |
| 3 | `windows.py` | memory-safe sliding windows via `sliding_window_view` |

JAXA specifically because it is **the only final MGEX product containing GEO
satellites** — CODE, GRG, IAC, WUM and GFZ all start BeiDou at C06 and carry no
GEO at all. No JAXA, no J07/J08, no GEO data.

No CLK files are downloaded: the clock column inside SP3 agrees with the 30-second
CLK product to **1.6 picoseconds**, ~1000× below the signal being measured.

### Data-integrity problems found

1. **PRN reuse.** `C13`, `C38`, `C39` are each **two different spacecraft**. Mean
   orbital radius jumps 42,164 km → 27,906 km across a gap in May. All three flip
   on the same date (2026-06-08), which is the giveaway that it is a constellation
   change and not physics. Every downstream script now splits them into
   `C13_a`/`C13_b` etc. Grouping by PRN silently splices two satellites together.
2. **Composition drift.** 97 satellites in January, 79 by early June, 91 after.
   Only **79 of 103 spacecraft are present in all 8 months**. BeiDou GSO is 7
   satellites in Jan–Feb but **1** in Mar–Apr. Month-over-month comparisons are
   partly measuring composition, not behaviour.
3. **Gross clock outliers not caught by the existing mask.** C10 reaches
   −133,309 m on 2026-01-19 (9 consecutive epochs); C06 reaches −3,898 m. One such
   row moves a satellite's clock RMS by four orders of magnitude. A flat
   `|clock| > 100 m` cut fixes it: C10 goes from 6,640 m RMS to 0.98 m.
4. **`anom_z` is unreproducible.** It is read by `windows.py` and present in the
   parquet, but produced by no script in the repo. We regenerated an equivalent
   robust score in `taskA_data.py`.

---

## 2. First analysis — what the errors actually look like

Almost all of the raw error is a **static per-satellite bias**:

| group | dR RMS | dR sd about a 7-day mean |
|---|---|---|
| GPS MEO | 1.44 m | **0.16 m** |
| Galileo MEO | 0.84 | **0.12** |
| BeiDou MEO | 1.14 | **0.07** |
| QZSS GEO | 3.18 | **0.52** |

**Consequence:** any skill metric computed against "predict zero" is measuring the
bias, not the model. Our first pass reported 0.998 "skill" for GPS radial purely
from this. Every subsequent number is scored against a per-satellite baseline.

### Periodicity is orbital, not daily

| group | ACF peak lag | interpretation | r(dR) |
|---|---|---|---|
| GPS MEO | **12.00 h** | orbital period | +0.87 |
| BeiDou MEO | **12.75 h** | orbital period | +0.69 |
| Galileo MEO | **239.5 h (9.98 d)** | 17-rev/10-day ground-track repeat | +0.62 |
| BeiDou/QZSS GSO, QZSS GEO | **24.00 h** | orbital period = day | +0.79 … **+0.94** |

**A 7-day input window does not contain one Galileo repeat cycle.** GPS gets 14
cycles, BeiDou MEO 13, GEO 7, Galileo 0.7. Architectural fact, not a tuning knob.

### Ephemeris age carries no signal

`age_s` explains **R² = 0.000** of variance, in every group, for every target. RMS
binned by age is flat from 0 to 3600 s. There is no "error builds up as the message
ages" sawtooth, because arcs refresh every 10 min (Galileo) to 2 h (GPS).

### 65–78% of the clock signal is a product artifact

Decomposing per-satellite clock into an epoch-wise constellation mean plus residual:

| | GPS | Galileo | BeiDou MEO |
|---|---|---|---|
| common mode share of variance | **75%** | **78%** | **65%** |

Orbit components show *no* common mode (−0.02 to −0.05). Then the decisive check —
correlation *between* constellations' common-mode series:

```
       GPS    GAL    BDS    QZS
GPS  1.000  0.963  0.914  0.923
GAL  0.963  1.000  0.910  0.892
BDS  0.914  0.910  1.000  0.854
QZS  0.923  0.892  0.854  1.000
```

Four independent ground segments with four independent clock ensembles **cannot**
share a signal at r = 0.85–0.96. This is the SP3 clock datum — the analysis
centre's per-epoch reference alignment — imprinted on every satellite. Removing a
single global common mode drops per-satellite clock σ from 0.411 m to 0.230 m.

Its ACF: +0.968 at 15 min, **+0.003 at 24 h**. Most of the clock variance is a
bookkeeping convention that is smooth at short lag and memoryless at one day.

### Effective sample size

`window_index.npz` advertises 1,701,887 windows, but at 15-minute stride
consecutive windows share 767 of 768 samples. Truly non-overlapping 8-day
stretches: **~1,940**. Daily-origin windows: 15,524. This is why random splits are
dangerous here (§10).

---

## 3. Pivot 1 — from forecasting to correction

The competition frames a 24-hour-ahead forecast. But on day 8 the day-8 broadcast
is *already there*: it is the freshest information about that epoch and the
original framing throws it away.

Recast as a **nowcast**: given the broadcast in force right now plus history, how
wrong is it? This is the "broadcast ephemeris improvement" / SISRE-correction
problem from the literature, and it is far easier than forecasting.

Measured difference between the two tasks:

| | nowcast (3-day lag) | 24 h forecast |
|---|---|---|
| radial | 1.295 → **0.105** (92%) | ~0.15 (82%) |
| clock | 7.046 → **0.818** (88%) | **~0 skill** |

The clock has **no 24-hour memory** (common-mode ACF 0.003 at 24 h). That is the
process, not a modelling shortfall. Nowcasting sidesteps it.

---

## 4. The first real finding — it is the antenna

Per-satellite median radial offsets clustered by hardware block, so we tested the
obvious physical explanation.

**Broadcast ephemeris refers to the satellite's antenna phase centre. Precise SP3
orbits refer to its centre of mass.** The difference is a published per-satellite
constant, along body-frame Z, which points at nadir — so it lands almost entirely
in *radial* and barely at all in along/cross-track.

Downloaded IGS `igs20.atx` (60 MB) **after** measuring, and compared:

| block | published z | measured | residual | scatter |
|---|---|---|---|---|
| BLOCK IIF | 1.438 | 1.443 | **−0.004 m** | 0.017 |
| BLOCK IIR-A | 1.163 | 1.150 | +0.012 | 0.011 |
| BLOCK IIR-B | 0.700 | 0.681 | +0.020 | 0.042 |
| BLOCK IIR-M | 0.722 | 0.691 | +0.031 | 0.029 |
| BLOCK IIIA | 1.935 | 1.870 | +0.065 | 0.194 |
| GALILEO-2 | 0.809 | 0.824 | −0.015 | 0.064 |
| BEIDOU-3M-SECM | 1.096 | 1.104 | −0.008 | 0.175 |

**GPS: R² = 0.949, residual scatter 0.110 m. Block IIF agrees to 4 mm across 11
satellites.** The ordering across five GPS blocks is reproduced exactly.

Blind-testing every frequency convention also recovered the ICD: GPS matches the
**L1/L2 ionosphere-free** combination (0.112 m RMS vs 0.407 for L1 alone), Galileo
matches **E5b** (0.072 m). The ANTEX header independently confirms this
("L1 and L2 set to the results for the ionosphere-free linear combination").

**This is not our discovery.** It is a standard, documented correction —
Montenbruck et al., *Broadcast versus precise ephemerides: a multi-GNSS
perspective*, GPS Solutions 2015, which ANTEX itself cites and which "derives
block-/constellation-specific radial antenna offset values for the center-of-mass
correction of broadcast orbits". We re-derived a published table from data. That
should *raise* confidence in the pipeline and *lower* any claim of novelty.

Output: `satellite_offset_table.csv`, 103 spacecraft.

**Known bugs in that table:** BeiDou-2 IGSO (+1.59 m residual) and QZSS-2I
(−0.65 m) use the wrong frequency convention; and the ANTEX join is done at a
single epoch, so the three PRN-reuse rows (`C13_a`, `C38_a`, `C39_a`) are matched
to the *new* spacecraft. Not yet fixed.

### The uncomfortable consequence

| radial correction | RMS |
|---|---|
| raw broadcast − precise | 1.295 m |
| minus published ANTEX offset (a lookup table) | **0.224 m** |
| minus per-satellite median from training | 0.184 m |
| full XGBoost model | 0.144 m |

**~86% of the radial "win" is a table lookup**, which a GNSS practitioner would
apply before calling the difference an error at all. The ML earns 0.184 → 0.144.

---

## 5. Two physics hypotheses, both falsified

Both were stated with falsifiable predictions *before* running.

**Sun geometry.** ANTEX offsets are 3-D and the header says the convention exists
for yaw-steering attitude; solar radiation pressure mismodelling is the documented
driver of along-track error. Prediction: along-track error spikes at low |β|
(eclipse season).
**Result: flat.** GPS 0.915 m at |β|<2°, 0.883 m at |β|>60°. R² = 0.008. Dead.

**Upload age.** GPS uploads a new orbit solution 1–2×/day; ephemerides between are
re-issues. Prediction: error grows with time since a *new upload* (proxied by a
large handover jump), even though it does not grow with `age_s`.
**Result: flat.** 0.87 / 0.90 / 0.89 / 0.89 / 0.91 m across 0→12 h. And GPS
handovers are not bimodal (median jump 0.43 m, 78% over 20 cm), so there is no
re-issue/upload distinction to exploit. Dead.

---

## 6. Pivot 2 — along-track does not matter

Chasing along-track was the wrong target all along. What a receiver experiences is
**SISRE**, in which radial and clock enter directly but along/cross are
down-weighted ~7× because they are nearly perpendicular to the line of sight
(weights from Montenbruck 2015: MEO wR ≈ 0.98, wA,C ≈ 0.139).

Share of GPS SISRE²: **radial+clock 91.9%, along-track 6.0%, cross-track 2.1%.**
For Galileo and BeiDou along-track is 0.1%.

Zeroing GPS along-track *perfectly* moves SISRE from 0.557 → 0.540 m. Two failed
hypotheses had been spent on a 3% term. Along-track was dropped.

---

## 7. Feature engineering — the handover jump

One broadcast-only feature is not in the raw data and had to be built
(`handover.py`, 1,000,746 events): when a new nav message takes over, evaluate the
**outgoing** and **incoming** records at the same instant. Their disagreement is
the ground segment publicly correcting itself. It needs no SP3 and is known the
moment the new message arrives.

Median jump is only ~2 mm, so the prediction was that it would be useless. **Wrong
— it added 11 points to radial skill (71% → 82%).** But not in the expected way:
the model's top features were `ho_R_s7`, `ho_C_s7`, `ho_A_s7` — the **7-day rolling
standard deviation** of the jumps, not the jumps. It reads how *jittery* the ground
segment has been lately, i.e. how much to distrust the current message.

---

## 8. The model ladder

All models see identical information: broadcast up to *t*, truth up to *t* − LAG.

| model | SISRE (single window) | note |
|---|---|---|
| raw broadcast | 7.393 | do nothing |
| 7-day window mean | 0.417 | the baseline to beat |
| **XGBoost** | 0.453 | **lost to the baseline** |
| MLP (flattened sequence) | 0.358 | |
| GRU (2-layer, 128) | 0.380 | **lost to the MLP** |
| TCN (dilated conv) | 0.337 | **ties v3**; best on along/cross |
| v2 (per-system scaling + Huber + blend) | 0.353 | |
| v3 (heteroscedastic Gaussian) | **0.337** | best RMS |
| v4 (Student-t) | 0.345 | best calibration |

Two negative results worth keeping: **gradient boosting lost to a 7-day average**,
and **a recurrent net lost to a plain MLP given the same data flattened**. The
signal is not sequence-shaped; it is constants plus slow drift.

### v1 → v2: the scale-domination bug

v1 pooled all constellations under plain MSE. BeiDou clock error is 12 m and
Galileo's is 0.38 m, so BeiDou contributed ~1000× the squared error and Galileo
was optimised as if it were noise — **v1 made Galileo clock worse than doing
nothing** (0.383 → 0.459 m). Fix: divide each target by its per-system residual sd
inside the loss. Galileo went 0.455 → 0.281 SISRE.

---

## 9. The latency discovery

We had assumed a 2-week SP3 lag. Probing CDDIS directly on 4 Sept:

| product | oldest available | lag |
|---|---|---|
| GFZ0MGXRAP (rapid) | 2026-09-02 | **2 days** |
| COD0OPSRAP (rapid) | 2026-09-02 | **2 days** |
| **JAX0MGXFIN (the one we use)** | 2026-09-01 | **3 days** |

The corpus ended 24 Aug because that is when it was last downloaded, not because
the product was unavailable. The skill-vs-latency curve:

| SP3 latency | radial | clock |
|---|---|---|
| 6 h | 0.103 | **0.473** |
| 1 d | 0.096 | 0.555 |
| **3 d** | 0.105 | **0.818** |
| 7 d | 0.112 | 1.131 |
| 14 d | 0.124 | 1.022 |
| 28 d | 0.137 | 1.803 |

Clock is 92% of SISRE, so moving from the assumed 14-day to the real 3-day
operating point is worth more than any modelling change made in this project.
(The 7-day-worse-than-14-day inversion is unexplained and flagged, probably the
BeiDou June regime change interacting with the window.)

Corpus re-fetched to 1 Sept: 244 days.

---

## 10. Pivot 3 — the residuals were Gaussian all along

v2 produced pooled clock residuals with **kurtosis 438, Shapiro W 0.507**. That
looks like catastrophic non-normality. It was a measurement error.

| | dR | dA | dC | clock |
|---|---|---|---|---|
| Shapiro W **within one satellite** (median) | **0.9948** | **0.9910** | **0.9943** | **0.9882** |
| Shapiro W **pooled across satellites** | 0.9089 | 0.8169 | 0.7795 | 0.5072 |

Satellite residual scales differ by **11× to 22×** (dR sd 0.029 → 0.403 m; dA
0.117 → 2.593 m). **Pooling Gaussians of different variance produces a scale
mixture, which is heavy-tailed by construction.** Every component was normal; the
histogram was not.

Standardise each satellite by its own sd and pool again:

| target | skew | kurtosis | Shapiro W |
|---|---|---|---|
| dR | −0.02 | 0.2 | **0.9997** |
| dA | 0.06 | 0.3 | **0.9981** |
| dC | −0.02 | 0.3 | **0.9983** |

So the fix is not a different loss on the mean — it is to **model the scale**. Huber
was the wrong instinct: it tells the model to ignore erratic satellites rather than
to represent them.

Crucially, scale is **predictable from broadcast alone**: handover-jump sd
correlates with residual sd at **r = 0.69 (radial)** and **r = 0.54 (clock)**. No
SP3 needed at prediction time.

### v3 — heteroscedastic, Gaussian NLL

The net emits a mean *and* a log-σ per target.

**SISRE 7.393 → 0.337 m.** Standardised residuals: dR W = 0.9895, dA 0.9888,
dC **0.9981**. Coverage: 72.5 / 95.7 / 99.3% against nominal 68.3 / 95.4 / 99.7.

### v4 — Student-t + PIT

Clocks genuinely jump (G27 swings ±3 m day to day), so a Gaussian predictive law
cannot be right for them. Student-t with learned degrees of freedom: the model
independently chose ν = 11.0 (dR), 12.1 (dA), 15.7 (dC), **7.8 (clock)** — finding
that clocks have the heaviest tails, matching the diagnosis.

PIT normal scores — transform each observation through its own predictive CDF then
inverse-normal; if the predictive law is correct these are exactly N(0,1):

| target | skew | kurt | Shapiro W | p |
|---|---|---|---|---|
| dR | −0.02 | 0.2 | 0.9987 | 0.0023 |
| dA | 0.07 | 0.2 | 0.9989 | 0.0095 |
| **dC** | −0.00 | 0.1 | **0.9994** | **0.2264** |
| clock | −0.15 | 2.0 | 0.9794 | 0.0000 |

dC **fails to reject normality outright**. Coverage 52.9 / 71.4 / 96.5 / 99.7%
against nominal 50 / 68.3 / 95.4 / 99.7.

v4 costs a little RMS versus v3 (0.345 vs 0.337) and buys calibration. Real
tradeoff, not a free lunch.

### G27 is simply a bad satellite

25% of its rows sit in the global tail. Daily median clock error swings ±3 m. Model
RMS 1.042 vs raw 1.044 — nothing helps because there is nothing to learn. Its
broadcast handover sd is 0.413 m against G04's 0.095. The honest output is a wide
error bar, which is exactly what the heteroscedastic model produces.

---

## 11. The validation reckoning

Everything above was reported on **one 8-day test window (24 Aug – 1 Sep)**. That
was wrong in two ways: the window is thin, and four model versions were designed
while looking at its score — model selection on the test set.

Forced to validate properly, `validate.py`:

### Rolling-origin folds

| fold | test SISRE | baseline | train | gap |
|---|---|---|---|---|
| fold0 @12000 | 0.455 | 0.501 | 0.558 | −0.103 |
| **fold1 @14400** | **1.595** | 1.655 | 0.573 | **+1.022** |
| fold2 @16800 | 0.614 | 0.700 | 0.714 | −0.100 |
| fold3 @19200 | 0.520 | 0.537 | 0.773 | −0.253 |
| **fold4 @21600** | **0.357** | 0.401 | 0.604 | −0.247 |

**Mean 0.708, sd 0.451.** The window reported all session (fold4, 0.357) is the
**easiest in the corpus**. Honest performance is roughly **double** what had been
claimed. The model still beats the baseline in **5/5 folds**, including the awful
fold1 — that is the finding that survived.

### Random 80/20 vs time split

| split | raw | baseline | test | model/baseline |
|---|---|---|---|---|
| RANDOM 80/20 | 7.272 | 0.760 | 0.611 | **0.804** |
| TIME | 7.393 | 0.417 | 0.367 | 0.880 |

Absolute numbers are not comparable (the random test set spans hard periods, the
time test set is one easy week). Normalised by each one's own baseline, random gets
19.6% over baseline and time gets 12.0% — the ~8-point gap is leakage from
near-duplicate neighbours. Random splits *do* leak here, but less dramatically than
first asserted.

### Satellite holdout

| | test | train | gap |
|---|---|---|---|
| unseen sats, seen time | 0.894 | 0.535 | +0.359 |
| unseen sats + future | 0.370 | 0.536 | −0.165 |
| seen sats + future | 0.375 | 0.537 | −0.162 |

Unseen satellites cost essentially nothing in the recent period (0.370 vs 0.375).
**The no-memorisation requirement holds** — no satellite ID is used anywhere, and
dropping `r_km`/`latf` changes results by <0.5%.

### Blocked Monte-Carlo CV — the definitive number

24 random contiguous 2-week test blocks, model retrained from scratch per block,
training data strictly before the block with an 8-day gap so no window straddles a
boundary.

| quantity | value |
|---|---|
| raw broadcast SISRE | 7.127 m |
| baseline SISRE | 0.798 ± 0.461 m |
| **MODEL SISRE** | **0.705 ± 0.433 m** (median 0.558) |
| model range across folds | [0.364, 1.806] |
| **95% interval** | **[0.365, 1.787]** |
| **vs raw broadcast** | **90.1% reduction** |
| vs baseline | 11.7% mean improvement |
| **beats baseline in** | **24 / 24 folds** |
| paired diff (baseline − model) | +0.0932 ± 0.0570, **t = 8.01** |
| overfit gap (test − train) | +0.181 ± 0.451 |

Read this carefully, because the two halves say different things:

- **Against raw broadcast: 90.1%, and it is not fragile.** Even the worst of 24
  folds (1.806 m) is a 75% reduction on 7.127 m.
- **Against the 7-day-mean baseline: 11.7%, small but utterly consistent.**
  24/24 folds and t = 8.01 means the improvement is real and highly significant,
  but it is a tenth, not a doubling. The margin also depends strongly on how much
  training history exists: ~0–2% on the earliest blocks, 11–23% on later ones.

The fold-to-fold spread (sd 0.433, range 5×) is the honest headline. Any single
window can be sold as 0.36 or as 1.81 depending on which one is chosen. That is
exactly the trap this project fell into before the folds were run.

---

## 12. Where it stands

### The one-line answer

**SISRE 7.127 m → 0.705 ± 0.433 m (90.1% reduction), 24/24 folds beating the
baseline at t = 8.01, with calibrated and verifiably Gaussian residuals.** Most of
that reduction is a published physical constant; the learned model contributes a
consistent 11.7% on top.

**Solid, survived every test:**

1. **Antenna phase centre offset.** Physics, not ML. Matches IGS published values
   to 4 mm on Block IIF. Radial 1.300 → 0.224 m from a lookup table.
2. **Residuals are Gaussian**, verified three ways (per-satellite Shapiro ≈ 0.99;
   standardised pooling W = 0.9997; PIT normal scores W = 0.9987–0.9994 with
   dC at p = 0.226). This is the direct answer to a Shapiro-Wilk-scored competition.
3. **Calibrated uncertainty** — coverage within ~1–3 points of nominal at every level.
4. **Beats the baseline in 24/24 blocked MC-CV folds** (t = 8.01), by 11.7% on average.
5. **SP3 latency is 3 days**, and the latency curve says that is worth more than
   any architecture change here.
6. **Generalises to unseen satellites** with no ID feature.

**Uncertain / not yet established:**

- The model-over-baseline margin is **11.7%** — real (24/24 folds, t = 8.01) but
  modest, and strongly dependent on training-set size (~0–2% on the earliest
  blocks, 11–23% on later ones).
- Fold-to-fold variance is large (sd 0.433 on a mean of 0.705). Performance on any
  given fortnight is only loosely predictable.
- Along-track remains unexplained (2–10% skill) — deliberately abandoned as 6% of SISRE.
- BeiDou clock regime changes (the June jump, ~10–26 m) are handled by lagged
  features but not understood.

**Corrections made to earlier claims in this document:**

- "2-week SP3 lag" → actually 3 days (§9).
- "88% radial reduction from the model" → 86 of those points are a lookup table (§4).
- "SISRE 0.337" → that was the best of 5 folds. Definitive 24-fold answer:
  **0.705 ± 0.433 m, 90.1% below raw broadcast, 11.7% below the baseline** (§11).
- "handover jumps will be useless" → wrong, worth 11 points, via their rolling sd (§7).
- "sequence models will win on along-track" → GRU lost to MLP (§8).

---

## 13. Files

| file | purpose |
|---|---|
| `fetch_all.py` | stage 1 — download broadcast + precise |
| `gnss.py` | ICD orbit evaluation, SP3 reader, RTN rotation |
| `build_errors.py` | stage 2 — broadcast−precise error series |
| `windows.py` | stage 3 — memory-safe sliding windows |
| `handover.py` | broadcast-only handover-discontinuity features |
| `antex.py` / `offset_table.py` / `build_table.py` | antenna offset table |
| `sungeom.py` | sun geometry / β angle (hypothesis falsified) |
| `taskA_data.py` | builds `taskA.npz`, the dense modelling grid |
| `taskA_tab.py` | baselines + ridge + XGBoost ladder |
| `taskA_nn.py` | MLP / GRU / TCN ladder |
| `taskA_v2.py` | per-system scaling + Huber + baseline blend |
| `taskA_v3.py` | heteroscedastic Gaussian NLL |
| `taskA_v4.py` | Student-t + PIT calibration |
| `validate.py` | random-vs-time split, rolling folds, satellite holdout |
| `mccv.py` | blocked Monte-Carlo cross-validation |
| `export_csv.py` / `export_share.py` | CSV exports for teammates |
| `satellite_offset_table.csv` | 103 spacecraft, measured vs published offsets |

**Compute:** local RTX 3050 6 GB, plus a rented RTX PRO 4500 Blackwell (32 GB).
Note the pod shipped torch 2.4.1+cu124 which compiles only to `sm_90`; Blackwell is
`sm_120` and **every CUDA op failed** until torch was upgraded to 2.11.0+cu128.

---

## 14. Method notes for whoever picks this up

- **Never score against zero.** ~90% of raw error is a static bias; scoring against
  zero reports ~0.99 skill for having learned a constant.
- **Never use a random split.** Windows overlap 99.7% at 15-minute stride; there
  are only ~1,940 genuinely independent 8-day stretches in 244 days.
- **Never pool residuals across satellites to test normality.** Scales differ 11–22×;
  the pooled histogram is a scale mixture and will look non-Gaussian no matter how
  good the model is.
- **Report per-system.** BeiDou's 12 m clock error swamps Galileo's 0.38 m in any
  pooled loss or pooled metric.
- **Weight by SISRE, not by component.** Along/cross-track are ~7× down-weighted;
  optimising them is optimising 3–6% of what a user feels.
