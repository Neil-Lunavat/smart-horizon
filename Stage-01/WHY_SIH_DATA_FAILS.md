# Why the PS-08 Dataset Cannot Support the Stated Task

**A data-exploration finding, with every claim reproduced from the delivered files.**

Problem statement PS-08 asks us to learn the time-varying build-up of broadcast
ephemeris and clock error from **seven days of data for one satellite**, and to
predict the eighth day at arbitrary timestamps.

We spent the exploration phase testing whether the delivered dataset can support
that. It cannot — and the reasons are not "the data is small" or "the data is
noisy". They are structural. This document separates three classes of problem:

- **Part I — the files as delivered are defective.** Duplicated rows, under-coverage, and a test set drawn from a different distribution than the training set.
- **Part II — the GEO file is not a physical time series at all.** It contains an artefact that no orbit or clock can produce.
- **Part III — even a *perfect* seven-day file is structurally insufficient.** The quantities that actually drive the error are unobservable within a seven-day, single-satellite, irregularly-sampled window.

Part III is the important one. Parts I and II could in principle be fixed by the
organisers. Part III cannot be fixed by anyone, because it is a property of the
observation window, not of the file.

Every number below is reproducible from `Data_PS-08/` and from our 236-day
reference corpus (`full.parquet`, 1,996,787 epochs, 100 satellites,
2026-01-01 → 2026-08-24, built from CDDIS broadcast navigation messages
differenced against JAX0MGXFIN final precise orbits).

---

## Summary

| # | Finding | Evidence |
|---|---|---|
| 1 | Every MEO file is ~50% duplicated rows | `MEO_Test` is 6 real points, not 11 |
| 2 | No file spans seven days; coverage is 13–32% | `MEO_Train` covers 13.5% of 168 h |
| 3 | Effective independent samples: **1.2 to 5.2** | `n_eff` after autocorrelation |
| 4 | Test day is a different distribution from the training week | GEO variance ratio 7.6×, Levene p = 5×10⁻⁷ |
| 5 | Training week is not stationary within itself | GEO second half 30× the variance of the first, p = 1×10⁻⁴ |
| 6 | No satellite ID → the applicable error physics is unknown | 4× spread in clock error across MEO satellites |
| 7 | **GEO file contains an unphysical sign-flip artefact** | consecutive rows antiparallel, cos = **−0.9994** |
| 8 | The same artefact appears **zero** times in 1,996,787 corpus rows | corpus cos = **+1.000**, 0.0000 antiparallel |
| 9 | **48–81% of ephemeris uploads are unobservable** at the given sample times | simulated on real arc boundaries |
| 10 | Age-of-ephemeris drives a 4.5–5.3× error envelope, and is **not provided** | corpus and SIH agree on the ratio |
| 11 | The window's own offset is **not identifiable**: clock R² = **0.000** | week-mean vs next-day-mean |
| 12 | 44–68% of clock variance sits at periods ≥ 1 day; up to 38% beyond 7 days | per-satellite PSD |
| 13 | **78%** of random 7-day windows contain no eclipse at all | shadow geometry over 236 days |
| 14 | A 7-day window contains a manoeuvre only **3.7%** of the time | 2.0 manoeuvres/satellite-year |
| 15 | **Every** polynomial clock fit on 7 days is worse than predicting zero | quadratic 0.265 m vs zero 0.249 m |

---

# Part I — The files as delivered

## 1. Every MEO file is roughly half duplicated rows

Each row appears twice, byte-identical across all four channels. These are not
two satellites sharing a timestamp — they are the same record repeated.

| file | rows | **unique** | wasted |
|---|---|---|---|
| `DATA_MEO_Train.csv` | 90 | **46** | 49% |
| `DATA_MEO_Train2.csv` | 244 | **143** | 41% |
| `DATA_MEO_Test.csv` | 11 | **6** | 45% |
| `DATA_MEO_Test2.csv` | 30 | **18** | 40% |
| `DATA_GEO_Train.csv` | 142 | 142 | — |
| `DATA_GEO_Test.csv` | 69 | 69 | — |

**`DATA_MEO_Test.csv` contains six data points.** Any metric computed on it —
Shapiro-Wilk included — is dominated by sampling noise.

## 2. No file actually spans seven days

"Seven-day training data" is not what was delivered. Coverage below is the union
of ±15 min windows around each retained sample, as a fraction of 168 hours:

| file | span (days) | gaps > 1 h | largest gap | hours covered | % of 7 days |
|---|---|---|---|---|---|
| `GEO_Train` | 6.74 | 70 | 6.0 h | 53.4 | **31.8%** |
| `MEO_Train` | 6.08 | 7 | 21.0 h | 22.7 | **13.5%** |
| `MEO_Train2` | 6.06 | 19 | 25.9 h | 30.0 | **17.9%** |
| `GEO_Test` | 0.98 | 4 | 2.0 h | 18.2 | 10.9% |
| `MEO_Test` | 0.21 | 0 | 1.0 h | 3.0 | **1.8%** |
| `MEO_Test2` | 0.23 | 3 | 1.2 h | 4.1 | 2.4% |

`MEO_Train` is seven runs of about seven hours each, separated by 21-hour holes.
Over 86% of the "training week" is simply absent. The "eighth day" in
`MEO_Test` is five hours of one afternoon.

## 3. Effective sample size is between 1 and 5

Raw row counts overstate the information content, because consecutive samples
are correlated. With `n_eff = n·(1−ρ)/(1+ρ)` at each file's own cadence:

| file | n | cadence | x | y | z | clock |
|---|---|---|---|---|---|---|
| `MEO_Train` | 46 | 1 h | **5.2** | **1.2** | **1.3** | 44.6 |
| `MEO_Train2` | 143 | 10 min | 70.6 | 40.8 | 30.9 | 100.8 |

`DATA_MEO_Train.csv` carries between **one and five independent observations**
of the along-track and cross-track error. The clock column looks better only
because it is uncorrelated — see §15, where that turns out to be a defect, not
a feature.

(The GEO files produce arithmetically inflated `n_eff` because their lag-1
autocorrelation is *negative*. That is the artefact of Part II, not information.)

## 4. The test day is not drawn from the training distribution

The premise of the task is that day 8 continues the process observed on days
1–7. For the GEO pair, it does not:

| channel | train std | test std | variance ratio | Levene p |
|---|---|---|---|---|
| x_error | 5.040 | 13.930 | **7.6×** | 5.2×10⁻⁷ |
| y_error | 8.703 | 19.761 | **5.2×** | 3.6×10⁻⁵ |
| z_error | 6.587 | 10.495 | 2.5× | 1.7×10⁻³ |
| satclockerror | 3.809 | 15.896 | **17.4×** | 2.7×10⁻³ |

A model fitted on the training week is being scored against a day whose error
variance is 2.5× to 17× larger, at p < 0.003 on every channel. This is not a
hard prediction problem; it is an ill-posed one.

## 5. The training week is not even stationary within itself

Splitting `GEO_Train` at its midpoint:

| channel | first half std | second half std | ratio | Levene p |
|---|---|---|---|---|
| x_error | 2.435 | 6.717 | **7.6×** | 3.2×10⁻⁴ |
| y_error | 2.209 | 12.151 | **30.3×** | 1.1×10⁻⁴ |
| z_error | 2.713 | 8.939 | **10.9×** | 2.9×10⁻⁴ |
| satclockerror | 2.127 | 4.970 | 5.5× | 2.4×10⁻² |

There is no single regime to learn. The "seven days" contain at least two
statistically distinct populations, and no label distinguishes them.

## 6. No satellite identifier

The files carry `utc_time, x_error, y_error, z_error, satclockerror` and nothing
else. No PRN, no constellation, no orbit slot. This matters because the error
physics is satellite-specific. Across MEO satellites in our corpus:

| constellation | median &#124;clock err&#124; (m) | median &#124;pos err&#124; (m) |
|---|---|---|
| BeiDou (C) | 0.148 | 1.151 |
| Galileo (E) | 0.092 | 0.881 |
| GPS (G) | 0.078 | 1.669 |

Per-satellite medians span **0.053 to 0.232 m** in clock (4×) and **0.772 to
2.406 m** in position (3.1×). Rubidium, caesium and passive-hydrogen-maser
clocks have qualitatively different drift laws; BeiDou, GPS and Galileo have
different upload cadences and fit intervals. Being told only "MEO" does not
determine which model applies.

---

# Part II — The GEO file is not a physical error series

This is the single most serious finding, and it is decisive on its own.

## 7. Consecutive samples are near-exact negations of each other

Lag-1 autocorrelation at 15-minute spacing:

| file | x | y | z | clock |
|---|---|---|---|---|
| `GEO_Train` | −0.651 | **−0.812** | −0.658 | **−0.849** |
| `GEO_Test` | −0.385 | −0.288 | −0.250 | −0.373 |

A satellite orbit error is a smooth function of time. At 15-minute spacing a
real series has ρ ≈ +0.99. A value of **ρ = −0.85** means the series *reverses
sign at every step*.

Looking at the four-vectors directly, consecutive rows are frequently exact
negatives:

```
dt = 240 s   A = [  6.48,  40.30,  39.23,  23.32]     cos = -0.9937
             B = [ -6.85, -41.10, -31.55, -23.44]     |B|/|A| = 0.94

dt = 960 s   A = [ 18.77,  27.92,  16.49,   4.65]     cos = -0.9994
             B = [-19.79, -29.96, -18.88,  -5.84]     |B|/|A| = 1.09

dt = 900 s   A = [-33.70,  45.44, -13.08,  55.87]     cos = -0.9993
             B = [ 36.59, -44.17,  13.53, -56.73]     |B|/|A| = 1.02
```

**All four channels flip sign together, with the same magnitude.** Position
error and clock error arise from physically independent mechanisms — orbit
dynamics versus onboard oscillator behaviour. They cannot reverse in lockstep,
and a geostationary satellite cannot move 40 m and back in four minutes.

Prevalence, among consecutive pairs where both vectors exceed 3 m:

- `GEO_Train`: **17%** antiparallel (cos < −0.7)
- `GEO_Test`: **40%** antiparallel; the *median* consecutive cosine is **−0.507**

The most likely mechanism is a sign-convention or switchover artefact: the
reported quantity is (uploaded − modelled) evaluated on both sides of a message
boundary, so the same discrepancy is emitted with both signs. Whatever the
cause, **the file is not a sampled time series of a physical error.**

Once the artefact is aliased away by the 2-hour sampling, essentially nothing is
left: lag-1 autocorrelation on the 2-hour grid is 0.05 / −0.18 / 0.14 / 0.23.

## 8. The artefact appears zero times in two million real epochs

We applied the identical test to our 236-day CDDIS-derived corpus:

| dataset | median cos(vₜ, vₜ₊₁₅ₘ) | fraction cos < −0.7 |
|---|---|---|
| Corpus MEO (40 sats) | **+0.996** | **0.0000** |
| Corpus GSO (7 sats) | **+0.999** | **0.0000** |
| Corpus GEO (2 sats) | **+1.000** | **0.0000** |
| SIH `GEO_Train` | +0.281 | 0.17 |
| SIH `GEO_Test` | **−0.507** | **0.40** |

Not one occurrence in 1,996,787 rows. The corpus lag-1 autocorrelation at
15 minutes is 0.99, as physics requires.

This is the cleanest possible statement of the problem: **the behaviour that
dominates the SIH GEO file does not exist in real GNSS data.** Training a model
to reproduce it would be training it to reproduce a bug.

---

# Part III — Why seven days is structurally insufficient

Everything above could, in principle, be fixed by re-issuing the files. This
section cannot be fixed. These are properties of a seven-day, single-satellite,
irregularly-sampled observation window.

## 9. The ephemeris upload — the dominant event — is unobservable

Broadcast ephemeris error is a **sawtooth**: it grows with the age of the
navigation message and resets to near zero when the ground segment uploads a new
one. Predicting day 8 requires knowing where the resets fall. The delivered
timestamps cannot tell you.

We took the **actual sample times from each SIH file**, laid them over the
**actual arc boundaries** of real satellites in our corpus, and counted how many
resets fall inside a sampling gap. A reset counts as observed only if some
sample lands within 15 minutes after it:

| sampling pattern taken from | resets in window | **not resolvable** |
|---|---|---|
| `DATA_GEO_Train.csv` | 70,395 | **48.5%** |
| `DATA_MEO_Train.csv` | 63,869 | **71.4%** |
| `DATA_MEO_Train2.csv` | 63,463 | **81.1%** |

Between half and four-fifths of the uploads leave no trace in the data. If a
reset occurs at hour 0 and another at hour 1, and only the first is sampled, the
model has no way to infer that the second happened — the error simply appears
to have jumped, indistinguishably from noise. **The most informative event in
the process is invisible more often than it is visible.**

This is a direct consequence of the "arbitrary timestamps" requirement. It is
not a nuisance in the evaluation protocol; it destroys observability of the
generating mechanism.

## 10. Age-of-ephemeris drives the error, and is not provided

The sawtooth is real and we can measure it on both datasets.

In `DATA_MEO_Train2.csv`, **every one of the 23 rows following a gap ≥ 45 min**
is large, and the row 1–7 minutes later collapses:

```
after a gap >= 45 min :  n=23   median |e| = 0.952 m
after a gap <  45 min :  n=119  median |e| = 0.180 m       ratio 5.3x
Mann-Whitney U p = 2.4e-13        spearman(gap, |e|) = 0.52

12:14  gap= 73m  |e|=1.228   ->  +7m  |e|=0.093    drop x13.2
02:46  gap= 65m  |e|=1.058   ->  +5m  |e|=0.119    drop x 8.9
23:46  gap=1435m |e|=0.983   ->  +5m  |e|=0.100    drop x 9.8
```

Our corpus reproduces exactly this, once the constant offset of §16 is removed:

| age of ephemeris | n | median &#124;e&#124; (m) | vs fresh |
|---|---|---|---|
| 0–15 min | 467,824 | 0.299 | 1.00× |
| 15–30 min | 278,102 | 0.351 | 1.18× |
| 30–60 min | 521,482 | 0.358 | 1.20× |
| 1–1.5 h | 178,401 | 0.816 | 2.73× |
| 1.5–2 h | 92,160 | 0.790 | 2.64× |
| > 2 h | 1,305 | 1.353 | **4.53×** |

**SIH MEO: 0.180 → 0.952 m = 5.3×. Corpus: 0.299 → 1.353 m = 4.5×.** Same
quantity, same mechanism, same magnitude. This confirms the SIH MEO data *is*
genuine broadcast-minus-precise error — and it confirms that **age-of-data is
the dominant driver of its envelope.**

Age is a column in our corpus (`age_s`). It is **not** in the SIH files, and per
§9 it cannot be reliably reconstructed from the timestamps. The model is being
asked to predict a sawtooth without being told where the teeth are.

*(Note: age governs the error* magnitude*, not its sign — conditioning on age
bins explains R² ≈ 0.02 of the signed component. That is precisely why the sign
must come from somewhere else, namely orbital phase, which is also absent.)*

## 11. The window's own offset is not identifiable — the ε problem

If the seven-day record sits on an unknown constant offset ε, nothing inside the
window determines ε, because the process has no fixed repeating period to anchor
against. We tested this directly: across all 7-day windows in the corpus, how
well does the mean of the window predict the mean of the day that follows?

| channel | corr(week-mean, next-day-mean) | R² |
|---|---|---|
| radial | +0.372 | 0.138 |
| along-track | +0.345 | 0.119 |
| cross-track | +0.630 | 0.397 |
| **clock** | **−0.011** | **0.000** |

**The mean level of the clock error on day 8 is statistically independent of its
mean level over the preceding seven days.** R² = 0.000. There is no amount of
modelling that recovers a quantity carrying zero mutual information with the
input. For position, at best 14–40% of the offset is transferable.

## 12. Most of the variance lives at periods a 7-day window cannot resolve

Per-satellite power spectra of the corpus error, as a share of total variance:

| system | channel | **> 7 days** | 1–7 days | 6 h – 1 day | < 6 h |
|---|---|---|---|---|---|
| GPS | along-track | **0.385** | 0.296 | 0.144 | 0.117 |
| GPS | clock | 0.008 | **0.626** | 0.261 | 0.101 |
| Galileo | clock | 0.007 | **0.596** | 0.302 | 0.091 |
| BeiDou | clock | 0.009 | **0.429** | 0.368 | 0.191 |
| BeiDou | cross-track | **0.229** | 0.069 | 0.690 | 0.017 |
| QZSS | clock | 0.007 | 0.434 | 0.503 | 0.056 |

Two readings:

- The **> 7 days** column is variance a seven-day file cannot resolve *by
  construction* — Nyquist. For GPS along-track that is **38.5%** of the total.
- The **1–7 day** column is barely better: within a seven-day window you observe
  **at most one cycle** of a seven-day component. One cycle does not identify an
  amplitude and a phase. For the clock channel this band holds **43–63%** of the
  variance across every constellation.

Combining the two, **44% to 68% of clock error variance sits at periods the
training window cannot identify.**

The longest of these is the **draconitic** signal — the beat between the orbit
plane and the Sun, at 351.4 d for GPS, ≈ 355.6 d for Galileo, ≈ 353 d for
BeiDou. Seven days is **2% of one draconitic cycle**. Solar radiation pressure
mismodelling, which is the largest single contributor to orbit prediction error,
is organised at exactly this period and its harmonics. A seven-day window cannot
see it, cannot estimate it, and cannot correct for it.

*(In fairness: 236 days is 0.67 of a draconitic year, so our corpus resolves the
harmonics but not the fundamental. The honest comparison is 67% of a cycle
versus 2% of a cycle.)*

## 13. Eclipse seasons are absent from most seven-day windows

During eclipse the satellite's solar panels are unlit, the yaw-steering attitude
law changes, and the SRP model breaks down. This is a leading source of orbit
prediction error and it is **seasonal**.

Computing shadow geometry over all 236 days (apparent solar position, cylindrical
umbra test against ECEF satellite positions):

- MEO satellites are in shadow for **1.21%** of epochs
- Eclipse-season days are **19.1%** of days (IQR 12.3–26.3%)
- On eclipse-season days, de-biased along-track error is **1.21×** larger
  (median 0.265 m vs 0.218 m); total error 1.17×; the 99th percentile rises from
  2.39 m to 2.74 m
- **P(a random 7-day window contains no eclipse at all) = 0.777**

**Roughly four out of five seven-day training files contain zero examples of the
eclipse regime.** If day 8 falls in an eclipse season and days 1–7 did not, the
model is extrapolating into a regime it has never seen, with a measurably
different error law. Nothing in the delivered data flags which case you are in.

## 14. Manoeuvres are effectively unlearnable from one week

Detected from step changes in the broadcast semi-major axis (the standard
detector), requiring consecutive complete days, |Δa| > 20 m and > 6× that
satellite's own day-to-day noise floor (median 33 m):

- **148 manoeuvres** across 117 satellites over 236 days
- 43 of 117 satellites manoeuvred at least once; median |Δa| = 1,923 m
- Rate = **2.0 manoeuvres per satellite-year** — consistent with published
  station-keeping cadences, which is the check that the detector is sound

Therefore:

- **P(a 7-day single-satellite window contains a manoeuvre) = 0.037** — about
  one week in 27
- **P(a manoeuvre occurs on day 8 given none in days 1–7) = 0.0054**

A manoeuvre invalidates the orbit model outright. The training file will contain
**zero** examples of one 96% of the time, yet the test day carries a non-zero
chance of one. This is an unlearnable event by construction: the input window is
too short to contain a single instance of the phenomenon it must anticipate.

## 15. No polynomial clock model fitted on seven days beats predicting zero

The broadcast clock correction is a quadratic (a₀ + a₁t + a₂t²). The residual
error after that quadratic — which is what the dataset reports — is driven by
oscillator ageing, temperature, relativistic effects and unmodelled frequency
steps. It is not itself polynomial.

We fitted polynomials of degree 0–3 to 672 samples (7 days) of corpus clock error
and extrapolated 96 samples (1 day). RMS in metres:

| system | predict **zero** | constant | linear | quadratic | cubic |
|---|---|---|---|---|---|
| BeiDou | **0.249** | 0.250 | 0.254 | 0.265 | 0.297 |
| Galileo | **0.140** | 0.141 | 0.144 | 0.151 | 0.175 |
| GPS | **0.123** | 0.123 | 0.130 | 0.137 | 0.162 |

**Every fitted polynomial is worse than predicting zero, and the degradation is
monotone in degree.** The seven-day window contains enough noise, and too few
cycles of the real low-frequency content (§12), that fitting a trend and
extrapolating one day actively injects error. This is the quantitative form of
"the clock drift is not mappable by a quadratic".

The SIH clock column is worse still. In `MEO_Train2` it takes only **70 distinct
values across 143 rows**, one value repeating 17 times, while the position
columns are 143/143 unique. The clock channel is **piecewise constant**, jumping
at message boundaries — a step process, not a drift. Its lag-1 autocorrelation at
1 hour is **0.016**, against **0.843** in our corpus. There is no trend in it to
learn.

---

## 16. One thing that is *not* a problem: antenna phase centre vs centre of mass

For completeness, because it is the objection most likely to be raised.

Broadcast ephemeris refers to the satellite's **antenna phase centre**; precise
SP3 products refer to the **centre of mass**. The difference appears as an offset
in the computed error. We measured it:

- Per-satellite median radial error: **−1.10 m** (IQR −1.29 to −0.83 m)
- Along-track: −0.02 m; cross-track: −0.00 m — as expected, the offset is almost
  purely radial
- Spread of this offset across satellites: σ = 0.368 m — i.e. it is **stable per
  satellite**
- It accounts for **68%** of the raw error magnitude: median |e| falls from
  1.163 m to 0.367 m once removed

Because it is a fixed per-satellite vector, it cancels identically in any
differenced or de-biased treatment and contributes nothing to the *time-varying*
signal we are asked to predict. **We treat it as a known constant, not as an
error source.** It is listed here so that it is not mistaken for one of the
defects above — and because failing to remove it inflates every magnitude
statistic by a factor of three, which is a trap in its own right.

---

# The alternative, and why it is sound

We built a 236-day corpus from primary sources: NASA CDDIS `BRDM00DLR`
multi-GNSS broadcast navigation messages, differenced against `JAX0MGXFIN` final
precise orbits — the only final MGEX product that carries GEO satellites.
1,996,787 epochs, 100 satellites, GPS + Galileo + BeiDou + QZSS, on a uniform
900 s grid, 2026-01-01 to 2026-08-24.

Against the same tests:

| property | SIH dataset | 236-day corpus |
|---|---|---|
| Epochs | 6 – 143 usable | **1,996,787** |
| Satellites | 1 per file, unidentified | **100, all labelled by PRN and constellation** |
| Time span | < 7 days, 13–32% covered | **236 days, uniform 900 s grid** |
| Lag-1 autocorrelation (15 min) | **−0.85** (GEO) | **+0.99** |
| Unphysical antiparallel pairs | 17–40% | **0.0000** |
| Age-of-ephemeris | absent, 48–81% unrecoverable | **`age_s`, exact, per epoch** |
| Orbital phase | absent | **`u_arg`, exact, per epoch** |
| Upload/arc boundaries | unlabelled | **`arc_id`, `arc_toe`, exact** |
| Eclipse regime | absent from 78% of windows | **≈ 3,900 satellite-days of eclipse** |
| Manoeuvres | 0 in 96% of windows | **148 labelled events** |
| Draconitic cycle | 2% of one period | **67% of one period** |
| Residual outliers | 17–40% of large GEO samples | **0.0012% (clock), 0.0050% (position)** |

The corpus is not flawless — it carries the −1.10 m APC/CoM offset (known,
removable), 23 rows with |clock| > 10 m, and 99 rows with |position| > 50 m that
need an outlier pass. That is **0.005%** of the data, against an artefact
affecting **17–40%** of the large-magnitude samples in the SIH GEO file.

Critically, the corpus is the **same physical quantity**: SIH MEO error grows
5.3× from fresh to stale ephemeris, the corpus 4.5×, at the same absolute scale
(0.18 → 0.95 m versus 0.30 → 1.35 m). We are not substituting a proxy. We are
substituting a correctly-sampled, fully-labelled, 14,000× larger measurement of
the identical process.

---

## Recommendation

1. **Do not train on the SIH GEO files.** They contain an artefact that does not
   exist in real GNSS data. Anything learned from them is learned from a bug.
2. **The SIH MEO files are genuine but too small to train on** — 1 to 5 effective
   independent samples. They remain useful as a *sanity check* that our
   corpus-trained model produces errors of the right magnitude on the right
   satellite class.
3. **Train on the 236-day corpus**, using the labelled drivers the SIH format
   discards: age-of-ephemeris, orbital phase, arc boundaries.
4. **State the structural limits explicitly** rather than hiding them: the day-8
   offset is unidentifiable for the clock (R² = 0.000), eclipse and manoeuvre
   regimes are absent from most seven-day windows, and 44–68% of clock variance
   sits at periods the window cannot resolve. Any model — ours or anyone
   else's — is subject to these bounds. A submission that claims otherwise is
   reporting overfit, not skill.

---

## Appendix — reproducing these numbers

All analyses run against `Data_PS-08/*.csv` and `full.parquet`.

| § | What it computes | Method |
|---|---|---|
| 1 | Duplicate rows | `df.drop_duplicates()` row-count delta |
| 2 | Coverage | union of ±15 min windows / 168 h |
| 3 | `n_eff` | `n(1−ρ)/(1+ρ)`, ρ = lag-1 at native cadence |
| 4, 5 | Distribution shift | Levene, KS two-sample, variance ratio |
| 7 | Antiparallel test | cosine of consecutive 4-vectors, both norms > 3 m |
| 8 | Corpus control | identical test, per satellite, exact 900 s pairs |
| 9 | Reset observability | real SIH timestamps overlaid on corpus `arc_toe` changes |
| 10 | Sawtooth | error magnitude binned by `age_s`, offset removed |
| 11 | Offset identifiability | corr(mean of 672-sample window, mean of next 96) |
| 12 | Spectra | Hann-windowed rFFT per satellite, banded by period |
| 13 | Eclipse | apparent solar position → ECEF, cylindrical umbra test |
| 14 | Manoeuvres | steps in daily-median broadcast `sqrt_a²` |
| 15 | Clock polynomials | `polyfit` degree 0–3 on 7 d, extrapolate 1 d |
| 16 | APC/CoM | per-satellite median RTN error vector |
