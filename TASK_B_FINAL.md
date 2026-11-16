# Task B — the final model, and the benchmark of everything we tried

**The deliverable.** Companion to `JOURNEY.md` (the full exploration log),
`TASK_A_AND_WHY_WE_LEFT_IT.md` (the operational detour), and
`WHY_SIH_DATA_FAILS.md` (why the delivered PS-08 files cannot support the task).

Written 2026-09-04. Every number below came off a run on this machine or the
rented GPU box. Nothing is quoted from a single fold.

---

## 0. The one-paragraph version

The competition hands you **seven days of error history** — timestamp, x, y, z,
clock — plus **which orbit class** the satellite is, and asks for the same four
numbers at arbitrary day-8 timestamps. We tested **30 methods**: 23 in a
head-to-head benchmark over **24 blocked cross-validation folds** (trivial
baselines, seasonal naive, spectral extrapolation, ARIMA/SARIMA, Gaussian
processes, singular spectrum analysis, six tabular learners), plus **five neural
architectures** including a transformer, plus two correction layers on top of the
winner.

**The winner is a physics-derived deterministic predictor with zero trained
parameters.** It scores **GEO 0.244 m** and **MEO 0.397 m** mean RMS against
2.806 / 2.518 for doing nothing.

**On MEO it swept: 0 of 22 rival methods won a single one of 24 folds**, and it
beat the runner-up by 21%. Every neural network that tried to *replace* it lost
by 56–84%; every learned layer that tried to *correct* it tied or made it worse.

**Two things did beat it, and both are in here rather than buried.** A GRU on the
residual won 5/5 MEO folds — by 0.4% (§4.2). And on **GEO**, plain fixed harmonic
regression with *no model selection at all* scored 0.372 against our 0.382,
winning 15/24 folds (§3.2). We chased the second one down, found the mechanism,
implemented the fix — and then did not ship it, because the gain failed to
replicate on an independent sample and cost tail robustness (§3.3).

The deliverable is one file, `smart_horizon.py`, with no weights, no corpus and
no network calls.

---

## 1. What the model actually is

### 1.1 The signature

```python
from smart_horizon import predict

out = predict(train_df, query_times, orbit="MEO")
```

| | |
|---|---|
| `train_df` | `utc_time, x_error, y_error, z_error, satclockerror` — 7 days, any sampling, gaps fine, duplicates fine, unsorted fine |
| `query_times` | day-8 timestamps, any spacing, arbitrary |
| `orbit` | `"MEO"` or `"GEO"` — the competition tells us this |
| returns | the four predictions **plus a 1-σ uncertainty per column** |

That is exactly the five columns they supply. Nothing else is required at
inference — no satellite ID, no broadcast message, no ephemeris, no internet.

### 1.1b Running it

```python
import pandas as pd
from smart_horizon import predict

hist = pd.read_csv("seven_days.csv")          # utc_time,x_error,y_error,z_error,satclockerror
want = pd.date_range("2026-01-08", periods=96, freq="900s")   # or ANY timestamps
out  = predict(hist, want, orbit="MEO")
out.to_csv("day8.csv", index=False)
```

or straight from files:

```python
from smart_horizon import predict_csv
predict_csv("seven_days.csv", "query_times.csv", orbit="GEO", out_csv="day8.csv")
```

`query_times` is unconstrained — irregular spacing, non-grid epochs, a single
timestamp, or 10,000 of them. The model is a continuous function of time, not a
step-ahead forecaster, so there is no horizon to configure and nothing to unroll.

Dependencies are numpy and pandas. Runtime is ~65 ms per window on one core.

### 1.2 Two models, because the physics is different and the class is given

This was the single largest win of the project, and it came from asking why one
model was being fitted to two different objects.

**GEO / GSO — the period is known a priori.**
A geosynchronous satellite's ground track repeats at the **sidereal day**,
86164 s, and a 7-day window therefore contains **7 clean cycles**. There is
nothing to search for. Only the harmonic order K ∈ {1,2,3,4} is chosen.

**MEO — the period is unknown and the axes do not share it.**
The orbital period differs per constellation (GPS 43082 s, Galileo 50567 s,
BeiDou 46393 s) and *we are not told which constellation it is*. Worse, in the
ECEF frame the three axes do not oscillate at the same frequency:

* **z** sits at the orbital period **T**
* **x and y** are beaten against Earth rotation at **1/(1/T ± 1/T_sid)**

So T is estimated per window by Lomb-Scargle over the physically possible MEO
band (38000–54000 s), and the two beat periods are then derived *analytically*
rather than searched for. This is why a single model over both classes was
leaving performance on the table: it was trying to fit a sidereal basis to a
signal that lives at a beat frequency.

**Clock — level estimators only, both classes, always.**
The day-8 clock mean is statistically independent of the 7-day window mean
(R² = 0.000). Harmonics on the clock made it worse every one of the six ways we
tried it. So the clock gets a level estimator and nothing else. This is a
*restriction* derived from the corpus, and it is worth roughly 93–95% error
reduction on the clock column.

### 1.3 Self-validation — what it is, and where it earns its keep

For each component the predictor builds a candidate set (level estimators, plus
harmonic bases at the appropriate periods), then:

1. fits every candidate on **days 1–6** of the supplied window,
2. scores each on **day 7**, held out,
3. refits the winner on **all 7 days**,
4. reports the held-out RMS as an honest **1-σ**.

The held-out day is the only model-selection signal used, and it comes from the
caller's own data. That is what makes the model adaptive without being trained:
it picks a different basis for a quiet satellite than for a noisy one, per
window, at call time.

§3.4 measures what this step is worth by benchmarking the same harmonic
regression *without* it, and the answer splits by class: **worth +44.8% on MEO,
worth −2.6% on GEO.** Selection pays only where the period must be estimated.
On GEO, where it is known exactly, it is a slight liability that we keep anyway
for tail robustness — the full argument is §3.3.

### 1.4 The level estimators

`mean_all`, `mean_3d`, `mean_2d`, `mean_1d`, `mean_12h`, `ewma_1d`, `ewma_2d` —
seven of them, competing on the held-out day. On the clock column, one of these
always wins.

---

## 2. Headline performance

Held-out windows, irregularly sampled (~30% retention plus multi-hour gaps —
their format), RMS in metres.

### GEO / GSO — 50 test windows, 1,838 query points

| method | x_err | y_err | z_err | clock_err | **mean** |
|---|---|---|---|---|---|
| do nothing | 2.178 | 1.682 | 0.941 | 6.425 | **2.806** |
| window mean | 0.531 | 0.455 | 0.873 | 0.332 | **0.548** |
| **SMART-HORIZON** | **0.326** | **0.238** | **0.144** | **0.268** | **0.244** |
| reduction | 85.9% | 85.8% | 84.2% | 95.0% | **90.9%** |

### MEO — 650 test windows, 21,158 query points

| method | x_err | y_err | z_err | clock_err | **mean** |
|---|---|---|---|---|---|
| do nothing | 0.807 | 0.821 | 0.815 | 7.628 | **2.518** |
| window mean | 0.808 | 0.824 | 0.812 | 0.468 | **0.728** |
| **SMART-HORIZON** | **0.434** | **0.445** | **0.381** | **0.328** | **0.397** |
| reduction | 46.4% | 45.9% | 53.4% | 95.5% | **84.1%** |

**Read the MEO row honestly.** The clock column is where the huge reduction is;
x/y/z improve by 46–53%, not 85%. The GEO numbers are the genuinely strong ones,
because there the period is known exactly.

### Stability across independent time blocks

Six non-overlapping blocks, 220 windows each, no refitting between them (there
is nothing to refit):

| block origin | GEO | MEO |
|---|---|---|
| 16396 | 0.317 | 0.412 |
| 17502 | 0.266 | 0.404 |
| 18608 | 0.417 | 0.366 |
| 19715 | 0.256 | 0.413 |
| 20821 | 0.381 | 0.378 |
| 21928 | 0.301 | 0.340 |
| **mean ± sd** | **0.323 ± 0.058** | **0.385 ± 0.027** |
| range | [0.256, 0.417] | [0.340, 0.413] |

**MEO varies by 7% across the corpus.** That is the direct consequence of having
no trained parameters — the model cannot be lucky or unlucky about a training
period, because it never saw one. Compare Task A's trained model at
0.705 ± 0.433, a 5× spread.

### The shipped file, checked through its public API

`smart_horizon.py` is a standalone re-implementation that deliberately imports
nothing from the corpus-side code, so it was re-verified from the outside: build
a DataFrame of the competition's five columns, call `predict()`, score what comes
back (`verify_deliverable.py`). 150 windows per condition.

| query times | | x | y | z | clock | **mean** | do-nothing |
|---|---|---|---|---|---|---|---|
| on-grid 15-min | GEO | 0.274 | 0.227 | 0.202 | 0.416 | **0.280** | 2.510 (88.9% better) |
| on-grid 15-min | MEO | 0.401 | 0.398 | 0.349 | 0.411 | **0.390** | 2.514 (84.5% better) |
| **arbitrary off-grid** | GEO | 0.497 | 0.309 | 0.144 | 0.300 | **0.312** | 3.025 (89.7% better) |
| **arbitrary off-grid** | MEO | 0.460 | 0.471 | 0.408 | 0.530 | **0.467** | 2.790 (83.3% better) |

The off-grid rows are the important ones: query epochs jittered ±400 s off the
15-minute grid, with truth interpolated to them, so the model is asked for times
it has never been handed. **It degrades by ~10% on GEO and ~20% on MEO and keeps
working.** Part of that gap is the interpolated truth itself, not the model.

### Uncertainty calibration

The predictor returns a 1-σ per column, taken from the held-out day of the
self-validation step. Checked against actual day-8 error on 700 corpus windows,
that raw σ is **optimistic by 1.15× (median) to 1.30× (variance-pooled)** — the
held-out day both *selects* the winning basis and *scores* it, so it cannot be an
unbiased forecast error.

`smart_horizon.py` therefore carries **one empirical constant**, `SIGMA_CAL =
1.25`, which scales the reported uncertainty and nothing else. Predictions are
untouched. After it:

| | x | y | z | clock |
|---|---|---|---|---|
| GEO, actual/reported | 1.11 | 0.90 | 0.95 | 0.99 |
| MEO, actual/reported | 1.11 | 1.14 | 1.28 | **1.46** |

GEO is well calibrated. **The MEO clock still under-covers by ~1.5×** and we are
leaving it there rather than tuning a second constant: the shortfall comes from a
few windows with clock regime jumps (§7), so inflating σ everywhere would make it
wrong for the other 95%.

### Sampling density

| | GEO | MEO |
|---|---|---|
| irregular ~30% + gaps (their format) | 0.387 | 0.460 |
| uniform 15-min (upper bound) | 0.394 | 0.416 |

GEO is **insensitive** to sampling density; MEO degrades ~10% under irregular
sampling, because the Lomb-Scargle period estimate gets harder. Both are usable.

---

## 3. The benchmark — everything we tested

**Protocol, identical for every method:** 24 random contiguous 2-week test
blocks, training data (where a method uses any) strictly before the block with an
8-day gap. GEO and MEO pools built **separately**, because GEO/GSO is only 15 of
103 spacecraft and a proportionally sampled pool leaves ~12 GEO windows per fold
— too few to score. Metric is RMS in metres, averaged over x/y/z/clock.

**23 methods, 24 folds each.** GEO pool 2,400 windows (135 test per fold), MEO
pool 3,000 windows (170 test per fold). Run on a 128-core box; window-local
methods scored once per window in parallel, tabular learners refitted per fold.

### 3.1 MEO — 24 blocked folds

| method | mean | sd | best | worst | x | y | z | clock | vs ours | wins |
|---|---|---|---|---|---|---|---|---|---|---|
| **ours** | **0.458** | 0.038 | 0.409 | 0.517 | 0.472 | 0.475 | 0.398 | 0.488 | — | — |
| ar-yw | 0.556 | 0.060 | 0.470 | 0.637 | 0.567 | 0.569 | 0.547 | 0.542 | +21.4% | 0/24 |
| sarima | 0.582 | 0.086 | 0.477 | 0.694 | 0.595 | 0.586 | 0.535 | 0.610 | +26.8% | 0/24 |
| fft | 0.608 | 0.079 | 0.502 | 0.729 | 0.560 | 0.565 | 0.552 | 0.756 | +32.7% | 0/24 |
| harm-fixed | 0.664 | 0.057 | 0.583 | 0.742 | 0.835 | 0.846 | 0.395 | 0.578 | +44.8% | 0/24 |
| arima | 0.664 | 0.088 | 0.548 | 0.790 | 0.722 | 0.715 | 0.620 | 0.599 | +44.8% | 0/24 |
| snaive-avg | 0.666 | 0.058 | 0.585 | 0.745 | 0.829 | 0.839 | 0.423 | 0.572 | +45.2% | 0/24 |
| extratrees | 0.684 | 0.056 | 0.604 | 0.763 | 0.741 | 0.741 | 0.693 | 0.559 | +49.1% | 0/24 |
| svr | 0.698 | 0.054 | 0.609 | 0.784 | 0.750 | 0.743 | 0.731 | 0.567 | +52.2% | 0/24 |
| gbm | 0.706 | 0.057 | 0.635 | 0.789 | 0.773 | 0.765 | 0.722 | 0.563 | +53.9% | 0/24 |
| rf | 0.736 | 0.068 | 0.639 | 0.828 | 0.768 | 0.781 | 0.722 | 0.671 | +60.4% | 0/24 |
| knn | 0.740 | 0.070 | 0.637 | 0.842 | 0.764 | 0.762 | 0.715 | 0.718 | +61.3% | 0/24 |
| ridge | 0.742 | 0.070 | 0.664 | 0.845 | 0.823 | 0.831 | 0.791 | 0.522 | +61.7% | 0/24 |
| mean | 0.756 | 0.059 | 0.678 | 0.840 | 0.827 | 0.835 | 0.790 | 0.573 | +65.0% | 0/24 |
| median | 0.765 | 0.059 | 0.686 | 0.846 | 0.841 | 0.848 | 0.798 | 0.573 | +66.9% | 0/24 |
| drift | 0.790 | 0.063 | 0.703 | 0.889 | 0.867 | 0.874 | 0.806 | 0.612 | +72.2% | 0/24 |
| snaive | 0.894 | 0.102 | 0.782 | 1.041 | 1.183 | 1.190 | 0.543 | 0.661 | +95.0% | 0/24 |
| holt-winters | 0.919 | 0.090 | 0.815 | 1.050 | 1.208 | 1.181 | 0.693 | 0.596 | +100.5% | 0/24 |
| gp | 0.970 | 0.112 | 0.830 | 1.135 | 1.236 | 1.248 | 0.585 | 0.812 | +111.6% | 0/24 |
| prony | 1.010 | 0.255 | 0.743 | 1.344 | 0.936 | 1.020 | 0.674 | 1.412 | +120.4% | 0/24 |
| last | 1.035 | 0.088 | 0.938 | 1.171 | 1.212 | 1.187 | 1.141 | 0.601 | +125.8% | 0/24 |
| ssa | 1.080 | 0.361 | 0.716 | 1.592 | 0.949 | 0.904 | 0.863 | 1.606 | +135.7% | 0/24 |
| zero | 2.076 | 0.168 | 1.755 | 2.475 | 0.824 | 0.830 | 0.796 | 5.856 | +352.9% | 0/24 |

**A clean sweep. Nothing beat it on a single one of 24 folds**, and the margin
over the runner-up is 21%. Per-fold: 0.423 0.425 0.425 0.430 0.411 0.409 0.427
0.438 0.453 0.451 0.465 0.517 0.511 0.512 0.506 0.504 0.515 0.492 0.493 0.479
0.448 0.431 0.420 0.419.

Three things worth reading off this table:

* **`harm-fixed` is +44.8% on MEO** — the same harmonic basis without
  self-validation. On MEO, selection is worth nearly half the error. Note its z
  column (0.395) is actually *better* than ours (0.398); the damage is all in x
  and y (0.835/0.846 vs 0.472/0.475), which is exactly the beat-frequency
  structure of §1.2. A fixed sidereal basis cannot represent it.
* **`gp`, the textbook answer for irregular quasi-periodic data, is +111.6%.**
  A periodic×RBF kernel forces one shared period on all three axes, which is
  wrong for MEO in ECEF for the same reason.
* **`snaive` (+95%) is much worse than `snaive-avg` (+45%)** — one cycle back is
  noisy; averaging every available cycle is a real estimator. Both still lose.

### 3.2 GEO — 24 blocked folds

| method | mean | sd | best | worst | x | y | z | clock | vs ours | wins |
|---|---|---|---|---|---|---|---|---|---|---|
| harm-fixed | **0.372** | 0.075 | 0.271 | 0.499 | 0.636 | 0.294 | 0.189 | 0.368 | **−2.6%** | **15/24** |
| **ours** | 0.382 | 0.056 | 0.293 | 0.477 | 0.643 | 0.295 | 0.198 | 0.392 | — | — |
| snaive | 0.429 | 0.072 | 0.319 | 0.541 | 0.631 | 0.346 | 0.385 | 0.354 | +12.3% | 2/24 |
| sarima | 0.440 | 0.069 | 0.341 | 0.552 | 0.500 | 0.348 | 0.566 | 0.345 | +15.2% | 1/24 |
| ar-yw | 0.446 | 0.067 | 0.352 | 0.550 | 0.527 | 0.356 | 0.532 | 0.367 | +16.7% | 0/24 |
| snaive-avg | 0.447 | 0.142 | 0.285 | 0.667 | 0.626 | 0.287 | 0.505 | 0.371 | +17.1% | 11/24 |
| holt-winters | 0.466 | 0.063 | 0.361 | 0.585 | 0.522 | 0.404 | 0.537 | 0.402 | +22.2% | 0/24 |
| arima | 0.513 | 0.055 | 0.431 | 0.602 | 0.552 | 0.457 | 0.641 | 0.400 | +34.3% | 0/24 |
| gp | 0.538 | 0.063 | 0.440 | 0.630 | 0.975 | 0.405 | 0.353 | 0.419 | +41.0% | 0/24 |
| ridge | 0.570 | 0.027 | 0.517 | 0.606 | 0.602 | 0.475 | 0.802 | 0.401 | +49.3% | 0/24 |
| mean | 0.591 | 0.046 | 0.522 | 0.665 | 0.680 | 0.472 | 0.797 | 0.415 | +54.8% | 0/24 |
| median | 0.607 | 0.052 | 0.531 | 0.689 | 0.722 | 0.478 | 0.817 | 0.412 | +59.1% | 0/24 |
| drift | 0.619 | 0.047 | 0.539 | 0.693 | 0.719 | 0.492 | 0.831 | 0.432 | +62.0% | 0/24 |
| gbm | 0.642 | 0.104 | 0.519 | 0.877 | 0.908 | 0.488 | 0.751 | 0.423 | +68.3% | 0/24 |
| extratrees | 0.645 | 0.102 | 0.516 | 0.882 | 0.961 | 0.486 | 0.709 | 0.422 | +68.9% | 0/24 |
| fft | 0.654 | 0.213 | 0.431 | 0.971 | 0.871 | 0.382 | 0.871 | 0.493 | +71.3% | 0/24 |
| rf | 0.664 | 0.103 | 0.534 | 0.894 | 0.946 | 0.508 | 0.735 | 0.469 | +74.0% | 0/24 |
| ssa | 0.682 | 0.100 | 0.485 | 0.856 | 0.683 | 0.487 | 0.986 | 0.573 | +78.7% | 0/24 |
| last | 0.715 | 0.036 | 0.661 | 0.770 | 0.602 | 0.633 | 1.130 | 0.495 | +87.3% | 0/24 |
| knn | 0.721 | 0.092 | 0.556 | 0.935 | 1.048 | 0.544 | 0.789 | 0.504 | +89.0% | 0/24 |
| svr | 0.730 | 0.152 | 0.511 | 1.073 | 1.254 | 0.505 | 0.719 | 0.441 | +91.2% | 0/24 |
| prony | 0.863 | 0.177 | 0.555 | 1.123 | 0.968 | 0.533 | 1.263 | 0.688 | +126.1% | 0/24 |
| zero | 2.417 | 0.472 | 1.378 | 2.952 | 1.984 | 1.823 | 0.844 | 5.018 | +533.2% | 0/24 |

**We lost on GEO.** `harm-fixed` — the same harmonic regression with *no model
selection at all* — scored 0.372 against our 0.382 and won 15 of 24 folds. §3.3
is what we did about it.

Everything else still loses by 12% or more, and the shape of the losses is
informative: on GEO, `sarima`/`ar-yw`/`holt-winters` beat ours on the **x**
column (0.500–0.527 vs 0.643) while being far worse on **z** (0.53–0.57 vs
0.198). GEO x is the hardest column in the problem and a pure time-series model
handles it better; we do not currently exploit that.

### 3.3 The GEO result, chased down

A benchmark that says the simpler method wins is worth more than one that
confirms you. So this got its own experiment (`geo_fix.py`), same pool, same 24
folds:

| variant | mean | sd | worst | x | y | z | vs ours | wins |
|---|---|---|---|---|---|---|---|---|
| fixed-K3 | **0.369** | 0.072 | 0.489 | 0.633 | 0.259 | 0.194 | −3.3% | 17/24 |
| fixed-K4 | 0.373 | 0.075 | 0.498 | 0.635 | 0.258 | 0.206 | −2.4% | 14/24 |
| noLvl-K34 | 0.376 | 0.072 | 0.498 | 0.633 | 0.274 | 0.203 | −1.6% | 14/24 |
| noLvl-K234 | 0.376 | 0.072 | 0.499 | 0.633 | 0.276 | 0.203 | −1.5% | 14/24 |
| ours-noLvl | 0.377 | 0.072 | 0.499 | 0.633 | 0.278 | 0.204 | −1.3% | 13/24 |
| fixed-K2 | 0.378 | 0.071 | 0.497 | 0.636 | 0.294 | 0.189 | −1.0% | 14/24 |
| **ours** | 0.382 | **0.056** | **0.477** | 0.643 | 0.295 | 0.198 | — | — |
| fixed-K2+lvl | 0.392 | 0.058 | 0.490 | 0.649 | 0.325 | 0.201 | +2.6% | 0/24 |
| fixed-K1 | 0.425 | 0.067 | 0.539 | 0.675 | 0.427 | 0.206 | +11.3% | 0/24 |

**The mechanism is identified.** `fixed-K2` (0.378) vs `fixed-K2+lvl` (0.392) is
a controlled comparison — identical basis, the only difference being whether
level estimators may win the held-out day. Allowing them costs **3.7%**. On GEO
the sidereal period is known exactly, so there is nothing useful to select, and
a noisy held-out day occasionally picks a flat line where a harmonic would have
extrapolated correctly. K itself barely matters once K ≥ 2 (0.369 / 0.373 /
0.378, all inside the fold sd of 0.072).

**And we did not ship it.** Barring the level estimators was implemented, then
tested against an independent sample through the deliverable's public API — and
the gain vanished:

| GEO, independent sample | with levels | without levels |
|---|---|---|
| on-grid mean | 0.280 | 0.277 |
| **off-grid mean** | **0.312** | **0.374** |
| off-grid x | 0.497 | **0.755** |

The 1.5–3% fold-mean gain did not replicate, and on off-grid query times it
reversed into a 20% loss concentrated entirely in x. Clamping the harmonic to
the window's dynamic range did not recover it — so this is not a divergence
artefact. On those windows the harmonic simply has the wrong shape and the flat
estimator is the better answer; the level estimators are earning their place as
a safety net, not merely as a competitor.

Consistent with this, `ours` has the **lowest fold variance of any GEO variant**
(sd 0.056 vs 0.071–0.075) and the **best worst-case fold** (0.477 vs 0.489+).

**Decision: keep the level estimators.** An unreplicated 3% mean gain that costs
tail robustness is not worth shipping. This is recorded rather than buried,
because it is the one place in the project where a simpler method beat ours on
the primary benchmark, and because a reviewer running `harm-fixed` on GEO will
find the same thing.

The divergence clamp was kept — it changes none of the numbers above, and it is
cheap insurance for any window unlike those in the corpus.

### 3.4 What self-validation is worth, by class

| | ours | harm-fixed | selection is worth |
|---|---|---|---|
| MEO | 0.458 | 0.664 | **+44.8%** |
| GEO | 0.382 | 0.372 | **−2.6%** |

The value of self-validation is **entirely conditional on the period being
unknown**. Where it must be estimated (MEO), it is worth nearly half the error.
Where it is known a priori (GEO), it is a slight liability. That is a coherent
result rather than a contradiction — and it is the strongest single argument for
the two classes being separate models.

### 3.5 Why these numbers differ from §2

§2 reports GEO 0.244 / MEO 0.397; this section reports 0.382 / 0.458 for the same
model. The difference is the **window pool**, not the model: §3 samples GEO and
MEO pools separately and uniformly over all spacecraft of each class, so hard
satellites carry the same weight as easy ones. §2 samples proportionally from the
whole constellation. Both are stated; the §3 protocol is the harsher one, and
every method in §3 is scored on the identical windows, which is what makes the
comparison valid.

---

## 4. Neural networks — the full result

Run on a rented RTX PRO 4500 Blackwell. Pool of 12,000 windows, 5 rolling-origin
folds, model retrained from scratch at each fold, every fold reported.

Five architectures:

| | |
|---|---|
| `mlp` | window flattened, no temporal structure — the control that separates architecture from information |
| `gru` | recurrent, 2 layers |
| `lstm` | recurrent, 2 layers |
| `tcn` | dilated causal convolutions, dilations 1→128 |
| `xfmr` | transformer with continuous time encoding — the only architecture that takes irregular sampling natively rather than through a resampled grid |

Each was run **twice**: predicting the value directly, and predicting the
deterministic model's **residual** (the fair test — it only has to find what the
physics missed).

### 4.1 Predicting the value directly

| model | GEO | vs ours | wins | MEO | vs ours | wins |
|---|---|---|---|---|---|---|
| mlp | 0.629 ± 0.058 | +56.3% | 0/5 | 0.674 ± 0.024 | +65.8% | 0/5 |
| gru | 0.633 ± 0.058 | +57.3% | 0/5 | 0.670 ± 0.019 | +64.8% | 0/5 |
| lstm | 0.647 ± 0.057 | +60.8% | 0/5 | 0.701 ± 0.019 | +72.4% | 0/5 |
| tcn | 0.634 ± 0.061 | +57.7% | 0/5 | 0.697 ± 0.024 | +71.4% | 0/5 |
| **xfmr** | 0.643 ± 0.054 | +59.9% | 0/5 | 0.748 ± 0.023 | **+83.9%** | 0/5 |
| *ours* | *0.402 ± 0.102* | — | — | *0.407 ± 0.011* | — | — |

**Every architecture is 56–84% worse, and 0/5 folds won in every single case.**

Note that the MLP — which sees the window as an unordered bag of numbers — ties
the GRU and beats the LSTM and the transformer. When architectures that model
sequence structure cannot beat one that ignores it, the sequence structure is not
where the information is.

### 4.2 Predicting the residual of the deterministic model

| model | GEO | vs ours | wins | MEO | vs ours | wins |
|---|---|---|---|---|---|---|
| mlp | 0.404 ± 0.104 | +0.4% | 1/5 | 0.412 ± 0.012 | +1.3% | 0/5 |
| gru | 0.402 ± 0.102 | −0.1% | 2/5 | **0.405 ± 0.011** | **−0.4%** | **5/5** |
| tcn | 0.406 ± 0.102 | +0.8% | 0/5 | 0.411 ± 0.012 | +1.1% | 0/5 |
| xfmr | 0.402 ± 0.102 | +0.0% | 1/5 | 0.407 ± 0.011 | +0.0% | 2/5 |

**All tie.** The GRU wins 5/5 folds on MEO — consistently — but by **0.4%**. That
is a real effect and a worthless one: it would be the difference between 0.397 m
and 0.395 m, at the cost of shipping a trained network.

**Interpretation:** there is no residual structure left for a network to find.
Five architectures, including one built specifically for irregular sampling,
converge to *exactly* the deterministic model's error.

---

## 5. Does learning add anything at all? — the direct test

Before the neural work, the same question was asked of gradient boosting, with a
richer feature set: 15 window-summary features (mean, sd, recent sd, drift,
roughness, MAD, skew, kurtosis, coverage, estimated period, …) **plus the
deterministic model's own output**, trained to predict that model's error.

Trained on the first 60% of the timeline, tested on the last 30%, gap between.

### GEO/GSO — 74 train / 76 test windows

| target | deterministic | + learned | change | verdict |
|---|---|---|---|---|
| x_err | 0.426 | 0.487 | **+14.3%** | HURTS |
| y_err | 0.293 | 0.320 | +9.0% | HURTS |
| z_err | 0.204 | 0.219 | +7.4% | HURTS |
| clock_err | 0.286 | 0.318 | +11.2% | HURTS |

### MEO — 826 train / 824 test windows

| target | deterministic | + learned | change | verdict |
|---|---|---|---|---|
| x_err | 0.462 | 0.458 | −0.9% | no gain |
| y_err | 0.474 | 0.475 | +0.2% | no gain |
| z_err | 0.396 | 0.399 | +0.9% | no gain |
| clock_err | 0.327 | 0.339 | +3.5% | HURTS |

**Gradient boosting made GEO 7–14% worse on every single target** and did nothing
for MEO. Given the window's own statistics *and* the physics model's prediction,
it could not find the error.

### And horizon shrinkage

The first learned layer tried — shrink the prediction toward the window mean as
the horizon grows, with the shrinkage coefficient fitted on the corpus:

| | model | model + shrink |
|---|---|---|
| GEO | **0.244** | 0.254 |
| MEO | **0.397** | 0.401 |

Also worse. The self-validating basis choice is already calibrated, so an
external shrinkage prior can only fight it.

---

## 6. So why does the corpus exist?

This is the fair question, and it deserves a straight answer: **the corpus is
what told us which model to build. It is not what the model contains.**

244 days × 103 satellites determined:

| finding | how it constrains the model |
|---|---|
| GEO error repeats at the **sidereal day** | fixes the GEO basis, no search needed |
| true GEO (J07/J08) is **slow drift, not periodic** — R² of x_error against sidereal harmonics is 0.004 / 0.010 | level estimators must stay in the GEO candidate set |
| MEO **z** at orbital period T, **x/y** at Earth-rotation beats 1/(1/T ± 1/T_sid) | the entire MEO basis construction |
| MEO period differs per constellation and is **not supplied** | forces per-window Lomb-Scargle rather than a constant |
| clock has **no 24 h memory** (R² = 0.000) | harmonics forbidden on the clock — worth 93–95% |
| more basis functions **fit noise that does not extrapolate** (harm-MULTI: 0.514 vs 0.465) | K capped, self-validation mandatory |
| fixed periods **hurt** MEO z (0.401 → 0.546) | half-sidereal is right for GPS, wrong for Galileo/BeiDou |
| residuals are **Gaussian per satellite**, scale-mixed when pooled | σ reported per window, never pooled |

Every one of those is a design decision extracted from 14,000× more data than the
competition supplies. What the corpus did **not** produce is weights — because
three independent attempts to add them (XGBoost, shrinkage, five neural
architectures) all failed to beat the model they were correcting.

**Shipping weights would only add a way to be wrong on satellites and epochs the
training set never saw.** A model with no trained parameters cannot be
distribution-shifted. On a competition evaluation set drawn from an unknown
period, possibly an unknown constellation, that is not a limitation — it is the
main safety property of the design.

---

## 7. Residual quality

Per-window Shapiro-Wilk, from the run in §2:

| target | std | skew | kurtosis | per-window W |
|---|---|---|---|---|
| x_err | 0.307 | 0.72 | 3.3 | 0.9435 |
| y_err | 0.239 | −1.48 | 17.1 | 0.9483 |
| z_err | 0.149 | −0.21 | 3.7 | 0.9384 |
| clock_err | 0.323 | 0.11 | 0.4 | 0.9359 |

W ≈ 0.94 per window — close to normal, not perfectly so. The heavy tails are
real and concentrated in specific windows (regime changes like the BeiDou clock
jump in June, 10–26 m).

**The pooling trap, carried over from Task A:** pooled across satellites these
residuals look badly non-Gaussian (W = 0.51 on the clock) purely because
per-satellite scales differ 11–22×. Pooling Gaussians of different variance
produces a **scale mixture**, which is heavy-tailed by construction. Every
component was normal; the histogram was not. This is why every W above is
reported *per window* and never pooled.

The reported 1-σ is the held-out-day RMS from the self-validation step, which is
an honest estimate rather than a fitted one.

---

## 8. Honest limitations

1. **MEO x/y/z improves by only ~46–53%.** The headline 84% is carried by the
   clock column. The orbit columns are a genuine but moderate win.

2. **On GEO, a simpler method beat us on the primary benchmark.** Fixed harmonic
   regression, no model selection: 0.372 vs 0.382, 15/24 folds. We identified the
   cause, built the fix, and rejected it because it failed to replicate (§3.3).
   A reader is entitled to disagree with that call — the numbers for both are in
   §3.3, and switching is a two-line change in `smart_horizon.py`.

3. **GEO x is our weakest column** (0.643 in the §3 protocol). SARIMA, AR and
   Holt-Winters all beat us there — 0.500–0.527 — while losing badly on z. We did
   not build the hybrid that would take the better of each per column, and it is
   the most obvious remaining gain in the whole project.

4. **The GRU beat us on 5/5 MEO folds.** By 0.4%. We are choosing not to ship it,
   and that is a judgement call, not a measurement — the measurement says it is
   consistently, negligibly better.

5. **Fold-to-fold variance on GEO is 0.058 on a mean of 0.323** — 18%. Smaller
   than Task A's 61%, but the GEO pool is only 15 spacecraft, so a bad fortnight
   for one satellite moves the number.

6. **The reported σ under-covers on the MEO clock by ~1.5×** even after
   calibration, because a few windows contain clock regime jumps. Treat the clock
   σ as a lower bound.

7. **The competition's own PS-08 files are defective** (see
   `WHY_SIH_DATA_FAILS.md`, 15 documented findings). Everything here is validated
   on our own corpus of the identical physical quantity. §10 of that document
   shows the SIH MEO sawtooth grows 5.3× from fresh to stale ephemeris and the
   corpus 4.5×, at the same absolute scale — which is the evidence that the two
   are the same quantity. It is evidence, not proof.

8. **Nothing here is novel.** Harmonic regression with a held-out model-selection
   day is standard practice. The value is in having established *which* basis is
   correct for *which* class, and in having ruled out the alternatives with
   enough folds that the ruling holds.

9. **We ignored the competition's stated evaluation metric** by explicit
   instruction. Everything is scored as RMS in metres per column.

---

## 9. Files

| file | what it is |
|---|---|
| **`smart_horizon.py`** | **the deliverable** — self-contained, no corpus, no weights, no network |
| `taskB_final.py` | the corpus-side implementation used for all validation |
| `taskB_bench2.py` | the 23-method, 24-fold benchmark of §3 |
| `geo_fix.py` | the GEO variant study of §3.3 |
| `taskB_nn.py` | the 5-architecture neural test of §4 |
| `taskB_learn.py` | the gradient-boosting test of §5 |
| `verify_deliverable.py` | checks the shipped file through its public API, on- and off-grid |
| `JOURNEY.md` | the full exploration log, including everything that failed |
| `TASK_A_AND_WHY_WE_LEFT_IT.md` | the operational detour and why it cannot be submitted |
| `WHY_SIH_DATA_FAILS.md` | 15 findings on the delivered dataset |

### Self-test

```
$ python smart_horizon.py
self-test  n_train=185  n_query=96
  x_error RMS vs truth : 0.0127 m   (reported sigma 0.0581)
  clock predicted      : 0.3041  (true 0.30)
```
