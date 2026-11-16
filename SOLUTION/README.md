# SMART-HORIZON

**Predicts day-8 GNSS broadcast ephemeris and clock error from 7 days of history,
at any timestamps you ask for.**

One file. No trained weights, no model download, no training step, no network
access, no satellite ID required. Dependencies are numpy and pandas.

```
SOLUTION/
├── smart_horizon.py     the model — this is the whole thing
├── run_example.py       runnable demo, scores itself against real held-back data
├── example_data/        two real 7-day windows (one GEO, one MEO) + truth
├── requirements.txt
└── README.md            you are here
```

**Try it right now:**

```bash
pip install -r requirements.txt
python run_example.py
```

---

## 1. What it takes in

```python
from smart_horizon import predict

out = predict(history_df, query_times, orbit="MEO")
```

### `history_df` — 7 days of error history

A pandas DataFrame (or a CSV via `predict_csv`) with these five columns:

| column | meaning | units |
|---|---|---|
| `utc_time` | timestamp | anything `pd.to_datetime` accepts |
| `x_error` | ECEF X error (broadcast − precise) | metres |
| `y_error` | ECEF Y error | metres |
| `z_error` | ECEF Z error | metres |
| `satclockerror` | clock error | metres |

**It is deliberately forgiving.** The input may be irregularly sampled, unsorted,
contain duplicate rows, and have multi-hour gaps — all of that is handled. The
examples in `example_data/` are ~28% of a 15-minute grid, which is roughly what
the competition supplies. Minimum 20 unique rows.

These aliases are accepted and renamed automatically: `clock_error`, `clock`,
`x_err`, `y_err`, `z_err`.

### `query_times` — when you want predictions

Any list, Series, or `DatetimeIndex`. **Completely unconstrained**: irregular
spacing, epochs off any grid, one timestamp or ten thousand. The model is a
continuous function of time, not a step-ahead forecaster — there is no horizon to
configure and nothing to unroll.

### `orbit` — `"MEO"` or `"GEO"`

The competition tells you this. It selects between two genuinely different
models; see §4. `"GSO"` is treated as `"GEO"` (anything starting with G).

---

## 2. What it gives back

A DataFrame with one row per query time and nine columns:

```
            utc_time  x_error  y_error  z_error  satclockerror  x_error_sigma  ...
 2025-06-01 00:00:00    0.876    0.128    0.803          5.260          0.239  ...
 2025-06-01 00:15:00    0.831    0.212    0.755          5.260          0.239  ...
```

* the four predicted errors, in metres, at each requested timestamp
* a **1-σ uncertainty per column** — an honest per-window estimate of how much to
  trust these numbers, derived by holding out the window's own final day

The σ columns are the part most submissions won't have. They let you say *how
confident* the model is on this particular satellite and this particular week,
rather than quoting one global accuracy figure.

### Straight from files

```python
from smart_horizon import predict_csv
predict_csv("history.csv", "query_times.csv", orbit="GEO", out_csv="day8.csv")
```

Runtime is **~65 ms per window** on one CPU core.

---

## 3. How accurate it is

RMS error in metres, mean over the four columns, on held-out windows from a
244-day × 103-satellite corpus.

| | do nothing (predict 0) | window mean | **SMART-HORIZON** | better by |
|---|---|---|---|---|
| **GEO / GSO** | 2.806 | 0.548 | **0.244** | **91.3%** |
| **MEO** | 2.518 | 0.728 | **0.397** | **84.2%** |

Per column:

| | x | y | z | clock |
|---|---|---|---|---|
| GEO — do nothing | 2.178 | 1.682 | 0.941 | 6.425 |
| GEO — **model** | **0.326** | **0.238** | **0.144** | **0.268** |
| MEO — do nothing | 0.807 | 0.821 | 0.815 | 7.628 |
| MEO — **model** | **0.434** | **0.445** | **0.381** | **0.328** |

**Read this honestly:** on MEO, the large headline number is carried by the clock
column (95.5% better). The orbit columns improve by 46–53%, which is a real but
moderate win. On GEO all four columns improve by 84–95%.

### It is stable

Six independent, non-overlapping time blocks, 220 windows each:

| | mean | sd | range |
|---|---|---|---|
| GEO | 0.323 | ±0.058 | [0.256, 0.417] |
| MEO | 0.385 | ±0.027 | [0.340, 0.413] |

**MEO varies by 7% across eight months of data.** That is a direct consequence of
having no trained parameters — the model cannot be lucky or unlucky about a
training period, because it never saw one.

### Arbitrary (off-grid) timestamps

Verified through the public API with query epochs jittered off the 15-minute
grid, so the model is asked for times it has never been handed:

| query times | GEO | MEO |
|---|---|---|
| on-grid 15-min | 0.280 | 0.390 |
| **arbitrary off-grid** | **0.312** | **0.467** |

It degrades ~10% (GEO) and ~20% (MEO) and keeps working. Part of that gap is
error in the interpolated reference values, not the model.

### Uncertainty calibration

The reported σ, checked against actual error on 700 windows:

| | x | y | z | clock |
|---|---|---|---|---|
| GEO, actual/reported | 1.11 | 0.90 | 0.95 | 0.99 |
| MEO, actual/reported | 1.11 | 1.14 | 1.28 | **1.46** |

GEO is well calibrated. **The MEO clock σ under-covers by about 1.5× — treat it
as a lower bound.** The shortfall comes from a small number of windows containing
clock regime jumps; inflating σ everywhere would make it wrong for the other 95%.

---

## 4. How it works

### The core idea

The error is not noise. It is a **periodic signal** whose period is set by orbital
mechanics, plus a **slowly-drifting clock offset**. So the model fits the right
periodic basis to your 7 days and evaluates it forward. Nothing is memorised.

### Two models, because GEO and MEO are different physics

**GEO / GSO — the period is known before you see any data.**
A geosynchronous satellite's ground track repeats at the **sidereal day**
(86,164 s), so a 7-day window contains **7 clean cycles**. There is nothing to
search for. Only the harmonic order K ∈ {1,2,3,4} is chosen.

**MEO — the period is unknown, and the three axes do not share it.**
The orbital period differs per constellation — GPS 43,082 s, Galileo 50,567 s,
BeiDou 46,393 s — and *you are not told which constellation it is*. Worse, in the
Earth-fixed (ECEF) frame the axes sit at different frequencies:

* **z** oscillates at the orbital period **T**
* **x and y** are beaten against Earth's rotation, at **1 / (1/T ± 1/T_sid)**

So T is estimated from your window by a **Lomb-Scargle periodogram** (which
handles irregular sampling natively) restricted to the physically possible MEO
band of 38,000–54,000 s, and the two beat periods are then derived
*analytically* rather than searched for.

This axis-frequency split is the single most important finding in the project.
Fitting one shared period to x, y and z — which is what a textbook approach does
— costs **45% accuracy** on MEO (see `harm-fixed` in §6).

**Clock — a level estimator, always, both classes.**
The day-8 clock mean is statistically independent of the 7-day window mean
(R² = 0.000). There is no 24-hour memory to exploit. Harmonics on the clock made
it worse every one of the six ways it was tried. So the clock gets one of seven
level estimators (`mean_all`, `mean_3d`, `mean_2d`, `mean_1d`, `mean_12h`,
`ewma_1d`, `ewma_2d`) and nothing else. This *restriction* is worth 93–95% on the
clock column.

### Self-validation: how it adapts with nothing trained

For every column, the model:

1. fits every candidate basis on **days 1–6** of your window,
2. scores each on **day 7**, held out,
3. refits the winner on **all 7 days**,
4. reports the held-out RMS as the **1-σ**.

The held-out day is the only model-selection signal, and it comes from your own
data. That is what makes it adaptive without being trained — it picks a different
basis for a quiet satellite than for a noisy one, per window, at call time.

A divergence guard clamps any harmonic extrapolation to within 2× the window's
observed range, as insurance on inputs unlike anything in the corpus.

---

## 5. Why *this* model — why nothing is trained

The obvious objection: there is a 244-day, 103-satellite corpus available, so why
ship a model with no learned weights?

**Because learning was tried, repeatedly, and it did not help.** Every result
below is cross-validated, never a single fold.

### Neural networks, predicting the values directly

5 architectures, 5 rolling-origin folds, retrained from scratch per fold, on a
rented GPU:

| | GEO | MEO | folds won |
|---|---|---|---|
| MLP | +56.3% worse | +65.8% worse | 0/5 |
| GRU | +57.3% | +64.8% | 0/5 |
| LSTM | +60.8% | +72.4% | 0/5 |
| TCN | +57.7% | +71.4% | 0/5 |
| Transformer | +59.9% | **+83.9%** | 0/5 |

Every architecture 56–84% worse, **0/5 folds won in every single case**. The
transformer used continuous time encoding, built specifically for irregular
sampling, and came last on MEO.

Telling detail: the **MLP** — which sees the window as an unordered bag of
numbers — ties the GRU and beats the LSTM and transformer. When architectures
that model sequence structure can't beat one that ignores it, the sequence
structure is not where the information is.

### Neural networks, correcting the physics model's residual

The fair test — they only have to find what the physics missed:

| | GEO | MEO | folds won |
|---|---|---|---|
| MLP | +0.4% | +1.3% | 1/5, 0/5 |
| GRU | −0.1% | **−0.4%** | 2/5, **5/5** |
| TCN | +0.8% | +1.1% | 0/5, 0/5 |
| Transformer | +0.0% | +0.0% | 1/5, 2/5 |

All tie. The GRU wins 5/5 MEO folds consistently — **by 0.4%**, which is 0.397 m
versus 0.395 m in exchange for shipping a neural network. Not worth it.

### Gradient boosting on the residual

Given 15 window-summary features *plus the physics model's own prediction*:

| | x | y | z | clock |
|---|---|---|---|---|
| GEO | **+14.3% worse** | +9.0% | +7.4% | +11.2% |
| MEO | −0.9% | +0.2% | +0.9% | +3.5% |

Made GEO worse on every single target.

### Horizon shrinkage

| | model | model + learned shrinkage |
|---|---|---|
| GEO | **0.244** | 0.254 |
| MEO | **0.397** | 0.401 |

Also worse.

### So what was the corpus for?

**It determined the model's design. It just isn't stored inside it.**

| what the corpus established | what it fixed in the model |
|---|---|
| GEO error repeats at the sidereal day | the GEO basis — no search needed |
| true GEO (J07/J08) is slow drift, not periodic (R² 0.004) | level estimators stay in the GEO candidate set |
| MEO z at T, x/y at Earth-rotation beats | the entire MEO basis construction |
| MEO period differs per constellation and isn't supplied | forces per-window Lomb-Scargle |
| clock has no 24 h memory (R² = 0.000) | harmonics banned on the clock — worth 93–95% |
| extra basis functions fit noise that doesn't extrapolate | K capped, self-validation required |
| fixed periods hurt MEO z (0.401 → 0.546) | half-sidereal is right for GPS, wrong for Galileo/BeiDou |

Seven independent attempts to add learned parameters on top all failed. The
honest conclusion is that **there is no residual structure left for weights to
capture** — so shipping weights would only add a way to be wrong on satellites
and epochs the training set never saw.

A model with no trained parameters cannot be distribution-shifted. On an
evaluation set drawn from an unknown period and possibly an unknown
constellation, that is the design's main safety property.

---

## 6. Benchmark scores

**23 methods, 24 blocked cross-validation folds each.** Random contiguous 2-week
test blocks, training data (for methods that use any) strictly before the block
with an 8-day gap. GEO and MEO pools built separately. Identical windows and
identical metric for every method.

### MEO — a clean sweep

| method | mean | sd | best | worst | vs ours | folds won |
|---|---|---|---|---|---|---|
| **ours** | **0.458** | 0.038 | 0.409 | 0.517 | — | — |
| ar-yw | 0.556 | 0.060 | 0.470 | 0.637 | +21.4% | 0/24 |
| sarima | 0.582 | 0.086 | 0.477 | 0.694 | +26.8% | 0/24 |
| fft | 0.608 | 0.079 | 0.502 | 0.729 | +32.7% | 0/24 |
| harm-fixed | 0.664 | 0.057 | 0.583 | 0.742 | +44.8% | 0/24 |
| arima | 0.664 | 0.088 | 0.548 | 0.790 | +44.8% | 0/24 |
| snaive-avg | 0.666 | 0.058 | 0.585 | 0.745 | +45.2% | 0/24 |
| extratrees | 0.684 | 0.056 | 0.604 | 0.763 | +49.1% | 0/24 |
| svr | 0.698 | 0.054 | 0.609 | 0.784 | +52.2% | 0/24 |
| gbm | 0.706 | 0.057 | 0.635 | 0.789 | +53.9% | 0/24 |
| rf | 0.736 | 0.068 | 0.639 | 0.828 | +60.4% | 0/24 |
| knn | 0.740 | 0.070 | 0.637 | 0.842 | +61.3% | 0/24 |
| ridge | 0.742 | 0.070 | 0.664 | 0.845 | +61.7% | 0/24 |
| window mean | 0.756 | 0.059 | 0.678 | 0.840 | +65.0% | 0/24 |
| median | 0.765 | 0.059 | 0.686 | 0.846 | +66.9% | 0/24 |
| drift | 0.790 | 0.063 | 0.703 | 0.889 | +72.2% | 0/24 |
| snaive | 0.894 | 0.102 | 0.782 | 1.041 | +95.0% | 0/24 |
| holt-winters | 0.919 | 0.090 | 0.815 | 1.050 | +100.5% | 0/24 |
| gaussian process | 0.970 | 0.112 | 0.830 | 1.135 | +111.6% | 0/24 |
| prony | 1.010 | 0.255 | 0.743 | 1.344 | +120.4% | 0/24 |
| persistence | 1.035 | 0.088 | 0.938 | 1.171 | +125.8% | 0/24 |
| ssa | 1.080 | 0.361 | 0.716 | 1.592 | +135.7% | 0/24 |
| zero | 2.076 | 0.168 | 1.755 | 2.475 | +352.9% | 0/24 |

**Nothing beat it on a single one of 24 folds**, and the runner-up is 21% behind.

Two results worth pulling out:

* **`harm-fixed` +44.8%** — the same harmonic basis fitted at one shared period
  with no per-window selection. Its z column (0.395) is actually *better* than
  ours (0.398); all the damage is in x and y (0.835/0.846 vs 0.472/0.475). That
  is precisely the beat-frequency structure of §4, and it is the single
  measurement that justifies the whole MEO design.
* **Gaussian process +111.6%** — the textbook answer for irregularly sampled
  quasi-periodic data, losing badly for the same reason: a periodic×RBF kernel
  forces one shared period onto all three axes.

### GEO — where we lost, and what we did about it

| method | mean | sd | worst | vs ours | folds won |
|---|---|---|---|---|---|
| **harm-fixed** | **0.372** | 0.075 | 0.499 | **−2.6%** | **15/24** |
| **ours** | 0.382 | **0.056** | **0.477** | — | — |
| snaive | 0.429 | 0.072 | 0.541 | +12.3% | 2/24 |
| sarima | 0.440 | 0.069 | 0.552 | +15.2% | 1/24 |
| ar-yw | 0.446 | 0.067 | 0.550 | +16.7% | 0/24 |
| snaive-avg | 0.447 | 0.142 | 0.667 | +17.1% | 11/24 |
| holt-winters | 0.466 | 0.063 | 0.585 | +22.2% | 0/24 |
| arima | 0.513 | 0.055 | 0.602 | +34.3% | 0/24 |
| gaussian process | 0.538 | 0.063 | 0.630 | +41.0% | 0/24 |
| ridge | 0.570 | 0.027 | 0.606 | +49.3% | 0/24 |
| window mean | 0.591 | 0.046 | 0.665 | +54.8% | 0/24 |
| *(median, drift, gbm, extratrees, fft, rf, ssa, persistence, knn, svr, prony)* | 0.607–0.863 | | | +59% to +126% | 0/24 |
| zero | 2.417 | 0.472 | 2.952 | +533.2% | 0/24 |

**On GEO a simpler method beat us.** `harm-fixed` — fixed harmonic regression
with *no model selection at all* — scored 0.372 against 0.382 and won 15/24
folds.

We chased it down rather than leaving it. The mechanism is identified: on GEO the
period is known exactly, so selection has nothing useful to choose, and a noisy
held-out day occasionally picks a flat line where a harmonic would have
extrapolated correctly. The controlled comparison is `fixed-K2` (0.378) versus
`fixed-K2+lvl` (0.392) — identical basis, the only difference being whether level
estimators may win. **Allowing them costs 3.7%.**

**We built the fix, then rejected it.** Barring the level estimators was
implemented and tested against an independent sample through the public API:

| GEO, independent sample | with levels (shipped) | without levels |
|---|---|---|
| on-grid | 0.280 | 0.277 |
| **off-grid** | **0.312** | **0.374** |
| off-grid x | 0.497 | **0.755** |

The gain **failed to replicate** and reversed into a 20% loss on off-grid query
times, concentrated entirely in x. Clamping the harmonic did not recover it — so
it is not a divergence artefact; on those windows the harmonic simply has the
wrong shape and the flat estimator is the better answer. Consistently, `ours` has
the **lowest fold variance of any GEO variant** (sd 0.056 vs 0.071–0.075) and the
**best worst-case fold** (0.477 vs 0.489+).

**Decision: keep the level estimators.** An unreplicated 3% mean gain that costs
tail robustness is not worth shipping. This is written down rather than buried,
because anyone who runs `harm-fixed` on GEO will find the same thing.

### What self-validation is worth, by class

| | ours | harm-fixed | selection is worth |
|---|---|---|---|
| MEO | 0.458 | 0.664 | **+44.8%** |
| GEO | 0.382 | 0.372 | **−2.6%** |

Self-validation pays **only where the period must be estimated**. Where it is
known a priori it is a slight liability. That is a coherent result, and it is the
strongest single argument for GEO and MEO being separate models.

> The §6 figures (0.382 / 0.458) are higher than the §3 headline (0.244 / 0.397)
> because this protocol samples each orbit class uniformly across all its
> spacecraft, so hard satellites carry the same weight as easy ones. It is the
> harsher measurement, and every method here is scored on identical windows,
> which is what makes the comparison valid.

---

## 7. Known limitations

1. **MEO x/y/z improves by only ~46–53%.** The 84% headline is carried by the
   clock column.
2. **On GEO, fixed harmonic regression beats us by 2.6%** on the primary
   benchmark. We rejected the fix for the reasons in §6; a reader is entitled to
   disagree, and switching is a two-line change.
3. **GEO x is the weakest column** (0.643 under the §6 protocol). SARIMA, AR and
   Holt-Winters all beat us there (0.500–0.527) while losing badly on z. A hybrid
   taking the better of each per column was not built, and is the most obvious
   remaining gain.
4. **A GRU on the residual beats us on 5/5 MEO folds** — by 0.4%.
5. **The MEO clock σ under-covers by ~1.5×**, from a few windows with clock
   regime jumps. Treat it as a lower bound.
6. **Nothing here is novel.** Harmonic regression with a held-out selection day
   is standard practice. The contribution is establishing *which* basis is
   correct for *which* class, and ruling out the alternatives across enough folds
   that the ruling holds.
7. **Validated on our own corpus, not the supplied PS-08 files**, which are
   defective — see `WHY_SIH_DATA_FAILS.md` in the parent directory for the 15
   documented findings. The corpus measures the identical physical quantity at
   the same absolute scale.

---

## 8. Further reading (parent directory)

| file | what it covers |
|---|---|
| `TASK_B_FINAL.md` | the long-form version of this document, with per-fold values |
| `JOURNEY.md` | the full exploration log, including everything that failed |
| `TASK_A_AND_WHY_WE_LEFT_IT.md` | the operational reframing, and why it can't be submitted |
| `WHY_SIH_DATA_FAILS.md` | 15 findings on the delivered dataset |
| `taskB_bench2.py` | the 23-method, 24-fold benchmark of §6 |
| `taskB_nn.py` | the 5-architecture neural test of §5 |
| `geo_fix.py` | the GEO variant study of §6 |
| `verify_deliverable.py` | checks this file through its public API, on- and off-grid |
