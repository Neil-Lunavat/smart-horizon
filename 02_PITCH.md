# SMART-HORIZON — presentation deck

**SH-DST-03: predict day 8 from seven days of satellite error.**

One slide per section. Each has a title, one image, and one line. The spoken
explanation is yours — the full written argument is in
[`01_FULL_STORY.md`](01_FULL_STORY.md).

---

---

## 1 · What we were given

![](figures/f01_raw_data.png)

Six files. Five columns. No satellite name — only "GEO" or "MEO".

---

## 2 · Half of every MEO file is a duplicate

![](figures/f03_duplicates.png)

One backwards jump per file, spanning the whole file — the block was written twice.
Removing it is provably lossless.

---

## 3 · No file actually spans seven days

![](figures/f02_coverage.png)

13% to 32% covered. Gaps up to 26 hours. **The smallest test set is 6 points.**

---

## 4 · The day we must predict is not like the week we are shown

![](figures/f04_distribution_shift.png)

Every GEO channel is wider on day 8 — 1.6× to 4.2× the spread, 2.6× to 17× the variance.

---

## 5 · And it is a ramp, not a glitch

![](figures/f05_volatility_ramp.png)

Volatility doubles every day from day 5 and keeps climbing into day 8.
Cutting the wild day out changes nothing — we tested it.

---

---

## 6 · So first we tried ordinary machine learning

![](figures/f21_ml_plateau.png)

400 windows · 27,000 rows · 19 features · six standard models.
**They all stall at the trivial baseline.**

---

## 7 · Why they stall

> The models are not weak. **The inputs do not determine the output.**
>
> What decides day 8 is where the satellite is in its orbit, which constellation it
> belongs to, and when the ground segment last uploaded.
>
> **None of those are columns in the file.**

So the orbital rhythm is not a pattern hiding in the data.
**It is information we have to bring in from outside it.**

---

---

## 8 · The physics: a geosynchronous satellite repeats every sidereal day

$$T_{\text{sid}} = 86164\ \text{s} = 23^{\text{h}}56^{\text{m}}04^{\text{s}}$$

Not 24 hours. Over a week the two drift **28 minutes** apart.
A 7-day window therefore holds **7.02 exact repeats** — nothing to search for.

---

## 9 · MEO: the period is fixed by design, but we are not told which

|         | period  |
| ------- | ------- |
| GPS     | 11.97 h |
| Galileo | 14.05 h |
| BeiDou  | 12.89 h |

They differ by 17%, and the file does not say which one we have.

---

## 10 · The key result — x and y hide two rhythms, not one

![](figures/f06_beat_frequencies.png)

$$
\sin\alpha\cos\beta=\tfrac12\left[\sin(\alpha-\beta)+\sin(\alpha+\beta)\right]
\;\Longrightarrow\;
P_{\pm}=\frac{1}{\left|\frac{1}{T}\pm\frac{1}{T_{\text{sid}}}\right|}
$$

The ground spins underneath the satellite, so x and y appear at **two beat periods**
— neither of them the orbital period. Only z keeps it.

---

## 11 · Evaluated

|         | x and y appear at          | z stays at |
| ------- | -------------------------- | ---------- |
| GPS     | **7.98 h** and **23.93 h** | 11.97 h    |
| Galileo | **8.85 h** and **34.00 h** | 14.05 h    |
| BeiDou  | **8.38 h** and **27.92 h** | 12.89 h    |

Fitting one rhythm to all three axes is **guaranteed** to be wrong on two of them.

_(GPS 23.93 h is the sidereal day — the known fact that GPS ground tracks repeat
daily, falling out of the algebra. A check that the derivation is right.)_

---

## 12 · And it is there in their own data

![](figures/f07_periodogram.png)

Orange lines placed by physics **before** looking. MEO peaks exactly on them.
**The GEO file does not** — its rhythm is 12 h, which no geostationary satellite has.

---

## 13 · The clock is a different kind of thing

![](figures/f08_clock_random.png)

A random walk plus resets — no daily rhythm exists to find.
Their own file: 143 rows, **70 distinct clock values.**
So we _forbid_ the model from looking for one.

---

---

## 14 · The model — the window grades itself

![](figures/f22_how_it_works.png)

Physics fixes which rhythms are allowed. The caller's own **day 7** picks the basis.
Ridge regression fits the coefficients.

---

## 15 · The jury asked us to generate our own data. We did.

![](figures/f09_synthesis.png)

Every term is a physical effect — offset, orbital harmonics at the derived beat
periods, drift between uploads, noise, and a clock built as a random walk.
Sampled at **their own irregular timestamps and gaps.**

---

## 16 · Then we hid day 8 and asked for it back

![](figures/f10_recovery_examples.png)

The model has never seen the black curve.

---

## 17 · 120 windows, each a different satellite

![](figures/f11_recovery_distribution.png)

|     | do nothing | **model** |                |
| --- | ---------- | --------- | -------------- |
| GEO | 0.804      | **0.173** | **78% better** |
| MEO | 0.760      | **0.373** | **51% better** |

---

## 18 · What it survives, and what kills it

![](figures/f12_sensitivity.png)

Quadruple the noise: it barely notices.
Corrupt **one row in ten**: it goes below useless.

---

---

## 19 · On their MEO data — it works

![](figures/f14_results.png)

**57% and 19% better than doing nothing** — and better than a predictor
**shown the answer key's own average.**

---

## 20 · It follows the shape of the day, not just the level

![](figures/f15_meo_predictions.png)

One clean case, one noisy case. Both shown.

---

## 21 · Sixteen methods, ranked on their data alone

![](figures/f20_benchmark.png)

Fit days 1–6, score day 7. First by **28%** over the runner-up.

---

## 22 · On their GEO data — nothing works

Every method scores **14.95 m**: ours, zero, the average, the last value.

> Give a predictor **the answer key's own average** — outright cheating —
> and it improves on doing nothing by **0.3%.**
>
> **There is nothing in that file to predict.**

---

## 23 · Because the file is not physical

![](figures/f16_geo_artifact.png)

Consecutive readings are near-exact opposites. **All four channels flip together** —
position and clock come from unrelated mechanisms and cannot do that.
The 24 largest jumps alternate **24 out of 24**. Odds: **1 in 8 million.**

---

## 24 · Proof: we reproduce the failure on demand

![](figures/f13_artifact_injection.png)

Clean synthetic signal → **0.017 m.**
Same physics, same noise, plus that sign pattern → **0.743 m.**

**This is what happened to their GEO file.**

---

---

## 25 · Their scoring metric — three findings

**Their reference file does not match the test they name.**
Standard Shapiro-Wilk gives 0.9852; they state 0.9810. **Shapiro-Francia** gives
their p-value exactly. We rebuilt their scorer to match.

---

## 26 · The metric rewards noise and is blind to accuracy

![](figures/f17_metric_broken.png)

Predict-zero, predict-average and predict-last-value: **identical scores.**
A **99% accurate** model: same score as doing nothing.
**Pure random noise: 0.98.**

---

## 27 · And a perfect answer cannot score 1.0

![](figures/f18_ceiling.png)

At 6 test points a flawless model averages **0.92** and lands anywhere
in **0.79–0.98** by chance. Our 0.907 is already at that ceiling.

---

---

## 28 · Why this will not overfit your evaluation set

The model fits its coefficients **inside whatever window you hand it**, at call time.
There is no stored training set for new data to be inconsistent with.

We showed this on 120 generated windows it had never seen.

**And if the evaluation set is GEO-like, every team's score will be noise —
we are the team that measured that in advance.**

---

---

## Closing

|                  |                                                                            |
| ---------------- | -------------------------------------------------------------------------- |
| **The data**     | half duplicated · 13–32% covered · 6-point test set · volatility ramps 10× |
| **Pure ML**      | six models, 27,000 rows — all plateau at the baseline                      |
| **The physics**  | orbit class fixes the rhythm; ECEF splits x and y into two beats           |
| **Validation**   | 120 generated windows, known answers: 78% / 51% better                     |
| **Their MEO**    | 57% and 19% better — beats a predictor shown the answer                    |
| **Their GEO**    | unpredictable, and we prove it three separate ways                         |
| **Their metric** | rewards noise, blind to accuracy, mostly luck at n = 6                     |

**Live demo:** `interactive.html` — move the sliders and break the model yourself.
