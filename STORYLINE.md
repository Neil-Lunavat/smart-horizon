# Storyline — the working list

Every point we can make, in order, with the evidence that backs it. This is the
planning document: `01_FULL_STORY.md` is the written version and `02_PITCH.md` is
the reduced version for the presentation.

**Legend** — 📊 figure exists · 📋 table exists · ⭐ in the 10-minute pitch

---

## Words the audience will need

| term | plain meaning |
|---|---|
| **Residual** | how wrong we were: prediction minus truth |
| **RMS** | the average size of our mistakes, in metres. Lower is better. |
| **W (Shapiro-Wilk)** | a 0–1 score for "do our mistakes look like random bell-curve noise?" **This is what the competition grades on.** It says nothing about whether the mistakes are big. |
| **Baseline** | the dumbest sensible answer — predict zero, predict last week's average. If you can't beat these you have nothing. |
| **Oracle / "cheat"** | a predictor secretly shown the answer key. Not shippable — it exists to mark the ceiling. If even the cheat can't do well, nobody can. |
| **Autocorrelation** | does each reading resemble the one before it? Real satellite data ≈ 0.99. Random noise ≈ 0. |
| **Periodogram** | a chart showing which repeating rhythms live inside a wiggly signal. |

---

## ACT 1 — What is actually in the box

| # | point | evidence |
|---|---|---|
| 1 | Six files, five columns, no satellite identifier. We are told only "GEO" or "MEO". | 📊 `f01` ⭐ |
| 2 | Half of every MEO file is duplicated rows: 90→46, 244→143, 11→6, 30→18. | 📊 `f03` ⭐ |
| 3 | It is a clean copy-paste — exactly one backwards time-step per file, spanning the whole file — so de-duplicating is provably lossless. | 📊 `f03` |
| 4 | The GEO files contain no duplicates. A different problem entirely. | 📋 |
| 5 | No file spans seven days. Coverage runs 13%–32% of the week. | 📊 `f02` ⭐ |
| 6 | Gaps reach 26 hours. `MEO_Train` is seven ~7-hour bursts separated by 21-hour holes. | 📊 `f02` |
| 7 | Because neighbouring readings are nearly identical, `MEO_Train` holds only **1–5 genuinely independent measurements**. | 📋 |
| 8 | `DATA_MEO_Test.csv` is **six data points**. Any score computed on it is mostly luck. | 📋 ⭐ |
| 9 | Day 8 is not the same kind of day as days 1–7: GEO test variance is 7.6×–17× the training week (Levene p < 0.003). | 📊 `f04` ⭐ |
| 10 | The training week is not stationary within itself either. | 📊 `f05` |
| 11 | **The volatility climbs steadily through the week and keeps climbing into day 8.** Day-by-day standard deviation of y: 1.9 → 1.7 → 1.9 → 3.2 → 2.4 → 7.3 → **14.0**, then **19.8** on the day we must predict. It is a ramp, not a glitch. A model fitted on the quiet early days is graded on the wild end — and no amount of cleaning fixes that, because the wildness is the signal, not an outlier. | 📊 `f05` ⭐ |
| 12 | Small stuff we handled: US date format that a naive parser reads wrong once the day exceeds 12, and a stray double space in one column header. | — |

---

## ACT 1.5 — We tried ordinary machine learning first, and it plateaued

| # | point | evidence |
|---|---|---|
| 13 | We treated it as a pure ML problem first: 400 windows, ~27,000 training rows, 19 engineered features (level, spread, trend, shape, time of day, hours ahead), six standard regressors — ridge, k-NN, random forest, gradient boosting, SVR, neural network. | 📊 `f21` ⭐ |
| 14 | **They all cluster just around the trivial baseline.** Best learner (gradient boosting) 0.4206 m against 0.4516 m for simply predicting the week's mean — a 7% gain from a large model on a large training set. | 📊 `f21` ⭐ |
| 15 | **The reason is not that the models are weak. It is that the inputs do not determine the output.** What actually decides day 8 is where the satellite sits in its orbit, which constellation it belongs to, and when the ground segment last uploaded new data. **None of those are columns in the file.** The learner is being asked to infer something that carries almost no shared information with anything it can see — so it correctly learns almost nothing. | 📊 `f21` ⭐ |
| 16 | This is the hinge of the whole project: **the orbital rhythm is not a pattern hiding in the data. It is information we bring in from outside it.** That is precisely why physics succeeds where learning plateaus — and the same physics model scores 0.3794 on the identical held-out windows, 10% ahead of the best learner. | 📊 `f21` ⭐ |

---

## ACT 2 — What physics says must be true

| # | point | evidence |
|---|---|---|
| 17 | Being told "GEO" or "MEO" is the only free information in the problem — and it is worth more than the five columns, because it fixes the orbital period. | 📋 |
| 18 | A geosynchronous satellite repeats every **sidereal day, 23h 56m 04s** — not 24h. Over seven days the two drift 28 minutes apart, enough to wreck a fit. | LaTeX ⭐ |
| 19 | So a 7-day window contains **7.02 clean repeats**. Nothing needs to be searched for. | LaTeX |
| 20 | MEO periods are fixed by constellation design — GPS 11.97 h, Galileo 14.05 h, BeiDou 12.89 h — **and we are not told which one we have.** | 📋 ⭐ |
| 21 | **The key derivation.** In Earth-fixed coordinates the axes do not share a rhythm. The z axis is the rotation axis and keeps the orbital period T. But x and y are multiplied by the Earth's rotation, and sin·cos splits one tone into two: the error appears at the **beat periods** 1/(1/T ± 1/T_sid). | 📊 `f06` LaTeX ⭐ |
| 22 | Evaluated: GPS 7.98 h and 23.93 h; Galileo 8.85 h and 34.00 h; BeiDou 8.38 h and 27.92 h. The GPS 23.93 h result *is* the sidereal day — the textbook fact that GPS ground tracks repeat daily, falling straight out of the algebra. A check that the derivation is right. | 📋 ⭐ |
| 23 | Consequence: a model that fits one rhythm to all three axes is **guaranteed** to be wrong on two of them. | 📊 `f06` |
| 24 | GEO and MEO are the same equation at two points: set T = T_sid and the beats collapse back onto the sidereal day. Not two models — one model, two settings. | LaTeX |
| 25 | **Check it against their own data.** Put the physics-predicted periods on a periodogram of the delivered files *before* looking. MEO-1 and MEO-2 peak exactly on the MEO orbital band. | 📊 `f07` ⭐ |
| 26 | And the GEO file does **not** peak at the sidereal day — it peaks near 12 h, which no geostationary satellite does. First independent hint that the GEO file is not what it claims to be. | 📊 `f07` |
| 27 | The clock is different in kind: an atomic clock's error is a random walk (thermal noise, ageing) punctuated by step resets when the ground segment uploads. **Neither component has a daily rhythm.** | 📊 `f08` LaTeX ⭐ |
| 28 | Their own data agrees: `MEO_Train2` has 143 rows but only **70 distinct clock values**, one repeating 17 times — a step process, not a drifting curve. | 📊 `f08` |
| 29 | So we **forbid** the model from fitting rhythms to the clock. Giving it more freedom there makes it worse: it fits noise and extrapolates it into day 8. | 📋 |
| 30 | The window's own offset is mathematically unrecoverable — there is no absolute reference inside a window to measure it against. For the clock, where there is no oscillation, that offset *is* the whole signal. A limit of the problem, not of the method. | LaTeX |

---

## ACT 3 — The model, and how we validated it without outside data

| # | point | evidence |
|---|---|---|
| 31 | The model is a **regularised regression onto a basis of orbital harmonics**, in three layers: the hypothesis space is fixed by physics; the basis is chosen per window by held-out validation; the coefficients are fitted by ridge regression. | 📊 diagram ⭐ |
| 32 | **The window grades itself.** Fit every candidate on days 1–6, score on day 7, keep the winner, refit on all seven. The only selection signal comes from the caller's own data. | 📊 diagram ⭐ |
| 33 | The menu: seven level estimators (mean, recent means, exponentially-weighted means) plus harmonic bases at the physically admissible periods. | 📋 |
| 34 | For MEO the period is estimated per window by Lomb-Scargle — which tolerates gaps — restricted to the 38 000–54 000 s corridor physics allows. The two beat periods are then derived by formula, never searched. | 📋 |
| 35 | A divergence clamp stops any fitted curve running away outside the range it was fitted on. | 📋 |
| 36 | **The jury asked us to generate our own data, so we did.** Every term is a documented physical effect: a static offset, orbital harmonics at the derived beat periods, slow drift between uploads, measurement noise, and a clock built as a random walk with resets. | 📊 `f09` ⭐ |
| 37 | We sample it at **the delivered files' own irregular timestamps, gaps and coverage**, so the test is as hard as the real thing. | 📊 `f09` |
| 38 | **The validation.** Generate a window, hide day 8, ask the model for it, compare against the truth we planted. The model has never seen the answer. | 📊 `f10` ⭐ |
| 39 | Across **60 independent windows per orbit class** — each with a different period, phase, drift and noise draw — the model is **80% better than doing nothing on GEO and 52% on MEO**, and beats the 7-day average in 100% (GEO) and 78% (MEO) of windows. | 📊 `f11` ⭐ |
| 40 | The model used here was **never fitted to the synthetic test set**. Its design was fixed on separate generated data before these windows were drawn. | 📋 |
| 41 | **Stress test — noise.** Quadrupling the measurement noise costs only a few points; the model holds ~50% improvement from noise 0.01 m to 0.40 m and still gives 35% at 1.0 m. | 📊 `f12` ⭐ |
| 42 | **Stress test — the artefact.** Corrupting even one row in ten sends the model *below useless*, from +51% to −288%. | 📊 `f12` ⭐ |
| 43 | **The interactive demo.** Sliders for noise, corruption, gaps and orbit class, showing live what happens to the day-8 prediction. Lets a judge break the model themselves. | 🖥️ `interactive.html` ⭐ |

---

## ACT 4 — Results on the delivered data

| # | point | evidence |
|---|---|---|
| 44 | **MEO-1: 57% better than doing nothing** — 0.444 → 0.190 m. | 📊 `f14` ⭐ |
| 45 | **MEO-2: 19% better** — 0.164 → 0.133 m. | 📊 `f14` ⭐ |
| 46 | **It beats the cheat on both** — 0.190 vs 0.211 and 0.133 vs 0.136. A predictor shown the answer key's own average still loses. That is only possible by tracking the shape *within* the day, which no constant can do. | 📊 `f14` ⭐ |
| 47 | Visibly: the prediction follows the real curve while last week's average sits flat and wrong. One clean case and one noisy one, both shown. | 📊 `f15` ⭐ |
| 48 | **Sixteen methods ranked using only the delivered data** — fit days 1–6, score day 7. Ours is first at 0.278; the next best is 0.386. No outside data anywhere. | 📊 `f20` ⭐ |
| 49 | **GEO: nothing works.** Every method — ours, zero, the average, the last value — lands within 0.02 m of each other at ~14.96 m. | 📊 `f14` ⭐ |
| 50 | **And here is the proof it is the file, not us.** A cheat handed the answer key's own average improves on doing nothing by **0.3%**. There is nothing there to predict, and that conclusion needs no faith in our model. | 📊 `f14` ⭐ |
| 51 | The GEO file flips sign almost every reading — neighbouring measurements are near-exact opposites, worst cosine −0.9994. | 📊 `f16` ⭐ |
| 52 | **All four channels flip together.** Position error comes from orbital dynamics, clock error from an atomic oscillator. They are unrelated mechanisms and cannot reverse in lockstep. | 📊 `f16` ⭐ |
| 53 | **The alternation is perfect: the 24 largest excursions run + − + − + − … 24 out of 24.** Odds by chance ≈ 1 in 8 million. | 📊 `f16` ⭐ |
| 54 | We tried to repair it. If it were a sign-convention bug the signs could be propagated back — but the *size* of each reading, which no sign flip can change, is itself incoherent. There is no correct series hiding underneath. | 📋 |
| 55 | We tried to model the artefact deliberately. A cheat handed the answer key and told to delete the 24 bad jumps becomes 88% more accurate — **and its normality score gets worse.** The artefact cannot be turned into an advantage. | 📋 |
| 56 | The flip is not predictable from anything supplied — not sample number, not clock time, not the gaps. | 📋 |
| 57 | **Proof by construction.** Inject exactly this artefact into clean synthetic data and a model that scored 0.017 m collapses to 0.743 m, falling back to a flat line. Identical underlying physics; the only change is the alternating sign. **This is what happened to their GEO file.** | 📊 `f13` ⭐ |

---

## ACT 5 — The scoring metric

| # | point | evidence |
|---|---|---|
| 58 | We implemented their metric to grade ourselves — and their own reference file does not match the test they name. Standard Shapiro-Wilk gives W = 0.9852; they state 0.9810. | 📋 ⭐ |
| 59 | **Shapiro-Francia** reproduces their p-value to four decimals (0.5838 vs 0.5840). Their scorer switches between the two tests depending on the data's kurtosis. We reproduce that behaviour, so we grade ourselves as they will. | 📋 ⭐ |
| 60 | **Their metric cannot separate three standard answers.** Predict-zero, predict-average and predict-last-value score *identically to four decimals* on all three files. | 📊 `f17` ⭐ |
| 61 | **A model that is 99% accurate scores exactly the same as doing nothing** — 0.7708 both. Mathematically forced: the score is unchanged by scaling. | 📊 `f17` ⭐ |
| 62 | **Pure random noise scores 0.98**, near the top of the scale, while being 5× less accurate. | 📊 `f17` ⭐ |
| 63 | **So the metric rewards noise and is blind to accuracy.** | 📊 `f17` ⭐ |
| 64 | A perfect answer cannot score 1.0 anyway. At 6 test points a flawless model averages 0.92 and lands anywhere in 0.79–0.98 by chance — a spread wider than the gap between any two serious teams. | 📊 `f18` ⭐ |
| 65 | Our MEO-1 score (0.907) is already at that ceiling and rejects nothing at the 5% level. | 📊 `f18` |
| 66 | Q-Q plots for all three files, as Priority 3 requires. | 📊 `f19` |
| 67 | Four concrete recommendations to the organisers. | 📋 |

---

## ACT 6 — Why this will not overfit the evaluation set

| # | point | evidence |
|---|---|---|
| 68 | The model's coefficients are fitted **inside whatever window it is handed**, at call time. There is no stored training set for a new satellite, month or constellation to be inconsistent with. | 📋 ⭐ |
| 69 | So if the evaluation data is different — another satellite, another period, tampered the same way — the model adapts rather than mis-remembering. We demonstrated exactly this on 120 synthetic windows it had never seen. | 📊 `f11` ⭐ |
| 70 | And if the evaluation set is GEO-like, **everybody's score will be noise.** We are the team that knew that in advance and said so. | 📊 `f17` ⭐ |

---

## Held back for the appendix (third document)

| # | point |
|---|---|
| 71 | A 244-day, 2-million-row reference dataset built from public archives, and what it confirmed. |
| 72 | The largest offset in real data turns out to be an antenna, not an error — where the signal leaves the spacecraft vs its centre of mass. Matches published values to 4 mm. |
| 73 | A simulated GPS receiver shows most of this "error" is invisible to a real user: anything shared across satellites is absorbed into the receiver's own clock, contributing **exactly zero** position error. |
| 74 | Deep learning at scale — five architectures including a transformer — all lost to the physics model. |
| 75 | We caught ourselves overclaiming and corrected it publicly: an early headline number turned out to be the easiest window in the dataset. |

---

## The 10-minute spine

**Act 1** 1, 5, 8, 9, 11 → **Act 1.5** 14, 15, 16 → **Act 2** 18, 20, 21, 22, 25, 27
→ **Act 3** 32, 36, 38, 39, 41, 42 → **Act 4** 44, 46, 48, 49, 50, 53, 57
→ **Act 5** 60, 61, 62, 63, 64 → **Act 6** 68, 70

28 points, 12 figures. That is `02_PITCH.md`.
