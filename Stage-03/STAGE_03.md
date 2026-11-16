# Stage 03 — the corpus-derived predictor, run on the actual PS-08 data

**Closing the loop.** Stage-01 diagnosed the delivered files. Stage-02 built a
244-day corpus, took the operational detour (Task A), and came back with a
physics-derived deterministic predictor validated entirely *on the corpus*.
Stage-03 does the thing that was still missing: runs that predictor on the six
CSVs the competition actually supplies, and scores it with the competition's own
metric.

Written 2026-09-04. Every number came off a run in this folder.

---

## 0. The one-paragraph version

Three things came out of this stage, in increasing order of importance.

**The metric is not what we thought, and their scorer is not the standard one.**
`Note.pdf` — issued after the original data, and not covered by Stage-01 — makes
**Shapiro-Wilk W on the residual** the Priority-1 criterion, with RMS nowhere in
the scheme. Their reference benchmark does not reproduce under Shapiro-Wilk. It
reproduces under **Shapiro-Francia**, which identifies the scorer as MATLAB's
`swtest.m`, a routine that *switches test on the sample's own kurtosis* (§2).

**On MEO the predictor works on their data, out of the box.** RMS 0.444 → 0.190
and 0.164 → 0.133 on the two MEO pairs, beating the 7-day window mean on both —
and beating a *cheating* oracle that is handed the test set's own mean (§5). W
improves too, so the two metrics agree here.

**On GEO nothing works, and that is a property of their file, not of the model.**
Predict-zero, window-mean, last-value and SMART-HORIZON all score 14.95–14.96 m,
within 0.02 m of each other. An oracle constant that knows the test mean improves
on doing nothing by **0.3%**. The file carries no predictable structure to find
(§4, §5).

The predictor's response to that corrupt input is the result worth keeping:
handed a window ranging ±40 m, it **predicted values in [−0.4, +1.6] m and
reported σ = 9.5–17.0 m** — it fell back to flat estimators and announced that it
did not know, with an uncertainty of the right order (§7).

---

## 1. There is no trained model to transfer

The framing this stage was requested under was "use the model trained on the
14,000× corpus". That needs correcting before any number below can be read
properly, and the correction is good news.

**`smart_horizon.py` has zero trained parameters.** No weights, no corpus, no
network. The 244 days decided *which basis is correct* — sidereal harmonics for
GEO, Earth-rotation beat frequencies for MEO, level estimators only for the
clock, harmonic order capped, self-validation mandatory — and the coefficients
are re-fitted inside whatever 7-day window the caller passes, on every call. The
only fitted constant in the file is `SIGMA_CAL = 1.25`, and it scales the
reported uncertainty, not the predictions.

So "mapping the corpus model back onto the PS-08 data" is **one function call**.
There is no transfer step, no fine-tuning, and no distribution shift to worry
about, because there is nothing fitted to be shifted.

That is also precisely why this stage is possible at all. Task A — the trained
model — needs live broadcast, SP3 truth and a satellite identity at inference.
The PS-08 files supply five columns and no PRN. **Task A could not be run on this
data at any accuracy.** Stage-02 abandoned it for that reason, and Stage-03 is
the demonstration that the replacement does not have the problem.

---

## 2. The metric changed, and we had to reverse-engineer their scorer

`Note.pdf` sits in `Data_PS-08/` dated after the training files. It supersedes the
original brief in three ways that matter:

1. **Test data is now supplied** for all three pairs — so the model can be scored
   end-to-end on their data, which is what this stage does.
2. **Final evaluation** hands over separate MEO and GEO 7-day training sets and a
   list of arbitrary day-8 timestamps. That is exactly `predict()`'s signature.
3. **Scoring is Shapiro-Wilk W on the residual (predicted − test)**, averaged over
   x/y/z/clock with equal weight — *higher is better* — with mean/sd as
   Priority 2 and a Q-Q plot as Priority 3. **RMS appears nowhere.**

They require teams to write their own scorer and benchmark it against
`SW_ReferenceData.xlsx`, stating the answer as **W = 0.9810, p = 0.5840, H = 0**.

That benchmark does not reproduce:

| test on their 45-value reference | W | p |
|---|---|---|
| stated in Note.pdf | 0.9810 | 0.5840 |
| Shapiro-**Wilk** (scipy / R `shapiro.test` / Royston AS R94) | 0.9852 | 0.8262 |
| Shapiro-**Francia** | **0.9814** | **0.5838** |

The p-value pins it: Shapiro-Francia matches theirs to 2×10⁻⁴. (Their quoted
W = 0.9810 is inconsistent with their own quoted p — that W implies p ≈ 0.567 —
so the W looks like a rounding slip and the p is the reliable anchor.)

That combination identifies the scorer as **MATLAB's `swtest.m`**, which picks the
test from the sample's own kurtosis: **Shapiro-Francia if kurtosis > 3, else
Shapiro-Wilk**. Their reference sample has kurtosis 3.33, so it took the Francia
branch. The sample is also exactly MATLAB `rng('default'); randn(1,45)`.

`sw_score.py` reproduces that switching behaviour, and every W in this document
is computed with it. The `which` column in the outputs records which branch each
column took — because the branch genuinely changes depending on the residual
being graded.

### What the metric actually rewards

**W is location- and scale-invariant.** Multiply every residual by 1000 and W does
not move. Three consequences, all verified in `out/scores.csv`:

- **A predictor wrong by a Gaussian 5 m outscores one wrong by a skewed 5 cm.**
  Accuracy and Priority 1 are different axes.
- **Every constant predictor scores identically.** Predict-zero, window-mean and
  last-value give *the same W to four decimals* on all three pairs, because a
  constant only shifts the residual. Their W is just the normality of the test
  data itself. The metric cannot distinguish them at all.
- **Only a time-varying prediction can move W.** SMART-HORIZON does, on all three
  pairs — which is the whole reason it registers on this metric.

---

## 3. Cleaning: what is dirty, and what is broken

The question this stage was asked to settle — *is the data bullshit, or is it just
unclean?* — has a different answer per channel. `diagnose.py` produces all of it.

### Dirty, and losslessly fixable

**The MEO duplication is a whole-block concatenation, not interleaved repeats.**
Each MEO file contains exactly **one** backward time step, and it spans the entire
file: the block was written twice. So `drop_duplicates().sort_values()` is not a
judgement call, it recovers the file exactly.

| file | rows | unique | one backward step? |
|---|---|---|---|
| `MEO_Train` | 90 | **46** | yes |
| `MEO_Train2` | 244 | **143** | yes |
| `MEO_Test` | 11 | **6** | yes |
| `MEO_Test2` | 30 | **18** | yes |
| `GEO_Train` / `GEO_Test` | 142 / 69 | 142 / 69 | no duplication |

Also cosmetic and fixable: `y_error  (m)` in `MEO_Train` carries a double space
where every other file has one, and the dates are US `m/d/Y` — left to a date
guesser, `9/1` parses as 1 September on some rows and 9 January on others once
the day exceeds 12.

### The test that separates dirty from broken

Broadcast-minus-precise error is smooth in time; at these spacings real error has
lag-1 autocorrelation near 1. But two confounds have to be removed first, and
both of them cut in the direction of a wrong answer:

- **Gap pairs.** Consecutive *rows* are not consecutive *samples* — coverage is
  13–32% and gaps reach 26 h. Mixing a 600 s neighbour with a 6 h jump biases ACF
  toward zero and **makes clean data look like noise**. (This bit us mid-analysis:
  the MEO files first appeared near-white, and only masking to genuinely adjacent
  pairs revealed the position channels are fine.)
- **Static bias.** The corpus carries a large per-satellite offset the SIH files
  do not; a constant inflates ACF toward 1 for free.

So: de-mean both sides, restrict to adjacent pairs, and compare against the
Stage-02 corpus subsampled to *the same spacing*.

| series | dt | x | y | z | clock |
|---|---|---|---|---|---|
| **corpus GEO/GSO** | 900 s | **0.981** | **0.971** | **0.994** | **0.954** |
| `GEO_Train` | 1260 s | **−0.465** | **−0.297** | 0.059 | **−0.325** |
| `GEO_Test` | 900 s | **−0.369** | **−0.311** | **−0.211** | **−0.411** |
| **corpus MEO** | 3600 s | **0.935** | **0.937** | **0.876** | **0.848** |
| `MEO_Train` | 3600 s | 0.798 | 0.951 | 0.943 | **0.016** |
| **corpus MEO** | 600 s | **0.991** | **0.991** | **0.988** | **0.959** |
| `MEO_Train2` | 600 s | 0.373 | 0.852 | 0.907 | 0.535 |

**Verdict, per channel:**

- **MEO position — genuine.** `MEO_Train` at 3600 s (0.798 / 0.951 / 0.943) is
  indistinguishable from real MEO at 3600 s (0.935 / 0.937 / 0.876). The
  sign-independent magnitude check agrees to three decimals: |v| ACF **0.694** on
  `MEO_Train` against **0.694** on the corpus. This is real data.
- **MEO clock — degenerate.** ACF 0.016 against a corpus 0.848. Stage-01 §15 found
  the mechanism: 70 distinct values across 143 rows, piecewise-constant, a step
  process rather than a drift. Not noise — but nothing to extrapolate either.
- **GEO — broken.** Negative ACF on every channel, on adjacent samples, after
  de-meaning, against a corpus that reads +0.95 to +0.99 on the same spacing.

### Is the GEO artefact repairable?

Worth asking properly rather than assuming. If the file were a correct series
emitted with an inconsistent **sign convention**, the signs could be propagated
back into consistency and the series recovered. So: set s₁ = +1 and
sᵢ₊₁ = sᵢ · sign(vᵢ · vᵢ₊₁), then re-measure.

| `GEO_Train` | x | y | z | clock | **‖v‖** |
|---|---|---|---|---|---|
| as delivered | −0.465 | −0.297 | 0.059 | −0.325 | **0.344** |
| sign-repaired | 0.827 | 0.349 | 0.408 | −0.259 | **0.351** |
| real GEO, same spacing | 0.981 | 0.971 | 0.994 | 0.954 | **0.964** |

Repair lifts the position channels substantially — and still lands far short.

**The referee is ‖v‖, the vector magnitude, because no sign flip can change it.**
It sits at **0.344** (Train) and **0.183** (Test) against **0.964** for real GEO.
If a flipping sign convention were the *only* problem, ‖v‖ would already be smooth
and only the signs would need fixing. It isn't.

**So the GEO file is broken, not dirty.** There is no consistent underlying series
for a sign fix to recover — the magnitudes are incoherent independently of sign.

### Outlier treatment — invited by the brief, and rejected on evidence

`SIH_Data_Discription.pdf` says outright: *"The data may contain outliers that
require suitable treatment."* So it was tested rather than assumed
(`outliers.py`). Filter: per-channel MAD on the **training** window only.

On the SIH files it looks like an easy win:

| pair | RMS raw | RMS MAD-5 | W raw | W MAD-5 |
|---|---|---|---|---|
| MEO-1 | 0.190 | **0.137** (−28%) | **0.9067** | 0.8849 |
| MEO-2 | 0.133 | **0.128** | **0.7977** | 0.7626 |
| GEO | 14.962 | 14.935 | 0.7715 | 0.7718 |

Two reasons not to ship it. First, **the criteria disagree**: cleaning improves
RMS and *lowers W on every pair* — and W is what is scored. Trimming the training
tails makes the predictor smoother, which helps accuracy a little and hurts
residual normality a little.

Second, and decisively — that is one window per pair, so it went through the same
resampling control as everything else. On **150 real corpus windows per sampling
pattern**:

| pair | RMS improves in | W improves in |
|---|---|---|
| GEO | 6% of windows | 8% |
| MEO-1 | 6% | 3% |
| MEO-2 | 5% | 4% |

**MAD cleaning helps 3–10% of the time — i.e. it is neutral-to-harmful almost
always, on both criteria.** The SIH gain was a single-window fluke, which is
precisely the trap Stage-02 fell into and documented.

**Decision: no outlier filtering.** The predictor already handles outliers where it
matters — self-validation scores every candidate on a held-out day, so a basis
that chased an outlier loses there and is never selected. And deleting rows up
front destroys real signal: on these files a large value after a long gap is
usually a genuine stale-ephemeris excursion (Stage-01 §10 measures 5.3× fresh to
stale), not a defect.

---

## 4. How much signal is there at all? — the oracle bound

Before asking how a model does, ask what the *ceiling* is. An "oracle constant"
predictor is handed the **test set's own mean** — which no legitimate model can
know — and is therefore an upper bound on every constant predictor.

| pair | RMS predicting zero | RMS of the cheating oracle | oracle gain |
|---|---|---|---|
| **GEO** | 14.956 | 14.911 | **0.3%** |
| MEO-1 | 0.444 | 0.211 | 52.6% |
| MEO-2 | 0.164 | 0.136 | 16.8% |

**On GEO, perfect knowledge of the answer's mean is worth 0.3%.** That single line
is the cleanest statement of why nothing works on that file, and it is
model-independent. It is also Stage-01 §11's ε-identifiability result reappearing
in its most extreme form.

---

## 5. The model on their data

`apply_model.py`. Residual is predicted − test, per Note.pdf.

### RMS (m) — lower is better

| method | GEO | MEO-1 | MEO-2 |
|---|---|---|---|
| zero (do nothing) | 14.956 | 0.444 | 0.164 |
| 7-day window mean | 14.948 | 0.328 | 0.144 |
| last value | 14.964 | 0.427 | 0.174 |
| **SMART-HORIZON** | 14.962 | **0.190** | **0.133** |
| *cheating oracle constant* | *14.911* | *0.211* | *0.136* |

### swtest W — higher is better, and this is Priority 1

| method | GEO | MEO-1 | MEO-2 |
|---|---|---|---|
| any constant predictor | 0.7708 | 0.8840 | 0.7898 |
| **SMART-HORIZON** | **0.7715** | **0.9067** | **0.7977** |

**MEO: it works.** 57% better than doing nothing on MEO-1, 19% on MEO-2, beating
the window mean on both. And on both it **beats the oracle constant** — 0.190 vs
0.211, 0.133 vs 0.136 — which a constant predictor cannot do by construction. That
is the harmonic basis earning its place: it is tracking within-day structure that
no level estimator can represent, on their data, not ours.

W moves the same direction, so on MEO the two metrics agree.

**GEO: the model is neutral, marginally on the wrong side.** 14.962 against 14.956
for doing nothing. Given §4's oracle bound of 0.3%, this is the correct outcome —
there is nothing to find, and the honest report is that we did not find it.

---

## 6. Do these three numbers mean anything? — the resampling control

Three single windows of 69, 6 and 18 test points. Stage-02's hardest lesson was
that a single window can be sold as anything: the figure quoted all session there
turned out to be the easiest fold in the corpus and the honest number was double
it. The bootstrap CI on W at n = 6 is **[0.58, 0.95]**, which says the same thing.

So `replicate.py` takes each pair's **sampling pattern** — train timestamps, test
timestamps, gaps, horizon — and lays it over real corpus data of the same orbit
class at 300 random start epochs, running the identical predictor and scorer.
This is Stage-01's arc-boundary device reused on the model.

| | corpus RMS median [p05, p95] | SIH RMS → pct | corpus W median | SIH W → pct |
|---|---|---|---|---|
| **GEO** | 0.217 [0.124, 0.542] | **14.96 → 100th** | 0.941 | **0.772 → 0th** |
| MEO-1 | 0.316 [0.164, 0.811] | 0.190 → 12th | 0.906 | 0.907 → **50th** |
| MEO-2 | 0.309 [0.161, 0.785] | 0.133 → 1st | 0.883 | 0.798 → 1st |

**GEO is off the scale on both axes simultaneously** — worse than every one of 300
real windows on RMS, and less normal than every one of them on W. It is not a hard
draw from the real distribution; it is not a draw from it at all. This is
Stage-01's Part II confirmed a second time, independently, through the model's
behaviour and the organisers' own metric.

**MEO-1 is dead typical** (W at the 50th percentile) and slightly easy on RMS.
**MEO-2 is an easy window on RMS (1st percentile) with an unusually non-normal
residual (1st percentile on W)** — the Q-Q plot shows why: single outliers in z
and clock, at n = 18.

Also from the same 300-trial runs, and the number to quote for final evaluation:
**the model beats predict-zero in 91–92% of real MEO windows**, and improves W over
predict-zero in 56–79% of them (median ΔW +0.006 to +0.018).

---

## 7. The ceiling on W, and what the GEO failure looks like from inside

### W is noise-dominated at these sample sizes

A residual that is **exactly Gaussian** does not score 1.0. At the n the
competition uses:

| n | median W of a perfect Gaussian sample | 5–95% |
|---|---|---|
| 6 | **0.921** | 0.785 – 0.981 |
| 18 | **0.952** | 0.891 – 0.981 |
| 69 | **0.983** | 0.963 – 0.992 |

So MEO-1's **0.9067 at n = 6 is essentially at the ceiling** — and it rejects
normality on 0 of 4 parameters. There is almost no headroom to compete for. At
n = 6 the 5–95% spread of a *perfect* predictor is [0.785, 0.981], which is wider
than the gap between any two plausible teams. **At small n, Priority 1 is
substantially a lottery**, and that is a property of the organisers' metric, not
of anyone's model.

### The uncertainty channel does its job

Handed the corrupt GEO window — training data ranging **±40 m** — the predictor:

- **predicted values in [−0.4, +1.6] m**, with z and clock exactly flat. It did not
  fit the noise and extrapolate; self-validation found that nothing predicted the
  held-out day and fell back to level estimators.
- **reported σ = 9.5 / 17.0 / 12.4 / 6.8 m**, against actual RMS of 13.8 / 19.7 /
  10.5 / 15.8 m — ratios of 0.85 to 2.34.

It detected that the file was unpredictable and said so, at roughly the right
magnitude. On adversarial input the failure mode is graceful degradation to the
do-nothing baseline plus an honest error bar, not divergence. That is the
self-validation design working exactly as intended, tested on data far outside
anything the corpus contains.

---

## 8. What this means for the submission

1. **Submit the model as-is for MEO.** It beats every constant predictor on their
   own MEO data on both metrics, including an oracle that cheats.
2. **Expect W ≈ 0.88–0.94, not 0.98.** That is what this predictor scores on 300
   real windows at these sample sizes, and the Gaussian ceiling at n = 18 is 0.952.
   A team reporting 0.98 on a real 18-point residual is reporting luck.
3. **Report the Gaussian null next to every W.** Priority 1 is unreadable without
   it, and it is the single most useful thing to put in front of an evaluator.
4. **Say plainly that the GEO file is unpredictable, and show the oracle bound.**
   0.3% from perfect knowledge of the test mean is a stronger argument than any
   model comparison, and it does not depend on trusting our model.
5. **The σ output is worth surfacing** — it is what lets the model say "I don't
   know" on a GEO-like window instead of producing a confident wrong answer.
6. **Priority 1 cannot distinguish constant predictors.** If the evaluation set is
   GEO-like, every team that predicts any constant scores identically. Worth
   knowing before drawing conclusions from a leaderboard.

---

## 9. Files

| file | what it is |
|---|---|
| `sw_score.py` | the organisers' metric, reverse-engineered and calibrated against their reference |
| `sih.py` | loader + the lossless cleaning, and the three train/test pairings |
| `diagnose.py` | dirty-vs-broken per channel, corpus control, GEO sign-repair test, oracle bounds |
| `apply_model.py` | the headline — predictor vs baselines on all three pairs, scored both ways |
| `outliers.py` | the outlier treatment the brief invites, tested and rejected |
| `replicate.py` | SIH sampling patterns replayed on 300 real corpus windows each |
| `report.py` | Priority 1/2/3 outputs: W/p/H, mean/sd, Q-Q plots |
| `smart_horizon.py` | the Stage-02 deliverable, unmodified |
| `data/` | the PS-08 files as delivered, including `Note.pdf` |
| `out/` | scores, predictions, replication draws, Q-Q plots, submission report |

Reproduce: `python sw_score.py && python diagnose.py && python apply_model.py &&
python replicate.py && python outliers.py && python report.py`. Only
`replicate.py`, `outliers.py` and the control block of `diagnose.py` need the
corpus (`../taskB_data.npz`); everything else runs on `data/` alone.

---

## 10. Limitations

1. **One window per pair.** §6 is the mitigation, not a fix — it tells us where the
   single draw sits in a distribution, but the SIH result is still n = 1 per pair.
2. **`MEO_Test` is 6 points.** Every statistic on it, W included, is dominated by
   sampling noise. It should not carry weight either way.
3. **The corpus control snaps timestamps to the 900 s grid.** That preserves
   sampling density, gaps and horizon but loses sub-grid jitter, so it is a control
   on *sampling*, not on off-grid behaviour — Stage-02 §2 measured that separately
   at ~10–20% degradation.
4. **The corpus is 2026 data; PS-08 is September 2025.** Different epoch, and for
   MEO an unknown constellation. The §3 smoothness comparison is like-for-like in
   statistic and spacing, not in date.
5. **We cannot verify the GEO file is not simply a satellite we have never seen.**
   The evidence against it is strong and multiple — negative adjacent-sample ACF,
   ‖v‖ incoherent under sign repair, 0.3% oracle gain, and 0th/100th percentile
   against 300 real windows — but it remains inference from statistics, not a
   statement from the organisers.
6. **Their quoted benchmark W (0.9810) still does not reproduce exactly**, only
   their p-value does (0.5838 vs 0.5840). The algorithm identification rests on
   the p-value and the kurtosis-switching behaviour.
7. **Both organiser PDFs have been read.** `SIH_Data_Discription.pdf` confirms the
   columns are ECEF x/y/z in metres with the clock scaled by c, that irregular
   sampling is intentional, and that outliers "require suitable treatment" (§3).
   It states Shapiro-Wilk as the criterion in prose; `Note.pdf` supersedes it with
   the operational detail — test data, the Priority 1/2/3 scheme, and the
   reference benchmark that identifies the scorer.

8. **One thing in the brief we cannot check.** The description says the clock error
   is converted from seconds via `e × c` with `c ≈ 3×10⁸ m/s`. If they used
   3.0×10⁸ rather than 299792458, every clock value carries a **+0.069%** scale
   error. That is far below the signal here and affects nothing in this stage, but
   it would matter to anyone converting our metres back to seconds.
