# Task A — the operational reframing, what it achieved, and why we left it

**Companion to `JOURNEY.md` (the full exploration log) and `WHY_SIH_DATA_FAILS.md`
(why the delivered dataset cannot support the stated task).**

This document covers one thing: the detour into Task A. What it was, why we tried
it, what it actually achieved, and the specific reason it could not be submitted.

Written 2026-09-04. Every number is from a run on this machine or the rented GPU.

---

## 1. What Task A was

Having concluded that the delivered PS-08 files cannot support the stated task, we
reframed the problem around what is *operationally* available.

The competition's framing is: **7 days of error in → 8th day of error out.** But
error = broadcast − precise, so computing your own *input* requires the precise
(SP3) product. We initially believed SP3 lagged by two weeks, which would make any
"prediction" arrive a fortnight after the day it predicted. There is no user for
that.

So Task A asked a different question:

> **Given the broadcast message that is live right now, plus whatever truth has
> already been published, how wrong is that message?**

Not a forecast — a **correction**. This is the "broadcast ephemeris improvement" /
SISRE-correction problem from the literature. Concretely, at epoch *t* the model is
given:

* broadcast, continuously, **right up to *t*** — geometry, and the discontinuity
  when one navigation message hands over to the next
* SP3-derived truth only up to ***t* − 3 days**

and must state the error at *t*. The model has to bridge a 3-day gap using
broadcast alone.

**Why this was worth trying:** it is a real system. A live correction stream with no
SP3 dependency at inference is something a receiver could actually consume, whereas
the competition's own framing describes a product nobody can build.

---

## 2. What Task A achieved

These results are real and survived the validation described in §3.

### The headline

Blocked Monte-Carlo cross-validation, 24 random contiguous 2-week test blocks,
retrained from scratch per block, training data strictly before the block with an
8-day gap:

| quantity | value |
|---|---|
| raw broadcast SISRE | 7.127 m |
| 7-day-mean baseline | 0.798 ± 0.461 m |
| **model** | **0.705 ± 0.433 m** (median 0.558) |
| 95% interval | [0.365, 1.787] |
| **vs raw broadcast** | **90.1% reduction** |
| vs baseline | 11.7% |
| **beats baseline in** | **24 / 24 folds** |
| paired t | **8.01** |

Even the worst of 24 folds (1.806 m) is a 75% reduction on raw broadcast.

### The physics finding — the antenna phase centre

The single most useful thing to come out of Task A, and it is not a model at all.

Broadcast ephemeris refers to the satellite's **antenna phase centre**; precise SP3
orbits refer to its **centre of mass**. The difference is a published per-satellite
constant along body-frame Z, which points at nadir — so it lands almost entirely in
*radial* and barely at all in along/cross-track.

We measured the per-satellite offsets, then downloaded IGS `igs20.atx` **afterwards**
and compared:

| block | published z | measured | residual | scatter |
|---|---|---|---|---|
| BLOCK IIF | 1.438 | 1.443 | **−0.004 m** | 0.017 |
| BLOCK IIR-A | 1.163 | 1.150 | +0.012 | 0.011 |
| BLOCK IIR-B | 0.700 | 0.681 | +0.020 | 0.042 |
| BLOCK IIR-M | 0.722 | 0.691 | +0.031 | 0.029 |
| BLOCK IIIA | 1.935 | 1.870 | +0.065 | 0.194 |
| GALILEO-2 | 0.809 | 0.824 | −0.015 | 0.064 |
| BEIDOU-3M-SECM | 1.096 | 1.104 | −0.008 | 0.175 |

**GPS R² = 0.949; Block IIF agrees to 4 mm across 11 satellites**, and the ordering
across all five GPS blocks is reproduced exactly. Nothing was fitted.

Blind-testing every frequency convention also recovered the ICD from data: GPS
matches the **L1/L2 ionosphere-free** combination (0.112 m RMS vs 0.407 for L1
alone), Galileo matches **E5b** (0.072 m). The ANTEX header independently states the
same thing.

This is a standard, documented correction (Montenbruck et al., *Broadcast versus
precise ephemerides: a multi-GNSS perspective*, GPS Solutions 2015). **We
re-derived a published table from data.** That raises confidence in the pipeline and
lowers any claim of novelty — both worth saying plainly.

Output: `satellite_offset_table.csv`, 103 spacecraft.

### The Gaussian-residuals finding

Task A is where we learned why pooled residuals look non-Gaussian when they are not.

| | dR | dA | dC | clock |
|---|---|---|---|---|
| Shapiro W **within one satellite** (median) | **0.9948** | **0.9910** | **0.9943** | **0.9882** |
| Shapiro W **pooled across satellites** | 0.9089 | 0.8169 | 0.7795 | 0.5072 |

Satellite residual scales differ by **11× to 22×**. Pooling Gaussians of different
variance produces a **scale mixture**, which is heavy-tailed by construction. Every
component was normal; the histogram was not.

Modelling the scale explicitly (a heteroscedastic head emitting mean and log-σ)
gave standardised residuals with W = 0.9895 / 0.9888 / **0.9981**, and a Student-t
predictive law with PIT calibration gave normal scores of W = 0.9987 / 0.9989 /
**0.9994**, with dC at **p = 0.226 — failing to reject normality outright**.
Coverage matched nominal at every level (52.9 / 71.4 / 96.5 / 99.7% against
50 / 68.3 / 95.4 / 99.7).

**This finding transferred directly to Task B** and is arguably the most valuable
thing the detour produced.

### Other durable results

* **SP3 final latency is 3 days, not 2 weeks.** Measured by probing CDDIS directly.
  The latency curve showed this is worth more than any architecture change:
  clock error at 14-day latency is 1.022 m, at 3-day 0.818 m, at 1-day 0.555 m.
* **65–78% of the clock signal is a product artifact.** The four constellations'
  common-mode clock series correlate at r = 0.85–0.96. Four independent ground
  segments cannot share a clock signal; this is the analysis centre's reference
  alignment imprinted on every satellite.
* **Ephemeris age carries no signal** (R² = 0.000 everywhere).
* **Along-track is only 6% of SISRE**, so two failed physics hypotheses (sun
  geometry, upload age) were spent chasing a term that barely matters.
* **The handover discontinuity works, but not how expected** — its *7-day rolling
  standard deviation* is the useful feature, not the jump itself. It reads how
  jittery the ground segment has been lately.

---

## 3. The validation reckoning — an error we made and corrected

This belongs in the record.

Everything above was **initially reported on one 8-day test window**, and four model
versions (v1 → v4) were designed while looking at that window's score. That is model
selection on the test set.

When forced to validate properly, the honest number appeared:

| fold | test SISRE | baseline |
|---|---|---|
| fold0 @12000 | 0.455 | 0.501 |
| **fold1 @14400** | **1.595** | 1.655 |
| fold2 @16800 | 0.614 | 0.700 |
| fold3 @19200 | 0.520 | 0.537 |
| **fold4 @21600** | **0.357** | 0.401 |

**The window quoted all session (fold4, 0.357) was the easiest in the corpus.** True
mean 0.708, sd 0.451 — roughly **double** what had been claimed.

Corrections issued at the time:

* "2-week SP3 lag" → actually 3 days
* "88% radial reduction from the model" → 86 of those points are a lookup table
* "SISRE 0.337" → best of 5 folds; the 24-fold answer is 0.705 ± 0.433
* "handover jumps will be useless" → wrong, worth 11 points
* "sequence models will win on along-track" → the GRU lost to a plain MLP

The lesson carried into Task B: **cross-validate before quoting anything**, and
report per-fold values so no single block can carry a number.

---

## 4. Why Task A did not work — as a submission

Task A works fine *as a system*. It fails as a deliverable, for one decisive reason
and two supporting ones.

### 4.1 The decisive reason: it cannot be run on their evaluation set

This is the whole argument.

At final evaluation, participants are handed **7 days of error data (MEO and GEO,
separately) and a list of day-8 timestamps**, and must return predictions. That is
the entire input.

The Task A model requires, at inference time:

* the **live broadcast navigation message** for the satellite in question
* **SP3-derived truth** up to 3 days before the prediction epoch
* the **satellite's identity**, to fetch either of the above

**None of these are supplied, and none can be reconstructed from five columns of
error values with no satellite identifier.**

So the model cannot be scored. Not "would score poorly" — *cannot be evaluated at
all*. A submission consisting only of Task A is a submission that cannot be run.

### 4.2 The margin over a trivial baseline was modest

The 90.1% reduction against raw broadcast is real, but the comparison that matters
for a *model* is against the simple baseline, and there the answer is **11.7%**.

Highly significant — 24/24 folds, t = 8.01 — but a tenth, not a doubling. And it
depends strongly on training-set size: roughly 0–2% on the earliest blocks, 11–23%
on later ones.

Worse, the breakdown is uncomfortable:

| radial correction | RMS |
|---|---|
| raw broadcast − precise | 1.295 m |
| minus published ANTEX offset (a lookup table, no training) | **0.224 m** |
| minus a per-satellite constant from history | 0.184 m |
| full learned model | 0.144 m |

**~86% of the radial win is a table lookup** that a GNSS practitioner would apply
before calling the difference an error at all. A proper SISRE study applies the
ANTEX correction *first*. Our headline 1.295 m "error" was largely an uncorrected
reference-frame difference.

### 4.3 Fold-to-fold variance was large

0.705 ± 0.433, range [0.365, 1.806] — a 5× spread. Performance on any given
fortnight was only loosely predictable, largely because of regime changes like the
BeiDou clock jump in June (10–26 m).

By contrast the Task B predictor came in at **MEO 0.385 ± 0.027** — a 7% spread —
because it has no trained parameters and therefore cannot be lucky or unlucky about
a training period.

---

## 5. Why we returned to Task B

Three reasons, in order of weight.

**1. It is the only thing that can be scored.** Whatever we believe about the
delivered data, the evaluation harness is Task B: five columns in, four columns out,
at timestamps they choose. A model that cannot consume that input cannot compete.

**2. The corpus solves the data problem without changing the task.** The reason
Task B looked hopeless was never the *task* — it was the 143 defective rows. We have
244 days × 100 satellites of correctly sampled, fully labelled data of the identical
physical quantity (`WHY_SIH_DATA_FAILS.md` §10 shows the SIH MEO sawtooth grows 5.3×
from fresh to stale ephemeris and the corpus 4.5×, at the same absolute scale). So
Task B can be trained and validated properly *on the corpus* and then run on their
five columns. That is 14,000× more data for the same problem.

**3. Everything Task A taught us transferred.** The detour was not wasted:

| Task A finding | how it is used in Task B |
|---|---|
| residuals are Gaussian per satellite, scale-mixed when pooled | per-window σ; normality reported per window, not pooled |
| heteroscedastic modelling with calibrated σ | the deliverable returns a 1-σ per prediction |
| clock has no 24 h memory (R² = 0.000) | clock restricted to level estimators; harmonics forbidden |
| GEO is the most predictable class | separate GEO and MEO models |
| orbital periodicity, not daily | sidereal basis for GEO, beat frequencies for MEO |
| **cross-validate before quoting** | blocked CV from the outset, per-fold values always shown |

**Result of the return:** GEO 2.806 → **0.244** mean RMS (91.3%), MEO 2.518 →
**0.397** (84.2%), stable at GEO 0.323 ± 0.058 and MEO 0.385 ± 0.027 across six
independent time blocks — and it runs on exactly the five columns they supply.

---

## 6. Honest summary

**Task A produced a working system and three durable findings** — the antenna
offset (matched to published values to 4 mm), the scale-mixture explanation of
apparent non-normality, and the true 3-day SP3 latency. It reduces SISRE by 90.1%
against raw broadcast with 24/24 folds beating baseline.

**It is not submittable**, because it needs live broadcast, recent SP3 and the
satellite's identity at inference — none of which the evaluation provides.

**It also over-promised before it was validated.** The first reported figure was
double-counted from the easiest fold, and the model's margin over a trivial baseline
turned out to be 11.7% rather than the impression given. That is recorded here
rather than quietly dropped, because the same mistake is the one most likely to be
repeated.

**It belongs in a submission as the "beyond the brief" section**: evidence that the
team understood the operational problem behind the problem statement, and built the
thing that would actually be useful — alongside the Task B model, which is the thing
that can be scored.
