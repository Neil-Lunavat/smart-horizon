# Stage 2 — what we actually found

**Plain English. The conclusions, not the method.**
The method, the code and the plots are in [`Stage-02/TASK_A.ipynb`](Stage-02/TASK_A.ipynb).
Every number below is computed there and reproduces on a laptop.

---

## What we set out to do

Stage 1 proved the delivered PS-08 files were unusable. So we built our own dataset —
244 days, 2,067,234 epochs, 100 satellites, downloaded from NASA — and tried to build
a model that corrects GPS broadcast ephemeris error.

**The short version of what happened: we succeeded, and then found out the thing we
were correcting was mostly not an error.**

---

## Finding 1 — the 7 metres was mostly not real

Raw broadcast-minus-truth is about **7.1 m**. That number is nearly meaningless, and
it breaks into four pieces:

| piece | size | is it a real error? |
|---|---|---|
| the antenna offset | ~1.1 m | **No** — a reference-frame difference |
| the reference file's clock datum | several m | **No** — a bookkeeping convention |
| a fixed bias per satellite | large | Partly — removable by averaging |
| genuinely time-varying error | **~0.8 m** | **Yes. This is the real one.** |

Everything interesting is in the last row. The first two rows are the reason the
headline number looks big, and neither of them is something a GPS user experiences.

---

## Finding 2 — the antenna offset (and we did not discover it)

Broadcast ephemeris tells you where the satellite's **antenna** is. Precise SP3
orbits tell you where its **centre of mass** is. Two different points, about a metre
apart along the Earth-facing axis. Nobody is wrong — they are answering different
questions.

We measured the offset from data, then downloaded the published IGS table and
compared:

* per-satellite median: **radial −1.101 m, along-track −0.021 m, cross-track −0.002 m**
  — almost purely radial, exactly as predicted
* GPS measured vs published: **R² = 0.950**
* Block IIF agrees to **11 mm across 11 satellites**

**This is a standard correction, documented since Montenbruck et al. 2015.** We did
not find it. We wrote our own comparison from scratch, skipped step one of the
standard recipe, got a big number, and correctly diagnosed our own omission from the
data. That is a good sign about the method and not a discovery, and it should be
stated that way.

**It does not matter to a receiver.** Broadcast gives the antenna position; the
receiver wants the antenna position. There is nothing to correct. The offset only
appears when you compare broadcast against SP3 — which analysts do and receivers
never do.

---

## Finding 3 — most of the "clock error" belongs to the reference file

Split each satellite's clock error into a part shared with its neighbours and a part
of its own. The shared part explains **85.2% (GPS), 85.5% (Galileo), 68.5% (BeiDou)**
of each satellite's variance. Run the same arithmetic on the orbit components and it
explains between −0.4% and 5.5% — i.e. nothing.

Then the test that settles it. Correlate the four constellations' shared components
against each other:

```
           GPS  Galileo  BeiDou   QZSS
GPS      1.000    0.961   0.913   0.921
Galileo  0.961    1.000   0.909   0.889
BeiDou   0.913    0.909   1.000   0.851
QZSS     0.921    0.889   0.851   1.000
```

**Four countries, four independent ground segments, four independent atomic clock
ensembles, correlating at 0.85–0.96.** There is no physical mechanism for that.

It comes from the other side of the subtraction. An SP3 clock is only defined up to a
per-epoch reference the analysis centre picks arbitrarily, and JAXA puts all four
constellations in one file with one reference. When it moves, every satellite's
"error" moves with it identically.

**So a large fraction of what we were calling satellite clock error is the reference
product's bookkeeping.**

---

## Finding 4 — a receiver never sees most of it

We built a single-point-positioning simulator: a receiver at a known location, real
satellite geometry, real broadcast errors injected as range errors, solving
least-squares for position **and its own clock**, with one clock unknown per
constellation like a real receiver.

Three controls:

| injected error | horizontal | vertical |
|---|---|---|
| a constant 10 m on **every** satellite | **0.000 m** | **0.000 m** |
| a 10 m **radial** offset on every satellite | 0.014 m | 0.354 m |
| white 1 m, independent per satellite | 1.001 m | 1.672 m |

**Row 1 is the key result.** A receiver solves for its own clock, so anything common
to all satellites is absorbed exactly. Not approximately — zero.

**Row 2 answers the "surely the antennas break trilateration" objection.** They do
not, because every navigation satellite antenna points at the Earth. All the offsets
point the same way — down — so most of the effect is a shared range bias and gets
absorbed too. What leaks is small, and it leaks into height, because elevation angle
is what determines height.

**Row 3 is what actually matters.** Only errors that differ *between* satellites
survive, and those come through roughly one-for-one.

---

## Finding 5 — the residual is real, but it is not predictable

This distinction is the centre of everything and it is easy to blur.

After removing the antenna offset and the datum artefact, what remains is
`predicted − true` for the same point on the spacecraft. **That is a genuine error.**
About **0.8 m of range error, roughly 1.3 m of 3D position error.** A GPS user really
does eat it.

It is real. It is just not **forecastable** from what we are holding, for three
reasons:

1. **It is somebody else's forecast error.** Broadcast is the ground segment
   extrapolating an orbit forward. Their error is driven mostly by solar radiation
   pressure — sunlight pushing on the spacecraft, depending on its attitude and
   thermal state. None of that is in the broadcast message. We tested the hypothesis
   directly and it came back flat: **R² = 0.008**.
2. **Clock noise is a random walk.** The clock's correlation with itself 24 hours
   later is **0.006**. If there is no correlation at a lag, there is no information to
   predict with. That is arithmetic, not a modelling shortfall.
3. **Uploads are on a schedule we do not have.** The error resets to near zero
   whenever the ground segment uploads a new message. That is an operations calendar,
   not physics.

**Five architectures — MLP, GRU, LSTM, TCN, transformer — converged to the same
error.** That is not five failures. It is five independent confirmations that there
is nothing left in there.

---

## Finding 6 — what the model is actually worth

Blocked cross-validation, 24 contiguous two-week test blocks, retrained from scratch
each time:

| | |
|---|---|
| raw broadcast | 7.127 m |
| **7-day average baseline** | **0.798 ± 0.471 m** |
| our model | 0.705 ± 0.442 m |
| vs raw | 90.1% |
| **vs the baseline that actually matters** | **12.7%**, 24/24 folds |

And on a simulated position fix, GPS only:

| | 3D position error |
|---|---|
| broadcast, uncorrected | 1.355 m |
| + 7-day average | 0.694 m |
| + our model | **0.622 m** |

**Read that honestly.** Halving the position error is real, and almost all of it
comes from subtracting a per-satellite constant — no learning involved. The neural
network adds about 10% on top of that. That is a true number and a small one.

And in a phone it would be invisible anyway: the ionosphere alone is several metres
and urban multipath can be worse. Our correction is worth a few tenths.

---

## Finding 7 — the one lever that actually works is data freshness

The model anchors on precise orbits that arrive late. We assumed two weeks; we
measured it and the product we use is **3 days** old on arrival.

Then we varied how stale the truth was allowed to be:

| truth age | orbit error | clock error |
|---|---|---|
| 6 hours | 0.103 m | **0.473 m** |
| 3 days (what we use) | 0.105 m | **0.818 m** |
| 14 days | 0.124 m | 1.022 m |

The orbit barely cares — orbits are smooth. **The clock nearly doubles**, because the
clock has no memory past a few hours.

**Fresher data is worth about 40%. The neural network is worth about 12%.** Same
model, same features — just less stale input. The bottleneck was never the algorithm.

*(Caveat: that 6-hour figure is a simulation. We restricted what the model could see
in historical data rather than downloading an actual ultra-rapid product.)*

---

## Finding 8 — the residuals were Gaussian the whole time

Worth recording because it is the direct answer to a Shapiro-Wilk-scored problem, and
because it is a trap anyone repeating this work will fall into.

Pooled across satellites, the residuals look badly non-Gaussian: **W = 0.897**, long
tails. The cause is not the model. Satellites differ in error *scale* by **14x**, and
**pooling Gaussians of different variance produces a heavy-tailed mixture by
construction**. Standardise each satellite by its own spread and pool again:
**W = 0.9994**.

Every component was normal. The histogram was not. Test per satellite, or standardise
before pooling, or you will report catastrophic non-normality no matter how good your
model is.

---

## Where this lands

**The error is real and worth fixing. It is not forecastable from the broadcast
signal alone. It is, however, perfectly observable if you have receivers on the
ground watching right now.**

And that is exactly what the industry does. IGS runs a real-time service. SBAS
(WAAS, EGNOS), Galileo HAS and QZSS CLAS all do the same thing: a few hundred ground
stations measure how wrong the broadcast is *at this moment* and stream the
correction out with seconds of latency. No forecasting. No model. Measurement plus
distribution.

They get centimetres. We got 0.6 m.

**So the problem is not doomed — it is already solved, by a method we did not have
the inputs for. Our route to it was what was doomed: trying to squeeze the answer out
of the broadcast signal alone, three days behind, with a model.**

Stage 2's real contribution is having measured precisely how far you can get without
a ground network. The answer is about 0.7 m, and then you stop.

---

## What we own, and what we do not

**We own:**

* A 244-day, 100-satellite dataset built from primary sources that did not exist before.
* A correct, reproducible account of what broadcast error is made of — and how little
  of it is error.
* A receiver simulation that proves the large numbers are invisible to users, with
  controls that behave exactly as physics requires.
* A measured ceiling on what machine learning can add here: ~12%, and that is the
  limit of the information, not our failure to reach it.
* The scale-mixture explanation of apparent non-normality, which transfers to anything
  else pooling errors across heterogeneous sources.

**We do not own:**

* Any improvement to GPS that a normal user would notice.
* The antenna phase centre correction. That is a 2015 paper and standard practice.
* A model anyone should be impressed by.

---

## What not to do next

**Stop modelling.** Five architectures already established there is nothing left. Any
further time on the model is wasted.

The only experiment left with a real payoff is to **replace the 3-day-old truth with
an actual real-time or ultra-rapid product** and re-run. That turns "fresher data
would be worth 40%" from a simulation into a measurement, and it is the one direction
where the numbers still move.
