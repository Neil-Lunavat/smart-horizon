# Stage 2 — the corpus, and Task A

**Stage 1 ([`../Stage-01`](../Stage-01)) proved the delivered PS-08 files cannot
support the stated task. This stage is what we did instead.**

Two things happen here:

1. **We build a real corpus.** 244 days, 2,067,234 epochs, 103 spacecraft, from NASA
   CDDIS broadcast navigation messages differenced against JAXA final precise orbits.
   The same physical quantity the competition supplies, 14,000x larger, on a uniform
   grid, with every satellite labelled.
2. **Task A.** Reframe the impossible forecast into an answerable nowcast — *given
   the broadcast message live right now, how wrong is it?* — build the model, and
   then test honestly whether the result is any use to an actual GPS receiver.

---

## Start here

**[`TASK_A.ipynb`](TASK_A.ipynb)** is the whole story, start to finish, with every
number computed from the data in this folder. Read that first.
[`TASK_A_AND_WHY_WE_LEFT_IT.md`](TASK_A_AND_WHY_WE_LEFT_IT.md) is the shorter prose
companion.

The notebook needs `../full2.parquet` (the corpus) and the artefacts listed under
"generated" below.

---

## What is in this folder

### Building the corpus

| file | what it does |
|---|---|
| `fetch_all.py` | downloads `BRDM00DLR` nav + `JAX0MGXFIN` precise orbits from CDDIS (needs credentials) |
| `gnss.py` | ICD orbit evaluation, SP3 reader, ECEF→RTN rotation |
| `build_errors.py` | broadcast − precise, per epoch, on a 900 s grid → `../full2.parquet` |
| `windows.py` | memory-safe sliding-window access (a library, not a stage) |
| `handover.py` | the broadcast-only handover-discontinuity feature → `handover.parquet` |
| `export_csv.py`, `export_share.py` | corpus exports for teammates without a parquet toolchain |

### The antenna phase centre

| file | what it does |
|---|---|
| `antenna_offsets.py` | `--step parse\|join\|table` — parses IGS `igs20.atx`, joins it to measured offsets, writes the table |
| `antex_offsets.csv` | published per-satellite offsets, ionosphere-free |
| `offset_table_raw.csv` | the intermediate join |
| `satellite_offset_table.csv` | the finished 103-spacecraft table: measured vs published, with residuals |

### Task A modelling

| file | what it does |
|---|---|
| `taskA_data.py` | builds `taskA.npz`, the dense modelling grid (truth + broadcast-only features) |
| `nowcast.py` | builds `features.parquet`, the flat feature matrix |
| `taskA_tab.py` | baselines + ridge + XGBoost |
| `taskA_nn.py` | MLP / GRU / TCN ladder |
| `taskA_v2.py` | per-system scaling + Huber + baseline blend |
| **`taskA_v3.py`** | **heteroscedastic Gaussian NLL — the model we kept** |
| `taskA_v4.py` | Student-t + PIT calibration |
| `taskA_diagnostics.py` | `--section sisre\|persys\|tails\|g27\|mixture` |
| `validate.py` | `--mode folds\|mccv` — rolling-origin folds and blocked Monte-Carlo CV |
| `splits.py` | random-vs-time split and satellite holdout |
| `latency.py` | the SP3 latency curve |
| `sungeom.py` | sun geometry / beta angle (hypothesis falsified — kept as the record) |

### Notebook plumbing

`mknb.py` assembles `TASK_A.ipynb` from `nbcells_1.py`, `nbcells_2.py`,
`nbcells_3.py`. The prose lives in `.py` files rather than inside notebook JSON so it
can be diffed and edited. To rebuild:

```bash
python mknb.py
jupyter nbconvert --to notebook --execute --inplace TASK_A.ipynb
```

---

## Running it from scratch

Every path in this folder is relative to **this directory**, and the corpus and raw
downloads live one level up.

```bash
cd Stage-02

# 1. the corpus (needs CDDIS credentials; ~717 MB, hours)
python fetch_all.py --start 2026-01-01 --end 2026-09-01 --out ../raw
python build_errors.py --raw ../raw --out ../full2.parquet --systems CEGJ --step 900

# 2. broadcast-only features
python handover.py                    # -> handover.parquet   (34 MB, ~2 min)

# 3. the modelling grid
python taskA_data.py                  # -> taskA.npz          (139 MB, ~3 min)

# 4. the model
python taskA_v3.py 5 10               # -> v3_pred.npz        (~10 min on an RTX 3050)

# 5. the notebook
python mknb.py && jupyter nbconvert --to notebook --execute --inplace TASK_A.ipynb
```

**Generated, not in version control:** `handover.parquet`, `taskA.npz`,
`v3_pred.npz`, `features.parquet`. `../raw/` and `../full2.parquet` are shared with
the other stages and are **not** regenerated casually — see `../REBUILD.md`.

---

## The headline results

**Task A works.** Blocked Monte-Carlo cross-validation, 24 contiguous 2-week test
blocks, retrained from scratch per block:

| | |
|---|---|
| raw broadcast SISRE | 7.127 m |
| 7-day-mean baseline | 0.798 ± 0.461 m |
| **model** | **0.705 ± 0.433 m** |
| vs raw broadcast | **90.1% reduction** |
| vs baseline | **11.7%**, **24/24 folds**, paired t = 8.01 |

**And most of that is not the model.** About 86% of the radial win is the published
antenna phase-centre offset — a table lookup. Much of the clock "error" is the
reference product's per-epoch datum, which a receiver absorbs into its own clock
solution and never sees. The notebook measures both, on a simulated position fix.

**Task A is not submittable.** At evaluation you are handed seven days of error
values and a list of day-8 timestamps. Task A needs the live broadcast message, SP3
truth to *t*−3 days, and the satellite's identity — none of which are supplied. It
cannot be scored, so the deliverable is Task B. Everything learned here transferred;
see the table at the end of the notebook.
