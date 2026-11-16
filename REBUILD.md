# REBUILD — how to regenerate everything that was deleted

Written 2026-09-04, at the point the project reached its conclusion and the
working tree was cleaned. This file exists so that nothing removed in that
cleanup is unrecoverable.

**Read `JOURNEY.md` for what was learned, `TASK_B_FINAL.md` for the deliverable's
numbers. This file is only about reproducing artifacts.**

---

## 0. The one rule

**`raw/` is the only irreplaceable thing in this repository.**

488 files (244 `raw/nav/*_nav.rnx.gz` + 244 `raw/orb/*_orb.sp3.gz`), covering
2026-01-01 → 2026-09-01. Every parquet, npz and CSV in the tree is derived from
it. It is **not backed up anywhere else**, and re-fetching needs NASA CDDIS
credentials — `fetch_all.py` has an explicit `LOGINPAGE` check because CDDIS
serves a login page instead of a 401 when unauthenticated.

Do not delete `raw/`. Do not delete `full2.parquet` unless `raw/` is safe.

---

## 1. Pipeline order

**All of these now live in `Stage-02/` and run from inside it.** Paths are relative
to that directory, so the corpus is `../full2.parquet` and the downloads `../raw`.

```bash
cd Stage-02

# 1 — download (needs CDDIS credentials; ~717 MB, hours)
python fetch_all.py --start 2026-01-01 --end 2026-09-01 --out ../raw

# 2 — broadcast minus precise  ->  ../full2.parquet   (244 days, 2,067,234 rows)
python build_errors.py --raw ../raw --out ../full2.parquet --systems CEGJ --step 900

# 3 — broadcast-only handover features  ->  handover.parquet
python handover.py

# Task A modelling grid  ->  taskA.npz
python taskA_data.py

# Task A feature matrix  ->  features.parquet   (written by nowcast.py)
python nowcast.py

# the model, and the notebook that tells the story
python taskA_v3.py 5 10
python mknb.py && jupyter nbconvert --to notebook --execute --inplace TASK_A.ipynb
```

`windows.py` (also in `Stage-02/`) is a library, not a stage — it is imported for
`sliding_window_view`-backed window access and writes nothing.

---

## 1b. Where things live now

The tree is organised by project stage. Each stage folder is self-contained and its
scripts run from inside it.

| | |
|---|---|
| `Stage-01/` | the delivered PS-08 files and the notebook proving they cannot support the task |
| `Stage-02/` | **the corpus build and all of Task A** — see `Stage-02/README.md` |
| `Stage-03/` | applying the delivered model to the PS-08 files |
| root | the corpus (`full2.parquet`), `raw/`, the Task B deliverable and its benchmarks, `SOLUTION/` |

`raw/` and `full2.parquet` stay at the root because more than one stage reads them.

---

## 2. What was deleted, and how to get it back

| artifact | size | regenerate with | needed by |
|---|---|---|---|
| `features.parquet` | 572 MB | `cd Stage-02 && python nowcast.py` | `latency.py`, `splits.py` |
| `full.parquet` | 207 MB | superseded — **use `full2.parquet`**, see §3 | (repointed) |
| `taskA.npz` | 139 MB | `cd Stage-02 && python taskA_data.py` | all `taskA_*.py`, `validate.py`, `taskA_diagnostics.py` |
| `share/` | 384 MB | `python export_share.py` | teammate handoff, now superseded by `SOLUTION/` |
| `corpus_csv/` | 104 MB | `python export_csv.py` | same |
| `igs20.atx` | 60 MB | download from IGS (see §4) | `antenna_offsets.py` |
| `handover.parquet` | 34 MB | `cd Stage-02 && python handover.py` | `nowcast.py`, `taskA_data.py` |
| `v2_pred_mlp.npz` | 10 MB | `python taskA_v2.py` | `taskA_diagnostics.py` |
| `tab_pred.npz` | 5 MB | `python taskA_tab.py` | `taskA_diagnostics.py` |
| `window_index.npz` | 32 MB | orphan — read by no script, do not rebuild | — |
| `sungeom.parquet` | 37 MB | `python sungeom.py` — hypothesis was falsified | — |
| `runpodctl.exe` | 14 MB | https://github.com/runpod/runpodctl/releases | — |

Deleted logs: none. All `*.log` files were moved to `logs/` and kept — they are
the receipts behind every table in the four markdown documents.

### Task B is unaffected

`taskB_final.py` prefers **`taskB_data.npz` (29 MB)**, which is retained, and only
falls back to `taskA.npz` + `full2.parquet` if it is missing. So the entire Task B
stack — `taskB_bench2.py`, `taskB_nn.py`, `taskB_learn.py`, `geo_fix.py`,
`verify_deliverable.py`, `demo.py`, `calib.py` — runs with nothing regenerated.

---

## 3. `full.parquet` → `full2.parquet`

`full.parquet` was the original 236-day corpus (2026-01-01 → 2026-08-24).
`full2.parquet` is the 244-day rebuild (→ 2026-09-01) made after the SP3 latency
discovery of `JOURNEY.md` §9. They have identical schemas; `full2` is strictly
longer.

At cleanup time these scripts still hardcoded the superseded file and were
repointed to `full2.parquet`:

- `export_csv.py`
- `export_share.py`
- `nowcast.py`
- `antenna_offsets.py` (was `offset_table.py`)
- `sungeom.py`

If any of these are ever restored from an older copy, re-apply the repoint.

---

## 4. Re-fetching `igs20.atx`

The IGS ANTEX file of satellite antenna phase-centre offsets, ~60 MB:

```
https://files.igs.org/pub/station/general/igs20.atx
```

Only two derived products matter and **both are retained in the tree**:

- `antex_offsets.csv` (6 KB) — parsed per-satellite offsets, ionosphere-free
- `satellite_offset_table.csv` (10 KB) — the finished 103-spacecraft table,
  measured vs published, with residuals

So `igs20.atx` only needs re-downloading if the *parsing* is to be redone.
`offset_table_raw.csv` (19 KB), the intermediate join, is also retained.

---

## 5. Known issues carried forward

1. **`satellite_offset_table.csv` has two documented bugs** (`JOURNEY.md` §4):
   BeiDou-2 IGSO (+1.59 m residual) and QZSS-2I (−0.65 m) use the wrong frequency
   convention, and the ANTEX join is done at a single epoch, so the three
   PRN-reuse rows (`C13_a`, `C38_a`, `C39_a`) are matched to the *new* spacecraft.
   Not fixed.

2. **`Data_PS-08/` was missing at cleanup time; it is now in `Stage-01/Data_PS-08/`
   (and again under `Stage-03/data/`).** All 15 findings of
   `Stage-01/WHY_SIH_DATA_FAILS.md` are reproducible again, and
   `Stage-01/WHY_SIH_DATA_FAILS.ipynb` does so.

3. **`smart_horizon.py` exists twice**, at the root and in `SOLUTION/`, currently
   byte-identical (md5 verified at cleanup). There is no git here and no test
   comparing them, so **they can silently diverge** — and one of them is the
   deliverable. `verify_deliverable.py`, `demo.py` and `calib.py` import the root
   copy. Before shipping, always re-check:
   ```bash
   md5sum smart_horizon.py SOLUTION/smart_horizon.py
   ```

4. **PRN reuse.** `C13`, `C38`, `C39` are each two different spacecraft. Any new
   script that groups by PRN must split them into `<PRN>_a` / `<PRN>_b` on orbital
   radius (`> 3.5e7` m = the GSO-era spacecraft). Several scripts already do this
   inline; there is no shared helper.

---

## 6. Merges applied during cleanup

| new file | absorbed |
|---|---|
| `validate.py` | `validate.py` + `mccv.py` — were byte-identical for 183 lines; now `--mode {folds,mccv,all}`. `mccv.py`'s docstring had been a stale copy of `validate.py`'s. |
| `taskA_diagnostics.py` | `persys.py`, `fix_sisre.py`, `tails.py`, `g27.py`, `mixture.py` — now `--section {sisre,persys,tails,g27,mixture,all}` |
| `antenna_offsets.py` | `antex.py`, `offset_table.py`, `build_table.py` — a linear 3-step chain, now `--step {parse,join,table,all}` |

Deleted as superseded drafts (findings fully recorded in `TASK_B_FINAL.md`):
`taskB.py` (v1 ladder), `taskB2.py` (self-validation prototype),
`taskB_bench.py` (first benchmark attempt, replaced by `taskB_bench2.py`),
`taskB_validate.py` (horizon shrinkage — the layer was measured and rejected).
