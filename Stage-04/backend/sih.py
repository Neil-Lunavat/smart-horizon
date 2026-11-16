#!/usr/bin/env python3
"""Loader and cleaner for the PS-08 files, plus the three train/test pairings.

Everything downstream in Stage-03 goes through here so that the cleaning applied
is stated once and identically for every experiment.

WHAT NEEDS CLEANING, AND WHAT MERELY LOOKS LIKE IT DOES
  header drift   `y_error  (m)` in MEO_Train carries two spaces where every other
                 file has one; units are in the header of every column
  duplication    the MEO files are the SAME BLOCK CONCATENATED TWICE, not
                 row-level repeats -- each has exactly ONE backward time step, of
                 the full span.  drop_duplicates + sort is therefore lossless and
                 not a judgement call.  90->46, 244->143, 11->6, 30->18
  ordering       the second block re-sorts cleanly into the first
  date format    US m/d/Y, parsed explicitly rather than left to the dateutil
                 guesser, which would read 9/1 as 1 Sep on some rows and Jan 9 on
                 others once the day exceeds 12

None of the above is a modelling decision -- it is the same file, correctly read.
The defects that CANNOT be cleaned are diagnosed in diagnose.py.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
COLS = ["x_error", "y_error", "z_error", "satclockerror"]

# (name, train file, test file, orbit class as the competition supplies it)
PAIRS = [
    ("GEO",   "DATA_GEO_Train.csv",   "DATA_GEO_Test.csv",   "GEO"),
    ("MEO-1", "DATA_MEO_Train.csv",   "DATA_MEO_Test.csv",   "MEO"),
    ("MEO-2", "DATA_MEO_Train2.csv",  "DATA_MEO_Test2.csv",  "MEO"),
]


def load(name, clean=True):
    """Read one PS-08 CSV. `clean=False` returns it exactly as delivered."""
    df = pd.read_csv(os.path.join(DATA, name))
    df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
    df["utc_time"] = pd.to_datetime(df["utc_time"], format="%m/%d/%Y %H:%M")
    if clean:
        df = df.drop_duplicates().sort_values("utc_time").reset_index(drop=True)
    return df[["utc_time"] + COLS]


def load_pair(key, clean=True):
    """Return (train_df, test_df, orbit) for one of GEO / MEO-1 / MEO-2."""
    rec = {p[0]: p for p in PAIRS}[key]
    return load(rec[1], clean), load(rec[2], clean), rec[3]


def dup_report():
    """Evidence that the duplication is a block concatenation, not noise."""
    rows = []
    for f in sorted(os.listdir(DATA)):
        if not f.startswith("DATA_") or not f.endswith(".csv"):
            continue
        raw = load(f, clean=False)
        cl = load(f, clean=True)
        dt = np.diff(raw["utc_time"].values).astype("timedelta64[s]").astype(float)
        back = int((dt < 0).sum())
        span = (raw.utc_time.max() - raw.utc_time.min()).total_seconds()
        # a single backward step of the full span == the file is one block twice
        block = back == 1 and abs(dt.min()) > 0.9 * span
        rows.append({"file": f[5:-4], "rows": len(raw), "unique": len(cl),
                     "dropped": len(raw) - len(cl),
                     "backward_steps": back, "block_concat": block})
    return pd.DataFrame(rows)


def spacing(df):
    """Modal sample spacing and the mask of genuinely adjacent pairs.

    Consecutive ROWS are not consecutive SAMPLES in these files -- coverage is
    13-32% and the gaps run to 26 h. Any lag-1 statistic computed over all row
    pairs mixes 600 s neighbours with multi-hour jumps and is biased toward zero,
    which will make clean data look like noise. Every lag-1 number in Stage-03
    uses this mask.
    """
    dt = np.diff(df["utc_time"].values).astype("timedelta64[s]").astype(float)
    near = dt[(dt > 0) & (dt <= 7200)]
    md = float(np.median(near)) if len(near) else float(np.median(dt))
    return md, np.abs(dt - md) <= 0.25 * md


if __name__ == "__main__":
    print(dup_report().to_string(index=False))
    print()
    for k, *_ in PAIRS:
        tr, te, orb = load_pair(k)
        md, m = spacing(tr)
        print(f"{k:6s} orbit={orb:3s}  train n={len(tr):3d} "
              f"({tr.utc_time.min():%m-%d %H:%M} -> {tr.utc_time.max():%m-%d %H:%M}, "
              f"{(tr.utc_time.max()-tr.utc_time.min()).total_seconds()/86400:.2f} d, "
              f"dt~{md:.0f}s)   test n={len(te):3d} "
              f"({te.utc_time.min():%m-%d %H:%M} -> {te.utc_time.max():%m-%d %H:%M})")
