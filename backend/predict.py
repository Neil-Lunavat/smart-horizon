#!/usr/bin/env python3
"""SMART-HORIZON command-line predictor.

Give it seven days of error data, get day 8 back on a 15-minute grid.

    python predict.py seven_days.csv
    python predict.py seven_days.csv --orbit GEO
    python predict.py seven_days.csv --out day8.csv --step 900

INPUT   any CSV with five columns, in any order, with or without units in the
        header:   utc_time, x_error, y_error, z_error, satclockerror
        Duplicates, gaps, unsorted rows and irregular sampling are all fine --
        they are cleaned automatically.

OUTPUT  a CSV covering the 24 hours after the last input timestamp, one row per
        step (default 900 s = 96 rows), with:

          utc_time                the prediction epoch
          x_error .. satclockerror    the predicted error, metres
          *_sigma                 1-sigma uncertainty on each, metres

ORBIT   pass --orbit GEO or --orbit MEO. If omitted the script guesses from the
        data and says so; the guess is reliable but the competition tells you
        the class, so pass it when you know it.

Custom timestamps instead of a grid:

    python predict.py seven_days.csv --at query_times.csv

where `query_times.csv` has a single timestamp column.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import predict, T_SID, _lomb            # noqa: E402
from sih import COLS                               # noqa: E402


def read_any(path):
    """Read a PS-08-style CSV: strip units from headers, parse US dates, clean."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
    ren = {"clock_error": "satclockerror", "clock": "satclockerror",
           "satclock": "satclockerror", "x_err": "x_error",
           "y_err": "y_error", "z_err": "z_error", "time": "utc_time",
           "timestamp": "utc_time", "epoch": "utc_time"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    missing = [c for c in ["utc_time"] + COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {path} is missing columns: {missing}\n"
                         f"       found: {list(df.columns)}")
    t = pd.to_datetime(df["utc_time"], format="%m/%d/%Y %H:%M", errors="coerce")
    if t.isna().any():                              # fall back to general parsing
        t = pd.to_datetime(df["utc_time"], errors="coerce")
    df["utc_time"] = t
    df = df.dropna(subset=["utc_time"])
    n0 = len(df)
    df = df.drop_duplicates().sort_values("utc_time").reset_index(drop=True)
    return df[["utc_time"] + COLS], n0 - len(df)


def guess_orbit(df):
    """GEO if the dominant rhythm is the sidereal day, else MEO.

    A geosynchronous satellite repeats once per sidereal day (86164 s); a MEO
    satellite orbits in 12-14 h. Searching the z channel over the whole plausible
    band and seeing which it lands nearest is a reliable discriminator.
    """
    t = (df.utc_time - df.utc_time.min()).dt.total_seconds().to_numpy(float)
    z = df["z_error"].to_numpy(float)
    P = _lomb(t, z, 30000.0, 100000.0, n=400)
    if P is None:
        return "MEO", None
    return ("GEO" if abs(P - T_SID) / T_SID < 0.12 else "MEO"), P


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Predict day-8 GNSS errors from a 7-day window.")
    ap.add_argument("input", help="CSV with utc_time,x_error,y_error,z_error,satclockerror")
    ap.add_argument("--orbit", choices=["GEO", "MEO", "geo", "meo"], default=None,
                    help="orbit class (the competition supplies this)")
    ap.add_argument("--out", default=None, help="output CSV (default: <input>_day8.csv)")
    ap.add_argument("--step", type=float, default=900.0,
                    help="prediction step in seconds (default 900 = 15 min)")
    ap.add_argument("--hours", type=float, default=24.0,
                    help="how far past the last sample to predict (default 24 h)")
    ap.add_argument("--at", default=None,
                    help="CSV of specific timestamps to predict, instead of a grid")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    df, dropped = read_any(a.input)
    if len(df) < 20:
        raise SystemExit(f"ERROR: need at least 20 distinct rows, got {len(df)}")

    orbit = (a.orbit or "").upper()
    guessed = ""
    if not orbit:
        orbit, P = guess_orbit(df)
        guessed = (f"  (guessed from the data: dominant period "
                   f"{P/3600:.2f} h -> {orbit}; pass --orbit to override)")

    if a.at:
        q = pd.read_csv(a.at)
        tcol = [c for c in q.columns if "time" in str(c).lower()] or [q.columns[0]]
        qt = pd.to_datetime(q[tcol[0]], format="%m/%d/%Y %H:%M", errors="coerce")
        if qt.isna().any():
            qt = pd.to_datetime(q[tcol[0]], errors="coerce")
        qt = qt.dropna()
    else:
        t0 = df.utc_time.max()
        n = int(round(a.hours * 3600.0 / a.step))
        qt = pd.Series([t0 + pd.Timedelta(seconds=a.step * (k + 1)) for k in range(n)])

    res = predict(df, qt, orbit=orbit)

    out = a.out or (os.path.splitext(a.input)[0] + "_day8.csv")
    res.to_csv(out, index=False)

    if not a.quiet:
        span = (df.utc_time.max() - df.utc_time.min()).total_seconds() / 86400.0
        print(f"input   {a.input}")
        print(f"        {len(df)} rows over {span:.2f} days"
              + (f"  ({dropped} duplicate rows removed)" if dropped else ""))
        print(f"        {df.utc_time.min()}  ->  {df.utc_time.max()}")
        print(f"orbit   {orbit}{guessed}")
        print(f"output  {out}")
        print(f"        {len(res)} predictions, "
              f"{res.utc_time.min()}  ->  {res.utc_time.max()}")
        print()
        print(res.head(3).to_string(index=False))
        print("        ...")
        sig = {c: float(res[c + "_sigma"].iloc[0]) for c in COLS}
        print("\nreported 1-sigma (m):  " +
              "   ".join(f"{c.replace('_error','')}={v:.3f}" for c, v in sig.items()))
    return res


if __name__ == "__main__":
    main()
