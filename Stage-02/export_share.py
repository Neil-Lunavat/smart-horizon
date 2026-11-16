#!/usr/bin/env python3
"""Export the two SOURCES of the corpus as plain CSV, joinable on (sat, utc_time).

    share/sp3.csv        where the satellite really was   (JAXA precise, truth)
    share/broadcast.csv  where the nav message SAID it was (what receivers used)

The parquet only stores the DIFFERENCE of these two, so the SP3 clock has to be
re-read from the raw files. Positions are then reconstructed exactly:
    broadcast = precise + error        (verified: max |px_parquet - px_sp3| = 0.0 m)

Both files carry the identical set of (sat, utc_time) keys, so:
    b = pd.read_csv("broadcast.csv"); s = pd.read_csv("sp3.csv")
    df = b.merge(s, on=["utc_time","sat"], suffixes=("_bc","_sp3"))
is a clean 1:1 join with no missing rows on either side.
"""
import glob, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from gnss import read_sp3

OUT = Path("share"); OUT.mkdir(exist_ok=True)
REUSE = {"C13", "C38", "C39"}

def _one(p):
    """One SP3 file, thinned to the 15-min grid and trimmed to its OWN day.

    SP3 spans 00:00 THROUGH 24:00, so consecutive files share a midnight epoch
    and disagree there by ~0.2 m (the seam between 3-day long-arc solutions).
    build_errors.py drops the foreign day; we must do the same or the join
    doubles up at every midnight."""
    day = Path(p).name.split("_")[0]
    s = read_sp3(p)
    keep = (((s.epoch.dt.minute * 60 + s.epoch.dt.second) % 900 == 0)
            & (s.epoch.dt.strftime("%Y%m%d") == day))
    return s[keep]

if __name__ == "__main__":
    files = sorted(glob.glob("../raw/orb/*_orb.sp3.gz"))
    print(f"reading {len(files)} SP3 files ...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        sp3 = pd.concat(ex.map(_one, files, chunksize=4), ignore_index=True)
    print(f"  SP3 rows on the 15-min grid: {len(sp3):,}", flush=True)

    pq = pd.read_parquet("../full2.parquet", columns=["epoch","sat","sys","orbit_class",
                          "dx","dy","dz","dClk_ns","age_s","u_arg","arc_id","px"])
    m = pq.merge(sp3, on=["epoch","sat"], how="inner", suffixes=("_pq",""))
    assert np.abs(m.px_pq - m.px).max() == 0.0, "SP3 positions disagree with parquet"
    print(f"  joined rows: {len(m):,}  (parquet had {len(pq):,})", flush=True)

    # PRN reuse: C13/C38/C39 are two different spacecraft each
    r_km = np.sqrt(m.px**2 + m.py**2 + m.pz**2) / 1e3
    sid = m.sat.to_numpy().astype(object)
    isre = m.sat.isin(REUSE).to_numpy()
    sid[isre] = np.where(r_km.to_numpy()[isre] > 35000,
                         m.sat.to_numpy()[isre] + "_a", m.sat.to_numpy()[isre] + "_b")
    m["spacecraft"] = sid
    m = m.sort_values(["sat","epoch"]).reset_index(drop=True)

    t   = m.epoch.dt.strftime("%Y-%m-%d %H:%M:%S")
    key = dict(utc_time=t, sat=m.sat, spacecraft=m.spacecraft, sys=m.sys,
               orbit_class=m.orbit_class)

    pd.DataFrame({**key,
        "x_m": m.px.round(3), "y_m": m.py.round(3), "z_m": m.pz.round(3),
        "clock_us": (m.pclk * 1e6).round(9),
    }).to_csv(OUT/"sp3.csv", index=False)
    print("  wrote share/sp3.csv", flush=True)

    pd.DataFrame({**key,
        "x_m": (m.px + m.dx).round(3), "y_m": (m.py + m.dy).round(3),
        "z_m": (m.pz + m.dz).round(3),
        "clock_us": (m.pclk * 1e6 + m.dClk_ns * 1e-3).round(9),
        "eph_age_s": m.age_s.round().astype("Int64"),
        "u_arg_rad": m.u_arg.round(6), "arc_id": m.arc_id,
    }).to_csv(OUT/"broadcast.csv", index=False)
    print("  wrote share/broadcast.csv", flush=True)
