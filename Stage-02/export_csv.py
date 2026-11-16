#!/usr/bin/env python3
"""Export full2.parquet to plain CSV for teammates without a parquet toolchain.

Writes:
  corpus_csv/by_satellite/<SAT>.csv   one file per satellite (~2.8 MB each)
  corpus_csv/all_errors.csv.gz        everything, gzipped
  corpus_csv/satellite_index.csv      one row per satellite: class, span, counts
  corpus_csv/README.md                column dictionary + known issues

Two corrections are applied that the parquet does not carry:
  * PRN reuse: C13/C38/C39 are two different spacecraft each (orbit radius
    changes 42164 km -> 27906 km across a May gap). Split into <PRN>_a / <PRN>_b.
  * Quality flags are added as a column; NO rows are dropped, so anyone can
    reproduce a stricter or looser cut than mine.
"""
import numpy as np, pandas as pd
from pathlib import Path

CL = 299792458.0
OUT = Path("corpus_csv"); (OUT/"by_satellite").mkdir(parents=True, exist_ok=True)
REUSE = {"C13", "C38", "C39"}

d = pd.read_parquet("../full2.parquet")
d["r_km"] = np.sqrt(d.px**2 + d.py**2 + d.pz**2) / 1e3

# --- split reused PRNs into distinct spacecraft -----------------------------
sat = d.sat.to_numpy().astype(object)
isre = d.sat.isin(REUSE).to_numpy()
sat[isre] = np.where(d.r_km.to_numpy()[isre] > 35000,
                     d.sat.to_numpy()[isre] + "_a", d.sat.to_numpy()[isre] + "_b")
d["sat_id"] = sat

# --- orbit class: one label per spacecraft, from its own median radius ------
med = d.groupby("sat_id").r_km.median()
def klass(sid):
    r = med[sid]
    if r < 35000: return "MEO"
    g = d[d.sat_id == sid]
    lon = np.degrees(np.ptp(np.unwrap(np.arctan2(g.py, g.px))))
    lat = np.degrees(np.arcsin((g.pz.abs() / (g.r_km * 1e3)).max()))
    return "GEO" if (lon < 5 and lat < 5) else "GSO"
cls = {s: klass(s) for s in med.index}
d["orbit_class_fixed"] = d.sat_id.map(cls)

# --- quality flags ----------------------------------------------------------
d["clock_error_m"] = d.dClk_ns * 1e-9 * CL
q = np.where(d.clock_error_m.abs() > 100, "clock_gross",
    np.where(d.anom_z > 10, "outlier", "ok"))
d["quality"] = q

cols = ["utc_time","sat","spacecraft","sys","orbit_class",
        "x_error_m","y_error_m","z_error_m","clock_error_m","clock_error_debiased_m",
        "radial_m","along_m","cross_m","sat_x_km","sat_y_km","sat_z_km",
        "eph_age_s","u_arg_rad","arc_id","anom_z","quality"]

def frame(g):
    return pd.DataFrame({
        "utc_time": g.epoch.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "sat": g.sat, "spacecraft": g.sat_id, "sys": g.sys,
        "orbit_class": g.orbit_class_fixed,
        "x_error_m": g.dx.round(4), "y_error_m": g.dy.round(4), "z_error_m": g.dz.round(4),
        "clock_error_m": g.clock_error_m.round(4),
        "clock_error_debiased_m": (g.dClk_debias_ns * 1e-9 * CL).round(4),
        "radial_m": g.dR.round(4), "along_m": g.dA.round(4), "cross_m": g.dC.round(4),
        "sat_x_km": (g.px/1e3).round(3), "sat_y_km": (g.py/1e3).round(3),
        "sat_z_km": (g.pz/1e3).round(3),
        "eph_age_s": g.age_s.round().astype("Int64"),
        "u_arg_rad": g.u_arg.round(6), "arc_id": g.arc_id,
        "anom_z": g.anom_z.round(2), "quality": g.quality})[cols]

d = d.sort_values(["sat_id","epoch"])
rows = []
for sid, g in d.groupby("sat_id", sort=True):
    f = frame(g)
    f.to_csv(OUT/"by_satellite"/f"{sid}.csv", index=False)
    rows.append({"spacecraft": sid, "prn": g.sat.iloc[0], "sys": g.sys.iloc[0],
                 "orbit_class": cls[sid], "n_epochs": len(g),
                 "first_utc": g.epoch.min(), "last_utc": g.epoch.max(),
                 "days_spanned": round((g.epoch.max()-g.epoch.min()).total_seconds()/86400,2),
                 "pct_ok": round((g.quality=="ok").mean()*100,2),
                 "mean_radius_km": round(g.r_km.mean(),1)})
pd.DataFrame(rows).to_csv(OUT/"satellite_index.csv", index=False)
frame(d).to_csv(OUT/"all_errors.csv.gz", index=False, compression="gzip")
print(f"{len(rows)} spacecraft, {len(d):,} rows written to {OUT}/")
