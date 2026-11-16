#!/usr/bin/env python3
"""The antenna phase-centre offset -- the project's one piece of real physics.

Previously three scripts (antex.py, offset_table.py, build_table.py) forming a
strictly linear chain. They are one file now because the middle step is useless
without the first and the last is useless without the middle.

WHY IT MATTERS
Broadcast ephemeris refers to the satellite's ANTENNA PHASE CENTRE. Precise SP3
orbits refer to its CENTRE OF MASS. The difference is a published per-satellite
constant along body-frame Z, which points at nadir -- so it lands almost entirely
in RADIAL and barely at all in along/cross-track. It is not an error at all; a
GNSS practitioner applies it before calling the difference an error.

Measured against IGS published values: GPS R^2 = 0.949, and Block IIF agrees to
4 mm across 11 satellites. Nothing was fitted. This re-derives a documented table
(Montenbruck et al., GPS Solutions 2015) from data -- which should raise
confidence in the pipeline and lower any claim of novelty.

Radial correction ladder, from JOURNEY.md section 4:
    raw broadcast - precise                          1.295 m
    minus published ANTEX offset (a lookup table)    0.224 m   <- this file
    minus per-satellite median from training         0.184 m
    full XGBoost model                               0.144 m
~86% of the radial win is the table lookup.

STEPS
  parse   read igs20.atx            -> antex_offsets.csv
  join    measure + join to ANTEX   -> offset_table_raw.csv
  table   final per-spacecraft      -> satellite_offset_table.csv

Usage: python antenna_offsets.py [parse|join|table|all]

Needs igs20.atx (steps parse, join) and full2.parquet (step join). Both are
described in REBUILD.md. Step `table` needs only offset_table_raw.csv, which is
retained in the tree, so the final table can be rebuilt with neither.

KNOWN BUGS in the output, unfixed and documented in JOURNEY.md section 4:
  * BeiDou-2 IGSO (+1.59 m residual) and QZSS-2I (-0.65 m) use the wrong
    frequency convention.
  * The ANTEX join is done at a SINGLE epoch, so the three PRN-reuse rows
    (C13_a, C38_a, C39_a) are matched to the NEW spacecraft, not the old one.
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

CL = 299792458.0
EPOCH = dt.datetime(2026, 5, 1)                    # middle of the corpus
CORPUS = "../full2.parquet"                           # was full.parquet (236 d); see REBUILD.md
REUSE = ["C13", "C38", "C39"]                      # one PRN, two spacecraft each

# ionosphere-free pair per system: (freq code 1, freq code 2, MHz1, MHz2)
IF = {"G": ("G01", "G02", 1575.42, 1227.60), "J": ("J01", "J02", 1575.42, 1227.60),
      "E": ("E01", "E05", 1575.42, 1176.45), "C": ("C02", "C06", 1561.098, 1268.52)}

STEPS = ["parse", "join", "table"]
WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
if WHICH not in STEPS + ["all"]:
    raise SystemExit(f"unknown step {WHICH!r}; expected {', '.join(STEPS)} or all")


def ifc(z1, z2, f1, f2):
    """Ionosphere-free combination of two per-frequency offsets."""
    k = f1 ** 2 / (f1 ** 2 - f2 ** 2)
    return k * z1 - (k - 1) * z2


def parse(path):
    """Read an ANTEX file into per-satellite records. Offsets are body-frame mm."""
    recs = []
    cur = None
    freq = None
    for ln in open(path, errors="ignore"):
        tag = ln[60:].strip()
        if tag == "START OF ANTENNA":
            cur = {"f": {}, "vf": None, "vu": None}
        elif tag == "TYPE / SERIAL NO" and cur is not None:
            cur["type"] = ln[0:20].strip()
            cur["prn"] = ln[20:40].strip()[:3]
            cur["svn"] = ln[40:50].strip()
        elif tag == "VALID FROM" and cur is not None:
            p = ln.split()
            cur["vf"] = dt.datetime(int(p[0]), int(p[1]), int(p[2]),
                                    int(p[3]), int(p[4]), int(float(p[5])))
        elif tag == "VALID UNTIL" and cur is not None:
            p = ln.split()
            cur["vu"] = dt.datetime(int(p[0]), int(p[1]), int(p[2]),
                                    int(p[3]), int(p[4]), int(float(p[5])))
        elif tag == "START OF FREQUENCY" and cur is not None:
            freq = ln[3:6].strip()
        elif tag == "NORTH / EAST / UP" and cur is not None and freq:
            v = [float(x) for x in ln[:60].split()]
            cur["f"][freq] = v                     # body-frame X, Y, Z in mm
        elif tag == "END OF ANTENNA" and cur is not None:
            if cur.get("prn", "")[:1] in "GECJ" and cur["prn"][1:].isdigit():
                recs.append(cur)
            cur = None
            freq = None
    return recs


def _valid_at_epoch(recs):
    """Only the ANTEX blocks whose validity span covers our corpus midpoint."""
    for a in recs:
        vf = a["vf"] or dt.datetime(1980, 1, 1)
        vu = a["vu"] or dt.datetime(2100, 1, 1)
        if vf <= EPOCH <= vu:
            yield a


# ------------------------------------------------------------------ 1. parse
def step_parse():
    print(f"\n{'#'*78}\n# PARSE  igs20.atx -> antex_offsets.csv\n{'#'*78}")
    rows = []
    for r in _valid_at_epoch(parse("igs20.atx")):
        s = r["prn"][0]
        if s not in IF:
            continue
        a, b, f1, f2 = IF[s]
        if a not in r["f"] or b not in r["f"]:
            continue
        z1, z2 = r["f"][a][2], r["f"][b][2]
        rows.append(dict(prn=r["prn"], svn=r["svn"], block=r["type"],
                         z1_m=z1 / 1e3, z2_m=z2 / 1e3,
                         z_if_m=ifc(z1, z2, f1, f2) / 1e3))
    A = pd.DataFrame(rows).drop_duplicates("prn")
    A.to_csv("antex_offsets.csv", index=False)
    print(f"{len(A)} satellites valid at {EPOCH:%Y-%m-%d}")
    print(A.groupby("block").agg(n=("prn", "size"), z_if_mean=("z_if_m", "mean"),
                                 z_if_sd=("z_if_m", "std")).round(3).to_string())


# ------------------------------------------------------------------- 2. join
def step_join():
    print(f"\n{'#'*78}\n# JOIN  measured offsets x ANTEX -> offset_table_raw.csv\n{'#'*78}")
    d = pd.read_parquet(CORPUS, columns=["sat", "sys", "orbit_class", "dR", "dA", "dC",
                                         "dClk_ns", "px", "py", "pz", "anom_z"])
    d["dClk_m"] = d.dClk_ns * 1e-9 * CL
    d = d[(d.anom_z <= 10) & (d.dClk_m.abs() < 100)]
    # PRN reuse -> distinct spacecraft, split on orbital radius
    r = np.sqrt(d.px ** 2 + d.py ** 2 + d.pz ** 2)
    sid = d.sat.to_numpy().astype(object)
    m = d.sat.isin(REUSE).to_numpy()
    sid[m] = np.where(r.to_numpy()[m] > 3.5e7,
                      d.sat.to_numpy()[m] + "_a", d.sat.to_numpy()[m] + "_b")
    d["scid"] = sid
    d["prn"] = d.sat
    M = d.groupby("scid").agg(
        prn=("prn", "first"), sys=("sys", "first"), cls=("orbit_class", "first"),
        n=("dR", "size"), dR=("dR", "median"), dA=("dA", "median"), dC=("dC", "median"),
        clk=("dClk_m", "median"), dR_iqr=("dR", lambda s: s.quantile(.75) - s.quantile(.25)))
    M = M[M.n > 3000].reset_index()

    rows = []
    for a in _valid_at_epoch(parse("igs20.atx")):
        if a["prn"][:1] not in "GECJ":
            continue
        rows.append(dict(prn=a["prn"], block=a["type"], svn=a["svn"],
                         **{f"z_{k}": v[2] / 1e3 for k, v in a["f"].items()}))
    A = pd.DataFrame(rows).drop_duplicates("prn")
    T = M.merge(A, on="prn", how="left")
    print(f"matched {T.block.notna().sum()}/{len(T)} spacecraft to ANTEX\n")

    # which frequency convention does each system's broadcast actually use?
    CAND = {"G": {"L1": "z_G01", "L2": "z_G02",
                  "IF(L1,L2)": ("z_G01", "z_G02", 1575.42, 1227.60)},
            "J": {"L1": "z_J01", "L2": "z_J02",
                  "IF(L1,L2)": ("z_J01", "z_J02", 1575.42, 1227.60)},
            "E": {"E1": "z_E01", "E5a": "z_E05", "E5b": "z_E07",
                  "IF(E1,E5a)": ("z_E01", "z_E05", 1575.42, 1176.45),
                  "IF(E1,E5b)": ("z_E01", "z_E07", 1575.42, 1207.14)},
            "C": {"B1I": "z_C02", "B3I": "z_C06", "B1C": "z_C01",
                  "IF(B1I,B3I)": ("z_C02", "z_C06", 1561.098, 1268.52),
                  "IF(B1C,B2a)": ("z_C01", "z_C05", 1575.42, 1176.45)}}
    print("Which antenna reference does each system's broadcast use?")
    print("  (RMS of  measured_|dR|  -  published_z, over that system's satellites)\n")
    for s, cands in CAND.items():
        sub = T[(T.sys == s) & T.block.notna()]
        if len(sub) < 2:
            continue
        out = []
        for lab, spec in cands.items():
            if isinstance(spec, tuple):
                a, b, f1, f2 = spec
                if a not in sub or b not in sub:
                    continue
                z = ifc(sub[a], sub[b], f1, f2)
            else:
                if spec not in sub:
                    continue
                z = sub[spec]
            e = (-sub.dR) - z
            if z.notna().sum() < 2:
                continue
            out.append((lab, np.sqrt(np.nanmean(e ** 2)), np.nanmean(e)))
        out.sort(key=lambda x: x[1])
        print(f"  {s}: " + "   ".join(f"{l}: rms={v:.3f} bias={b:+.3f}" for l, v, b in out))
        print(f"     -> best: {out[0][0]}")
    T.to_csv("offset_table_raw.csv", index=False)
    print("\nwrote offset_table_raw.csv")


# ------------------------------------------------------------------ 3. table
def step_table():
    print(f"\n{'#'*78}\n# TABLE  -> satellite_offset_table.csv\n{'#'*78}")
    T = pd.read_csv("offset_table_raw.csv")
    # reference convention chosen empirically per system in step_join
    z = pd.Series(np.nan, index=T.index)
    g = T.sys == "G"
    z[g] = ifc(T.z_G01[g], T.z_G02[g], 1575.42, 1227.60)
    e = T.sys == "E"
    z[e] = T.z_E07[e]
    c = T.sys == "C"
    z[c] = T.z_C01[c].fillna(T.z_C06[c])
    j = T.sys == "J"
    z[j] = T.z_J01[j]
    T["apc_z_m"] = z
    T["radial_pred_m"] = -T.apc_z_m                # broadcast APC sits below CoM
    T["resid_m"] = T.dR - T.radial_pred_m
    ok = T.apc_z_m.notna()
    print("HOW MUCH OF THE PER-SATELLITE RADIAL OFFSET IS EXPLAINED BY THE ANTENNA?\n")
    print(f"{'group':12s} {'n':>3s} {'sd(measured)':>13s} {'sd(residual)':>13s} "
          f"{'R2':>7s} {'bias':>7s}")
    for k, h in T[ok].groupby(T.sys + " " + T.cls):
        if len(h) < 2:
            continue
        print(f"{k:12s} {len(h):3d} {h.dR.std():13.3f} {h.resid_m.std():13.3f} "
              f"{1-h.resid_m.var()/h.dR.var():7.3f} {h.resid_m.mean():+7.3f}")
    a = T[ok]
    print(f"{'ALL':12s} {len(a):3d} {a.dR.std():13.3f} {a.resid_m.std():13.3f} "
          f"{1-a.resid_m.var()/a.dR.var():7.3f} {a.resid_m.mean():+7.3f}")
    out = T[["scid", "prn", "svn", "sys", "cls", "block", "n", "dR", "dA", "dC", "clk",
             "apc_z_m", "radial_pred_m", "resid_m", "dR_iqr"]].copy()
    out.columns = ["spacecraft", "prn", "svn", "sys", "orbit_class", "block", "n_epochs",
                   "meas_radial_m", "meas_along_m", "meas_cross_m", "meas_clock_m",
                   "antex_z_m", "predicted_radial_m", "residual_m", "radial_iqr_m"]
    out = out.round(4).sort_values(["sys", "orbit_class", "block", "prn"])
    out.to_csv("satellite_offset_table.csv", index=False)
    print(f"\nwrote satellite_offset_table.csv  ({len(out)} spacecraft)\n")
    print("--- by block type ---")
    b = T[ok].groupby("block").agg(n=("prn", "size"), antex_z=("apc_z_m", "mean"),
                                   meas=("dR", lambda s: -s.mean()),
                                   resid=("resid_m", "mean"), sd=("resid_m", "std"))
    print(b.round(3).to_string())


RUN = {"parse": step_parse, "join": step_join, "table": step_table}

if __name__ == "__main__":
    for name in (STEPS if WHICH == "all" else [WHICH]):
        RUN[name]()
