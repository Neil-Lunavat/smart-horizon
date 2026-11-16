#!/usr/bin/env python3
"""Broadcast-only feature: the jump when one nav message hands over to the next.

At each new TOE, evaluate the OUTGOING record and the INCOMING record at the same
instant. Their disagreement is the ground segment publicly correcting itself. It
needs no SP3, and it is known the moment the new message arrives -> causal.
"""
import glob, datetime as dt, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from gnss import read_nav, eval_broadcast_vec, GPS_EPOCH
from build_errors import qc_nav

def one(nav_path):
    day = Path(nav_path).name.split("_")[0]
    prev = dt.date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}") - dt.timedelta(days=1)
    pnav = Path("../raw/nav")/f"{prev:%Y%m%d}_nav.rnx.gz"
    try:
        nav = read_nav(nav_path, ("C","E","G","J"))
        if pnav.exists():
            p = read_nav(pnav, ("C","E","G","J"))
            if not p.empty:
                nav = (pd.concat([p,nav],ignore_index=True).sort_values(["sat","t_toe"])
                         .drop_duplicates(["sat","t_toe"]).reset_index(drop=True))
        if nav.empty: return None
        nav = qc_nav(nav).sort_values(["sat","t_toe"]).reset_index(drop=True)
        toe = (nav.t_toe - GPS_EPOCH).dt.total_seconds().to_numpy()
        sat = nav.sat.to_numpy()
        same = np.r_[False, sat[1:] == sat[:-1]]
        gap  = np.r_[0.0, np.diff(toe)]
        j = np.nonzero(same & (gap > 0) & (gap <= 7*3600))[0]
        if len(j) == 0: return None
        prv, cur, t = nav.iloc[j-1], nav.iloc[j], toe[j]
        xa,ya,za,ca,*_ = eval_broadcast_vec(prv, t)          # outgoing, extrapolated
        xb,yb,zb,cb,*_ = eval_broadcast_vec(cur, t)          # incoming, fresh
        # rotate the jump into RTN using a numerically differentiated broadcast velocity
        x2,y2,z2,*_ = eval_broadcast_vec(cur, t+1.0)
        pos = np.column_stack([xb,yb,zb]); vel = np.column_stack([x2-xb,y2-yb,z2-zb])
        dd  = np.column_stack([xa-xb, ya-yb, za-zb])
        eR = pos/np.linalg.norm(pos,axis=1,keepdims=True)
        h  = np.cross(pos,vel); eC = h/np.linalg.norm(h,axis=1,keepdims=True)
        eA = np.cross(eC,eR)
        return pd.DataFrame({"sat":cur.sat.to_numpy(), "arc_toe":t,
            "ho_R":np.einsum("ij,ij->i",dd,eR), "ho_A":np.einsum("ij,ij->i",dd,eA),
            "ho_C":np.einsum("ij,ij->i",dd,eC), "ho_clk_m":(ca-cb)*299792458.0,
            "ho_gap_s":gap[j]})
    except Exception as e:
        print(f"  {day}: {type(e).__name__}: {e}", flush=True); return None

if __name__ == "__main__":
    files = sorted(glob.glob("../raw/nav/*_nav.rnx.gz"))
    print(f"{len(files)} nav files", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        parts = [r for r in ex.map(one, files, chunksize=4) if r is not None]
    h = pd.concat(parts, ignore_index=True).drop_duplicates(["sat","arc_toe"])
    h.to_parquet("handover.parquet", index=False)
    print(f"wrote handover.parquet  rows={len(h):,}  sats={h.sat.nunique()}")
    print(h[["ho_R","ho_A","ho_C","ho_clk_m","ho_gap_s"]].describe(
        percentiles=[.5,.9,.99]).round(3).to_string())
