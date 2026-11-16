#!/usr/bin/env python3
"""Stage 2 (rebuilt). Broadcast-minus-precise error series.

Every choice here was verified against real CDDIS files on 2026-08-02..04:
  * SP3 clocks == 30S CLK file to 1.6 ps  -> CLK downloads skipped
  * BDS nav carries BDT week (1074 == GPS 2430), TOE in BDT SOW -> +1356 wk, +14 s
  * BDS/NavIC IODE is AODE (near-constant) -> arcs keyed on TOE, never IODE
  * ~0.17% of usable BDS nav records are stale (elements not advanced, TOE bumped)
    producing 1e7 m errors -> rejected by consecutive-record consistency check
  * BDS GEO (C01-C05) is unusable: no AC publishes precise orbits for it, and its
    DLR-merged nav records fail self-consistency at 1e6 m. Excluded.
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from gnss import (read_nav, read_sp3, eval_broadcast, eval_broadcast_vec, ecef_to_rtn, classify, GPS_EPOCH)

MAXAGE = {"G":7200.,"E":4800.,"C":3600.,"J":7200.,"I":7200.}
EXCLUDE = {"C01","C02","C03","C04","C05","C59","C60","C61"}

def qc_nav(nav, tol_m=1000.0, tol_clk_ns=100.0):
    """Drop records that disagree with the last good record at their own TOE, in
    position (>tol_m) or clock (>tol_clk_ns). Truth-free: uses only the broadcast
    messages. Catches stale records where elements/af0 were not advanced but TOE
    was bumped (C38, C07: 1e7 m; C06, C10: 4e5 ns).

    Iterated: each pass compares against the surviving predecessor, so a bad
    record does not condemn the good one after it."""
    nav = nav.sort_values(["sat", "t_toe"])
    for _ in range(4):
        toe = (nav["t_toe"] - GPS_EPOCH).dt.total_seconds().to_numpy()
        sat = nav["sat"].to_numpy()
        same = np.r_[False, sat[1:] == sat[:-1]]
        dtv  = np.r_[0.0, np.diff(toe)]
        cand = same & (dtv > 0) & (dtv <= 7*3600)
        if not cand.any():
            break
        j = np.nonzero(cand)[0]                      # candidate rows
        prev = nav.iloc[j-1]; cur = nav.iloc[j]
        t = toe[j]
        xa,ya,za,ca,*_ = eval_broadcast_vec(prev, t)
        xb,yb,zb,cb,*_ = eval_broadcast_vec(cur,  t)
        d  = np.sqrt((xa-xb)**2 + (ya-yb)**2 + (za-zb)**2)
        dc = np.abs(ca-cb)*1e9
        bad = (d > tol_m) | (dc > tol_clk_ns)
        if not bad.any():
            break
        nav = nav.drop(nav.index[j[bad]])
    return nav


def build_day(nav_path, orb_path, systems, step=900, tol_m=1000.0, day=None, prev_nav=None):
    nav=read_nav(nav_path,tuple(systems))
    # A day's own nav file has no record valid before its first TOE, leaving a
    # hole at every midnight (hour 0 was 15% short, BDS 25%). That hole breaks
    # every 7-day window. Prepend the previous day's records to close it.
    if prev_nav is not None and Path(prev_nav).exists():
        p=read_nav(prev_nav,tuple(systems))
        if not p.empty:
            nav=(pd.concat([p,nav],ignore_index=True)
                   .sort_values(["sat","t_toe"])
                   .drop_duplicates(["sat","t_toe"]).reset_index(drop=True))
    if nav.empty: return pd.DataFrame()
    n0=len(nav); nav=qc_nav(nav,tol_m); ndrop=n0-len(nav)
    sp3=read_sp3(orb_path)
    sp3=sp3[sp3.sat.str[0].isin(list(systems)) & ~sp3.sat.isin(EXCLUDE)]
    if sp3.empty: return pd.DataFrame()
    out=[]
    for sat,g in sp3.groupby("sat",sort=False):
        g=g.sort_values("epoch").reset_index(drop=True)
        pos=g[["px","py","pz"]].to_numpy()
        if len(pos)<20: continue
        t=g.epoch.astype("int64").to_numpy()/1e9
        vel=np.gradient(pos,t,axis=0)
        nsat=nav[nav.sat==sat]
        if nsat.empty: continue
        t_abs=(g.epoch-GPS_EPOCH).dt.total_seconds().to_numpy()
        toe=(nsat.t_toe-GPS_EPOCH).dt.total_seconds().to_numpy()
        o=np.argsort(toe); toe=toe[o]; recs=nsat.iloc[o]
        idx=np.searchsorted(toe,t_abs,side="right")-1
        ok=idx>=0
        age=np.where(ok,t_abs-toe[np.clip(idx,0,None)],np.nan)
        ok&= age<=MAXAGE.get(sat[0],7200.)*2.0
        bx=np.full(len(g),np.nan); by=bx.copy(); bz=bx.copy(); bc=bx.copy()
        aod=bx.copy(); arc=bx.copy(); uk=bx.copy()
        for ri in np.unique(idx[ok]):
            m=ok&(idx==ri); rec=recs.iloc[ri]
            x,y,z,dts,_,u=eval_broadcast(rec,t_abs[m])
            bx[m],by[m],bz[m],bc[m]=x,y,z,dts; uk[m]=u
            aod[m]=t_abs[m]-toe[ri]; arc[m]=toe[ri]
        d=np.column_stack([bx-pos[:,0],by-pos[:,1],bz-pos[:,2]])
        dR,dA,dC=ecef_to_rtn(d,pos,vel)
        keep=((g.epoch.dt.minute*60+g.epoch.dt.second)%step==0).to_numpy()
        df=pd.DataFrame({"epoch":g.epoch,"sat":sat,"sys":sat[0],
            "dR":dR,"dA":dA,"dC":dC,"dClk_ns":(bc-g.pclk.to_numpy())*1e9,
            "dx":d[:,0],"dy":d[:,1],"dz":d[:,2],
            "px":pos[:,0],"py":pos[:,1],"pz":pos[:,2],
            "age_s":aod,"arc_toe":arc,"u_arg":uk})
        df["orbit_class"]=classify(pos)
        out.append(df[keep])
    r=pd.concat(out,ignore_index=True) if out else pd.DataFrame()
    # SP3 spans 00:00 THROUGH 24:00, so consecutive files share a midnight epoch
    # and disagree there by ~0.2 m (seam between 3-day long-arc solutions).
    # Keep only epochs belonging to this file's own day.
    if day is not None and not r.empty:
        r = r[r["epoch"].dt.strftime("%Y%m%d") == day]
    if not r.empty: r.attrs["ndrop"]=ndrop
    return r

def _one(args):
    """Top-level so ProcessPoolExecutor can pickle it (required on Windows)."""
    nav_path, orb, systems, step, prev_nav = args
    day = Path(nav_path).name.split("_")[0]
    try:
        df = build_day(nav_path, orb, systems, step, day=day, prev_nav=prev_nav)
    except Exception as e:
        return day, None, f"{type(e).__name__}: {e}"
    if df.empty:
        return day, None, "empty"
    return day, df, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",default="../raw"); ap.add_argument("--out",default="errors.parquet")
    ap.add_argument("--systems",default="CEGJ"); ap.add_argument("--step",type=int,default=900)
    ap.add_argument("--workers",type=int,default=max(1,(os.cpu_count() or 4)-1))
    a=ap.parse_args()
    import datetime as _dt
    jobs=[]
    for nav_path in sorted(glob.glob(f"{a.raw}/nav/*_nav.rnx.gz")):
        day=Path(nav_path).name.split("_")[0]
        orb=Path(a.raw)/"orb"/f"{day}_orb.sp3.gz"
        prev=(_dt.date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}")-_dt.timedelta(days=1))
        pnav=Path(a.raw)/"nav"/f"{prev:%Y%m%d}_nav.rnx.gz"
        if orb.exists(): jobs.append((nav_path,str(orb),a.systems,a.step,str(pnav)))
    print(f"{len(jobs)} days, {a.workers} workers")
    frames=[]; nbad=0
    # CPU-bound parsing -> processes, not threads (the GIL would serialise threads)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i,(day,df,err) in enumerate(ex.map(_one,jobs,chunksize=2),1):
            if err: nbad+=1; print(f"  {day}: {err}",flush=True)
            else:
                frames.append(df)
                if i%40==0: print(f"  ..{i}/{len(jobs)} days",flush=True)
    if not frames: raise SystemExit("nothing built")
    A=pd.concat(frames,ignore_index=True)
    A=A.sort_values(["sat","epoch"]).reset_index(drop=True)
    A["arc_id"]=A.groupby("sat")["arc_toe"].transform(lambda s:(s.diff().fillna(1)!=0).cumsum())
    A["day"]=A["epoch"].dt.floor("D")
    A["dClk_debias_ns"]=A["dClk_ns"]-A.groupby(["sat","day"])["dClk_ns"].transform("median")
    A=A.drop(columns=["day"])
    A=A.dropna(subset=["dR","dA","dC"])
    A.to_parquet(a.out,index=False)
    print(f"wrote {a.out}  rows={len(A):,}  sats={A.sat.nunique()}  "
          f"days={len(frames)}  mem={A.memory_usage(deep=True).sum()/1e6:.0f} MB")

if __name__=="__main__": main()
