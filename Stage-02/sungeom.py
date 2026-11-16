#!/usr/bin/env python3
"""Is along-track error structured by SUN GEOMETRY? Analysis first, no model.

Hypothesis (from the ANTEX header's own words: offsets follow a yaw-steering
attitude convention, Montenbruck et al. 2015) plus the documented fact that
along-track broadcast error is driven by solar-radiation-pressure mismodelling:
  * |along-track| error should grow at LOW |beta| (eclipse season, yaw turns)
  * along-track should be structured in ORBIT ANGLE RELATIVE TO THE SUN
Both computable from broadcast + a calendar. No SP3, no satellite ID.
"""
import numpy as np, pandas as pd
CL=299792458.0; RE=6378137.0
d=pd.read_parquet("../full2.parquet",columns=["epoch","sat","sys","orbit_class",
                    "dR","dA","dC","dClk_ns","px","py","pz","anom_z"])
d["dClk_m"]=d.dClk_ns*1e-9*CL
d=d[(d.anom_z<=10)&(d.dClk_m.abs()<100)].sort_values(["sat","epoch"]).reset_index(drop=True)

# --- ECEF -> ECI (Earth rotation only; precession/nutation irrelevant at this level)
jd = 2440587.5 + d.epoch.astype("int64").to_numpy()/1e9/86400.0
n  = jd - 2451545.0
gm = np.deg2rad((280.46061837 + 360.98564736629*n) % 360.0)
cx,sx = np.cos(gm), np.sin(gm)
X = d.px.to_numpy(); Y = d.py.to_numpy(); Z = d.pz.to_numpy()
xi =  X*cx - Y*sx;  yi =  X*sx + Y*cx;  zi = Z            # ECI position
# --- Sun unit vector in ECI (low-precision solar ephemeris, ~0.01 deg)
L  = np.deg2rad((280.460 + 0.9856474*n) % 360.0)
gg = np.deg2rad((357.528 + 0.9856003*n) % 360.0)
lam= L + np.deg2rad(1.915)*np.sin(gg) + np.deg2rad(0.020)*np.sin(2*gg)
eps= np.deg2rad(23.439 - 4e-7*n)
sx_,sy_,sz_ = np.cos(lam), np.cos(eps)*np.sin(lam), np.sin(eps)*np.sin(lam)
# --- orbit normal from consecutive ECI positions of the same satellite
P = np.column_stack([xi,yi,zi])
same = np.r_[d.sat.to_numpy()[1:]==d.sat.to_numpy()[:-1], False]
P2 = np.roll(P,-1,axis=0)
Nv = np.cross(P,P2); Nv[~same] = np.nan
Nv /= np.linalg.norm(Nv,axis=1,keepdims=True)
S  = np.column_stack([sx_,sy_,sz_])
d["beta"] = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i",S,Nv),-1,1)))
# --- orbit angle mu, measured from orbital MIDNIGHT (anti-sun direction)
Sp = S - Nv*np.einsum("ij,ij->i",S,Nv)[:,None]            # sun projected into plane
Sp/= np.linalg.norm(Sp,axis=1,keepdims=True)
Ph = P/np.linalg.norm(P,axis=1,keepdims=True)
cosmu = np.einsum("ij,ij->i",Ph,Sp)
sinmu = np.einsum("ij,ij->i",np.cross(Nv,Sp),Ph)
d["mu"] = (np.degrees(np.arctan2(sinmu,cosmu))+180)%360   # 0 = midnight, 180 = noon
r = np.linalg.norm(P,axis=1)
d["eclipse"] = (np.einsum("ij,ij->i",Ph,S)<0) & (r*np.sqrt(1-np.einsum("ij,ij->i",Ph,S)**2)<RE)
d=d.dropna(subset=["beta","mu"])
print(f"rows {len(d):,}   eclipse fraction {d.eclipse.mean()*100:.2f}%\n")

# per-satellite demean so we measure STRUCTURE, not the constant offset
for c in ["dR","dA","dC","dClk_m"]:
    d[c+"_d"]=d[c]-d.groupby("sat")[c].transform("median")
d["grp"]=np.where(d.sys=="G","GPS",np.where(d.sys=="E","GAL",
         np.where(d.sys=="C","BDS","QZS")))
print("=== |beta| vs RMS of the de-meaned error (m) ===")
bb=pd.cut(d.beta.abs(),[0,2,5,10,15,20,30,45,60,90])
for g,h in d.groupby("grp"):
    t=h.groupby(bb,observed=True).apply(lambda x:pd.Series(
        {c:np.sqrt((x[c+"_d"]**2).mean()) for c in ["dR","dA","dC","dClk_m"]}|{"n":len(x)}),
        include_groups=False)
    print(f"\n--- {g}"); print(t.round(3).to_string())
print("\n=== orbit angle mu (0=midnight,180=noon) vs RMS along-track, LOW |beta|<10 only ===")
lo=d[d.beta.abs()<10]
mm=pd.cut(lo.mu,np.arange(0,361,30))
for g,h in lo.groupby("grp"):
    t=h.groupby(mm,observed=True).apply(lambda x:pd.Series(
        {"dA":np.sqrt((x.dA_d**2).mean()),"dC":np.sqrt((x.dC_d**2).mean()),"n":len(x)}),
        include_groups=False)
    print(f"\n--- {g}"); print(t.round(3).to_string())
print("\n=== variance of de-meaned along-track explained by (beta,mu) bins ===")
for g,h in d.groupby("grp"):
    b2=pd.cut(h.beta,np.arange(-90,91,10)); m2=pd.cut(h.mu,np.arange(0,361,15))
    for c in ["dA_d","dC_d","dR_d"]:
        pred=h.groupby([b2,m2],observed=True)[c].transform("mean")
        print(f"  {g} {c}: R2 = {1-(h[c]-pred).var()/h[c].var():.4f}")
d[["epoch","sat","sys","orbit_class","beta","mu","eclipse"]].to_parquet("sungeom.parquet",index=False)
