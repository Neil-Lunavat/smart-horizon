#!/usr/bin/env python3
"""Task A dataset: NOWCAST the error at epoch t.

Available at prediction time t:
  * broadcast, continuously, right up to t   (geometry + handover jumps)
  * SP3-derived truth only up to t - LAG     (LAG = 3 d, the real JAXA final lag)
So the model must bridge a 3-day gap using broadcast alone.

Emits dense per-satellite grids so both tabular and sequence models can be fed
from the identical arrays -- no model gets information another does not.
"""
import numpy as np, pandas as pd
CL=299792458.0; STEP=900; RE=6378137.0
LAG_D=3.0; LAG=int(LAG_D*86400/STEP)                 # 288 steps
T=["dR","dA","dC","dClk_m"]

d=pd.read_parquet("../full2.parquet")
d["dClk_m"]=d.dClk_ns*1e-9*CL
d=d[d.dClk_m.abs()<100]                              # C10/C06 gross blowups

# --- anom_z: robust per-satellite outlier score (regenerating the lost script) --
for c in T:
    a=d[c].abs()
    med=d.groupby("sat")[c].transform(lambda s:s.abs().median())
    mad=d.groupby("sat")[c].transform(lambda s:(s.abs()-s.abs().median()).abs().median())
    d["z_"+c]=(a-med)/(1.4826*mad+1e-9)
d["anom_z"]=d[["z_"+c for c in T]].max(axis=1)
d=d.drop(columns=["z_"+c for c in T])
d.loc[d.anom_z>10,T]=np.nan

# --- PRN reuse -> distinct spacecraft ------------------------------------------
r=np.sqrt(d.px**2+d.py**2+d.pz**2)
sid=d.sat.to_numpy().astype(object); m=d.sat.isin(["C13","C38","C39"]).to_numpy()
sid[m]=np.where(r.to_numpy()[m]>3.5e7,d.sat.to_numpy()[m]+"_a",d.sat.to_numpy()[m]+"_b")
d["scid"]=sid

# --- sun geometry (broadcast-derivable: needs only date + position) ------------
jd=2440587.5+d.epoch.astype("int64").to_numpy()/1e9/86400.0; n=jd-2451545.0
gmst=np.deg2rad((280.46061837+360.98564736629*n)%360.0)
cx,sx=np.cos(gmst),np.sin(gmst)
X,Y,Z=d.px.to_numpy(),d.py.to_numpy(),d.pz.to_numpy()
Pi=np.column_stack([X*cx-Y*sx, X*sx+Y*cx, Z])
L=np.deg2rad((280.460+0.9856474*n)%360.0); g=np.deg2rad((357.528+0.9856003*n)%360.0)
lam=L+np.deg2rad(1.915)*np.sin(g)+np.deg2rad(0.020)*np.sin(2*g); eps=np.deg2rad(23.439-4e-7*n)
S=np.column_stack([np.cos(lam),np.cos(eps)*np.sin(lam),np.sin(eps)*np.sin(lam)])
same=np.r_[d.scid.to_numpy()[1:]==d.scid.to_numpy()[:-1],False]
Nv=np.cross(Pi,np.roll(Pi,-1,axis=0)); Nv[~same]=np.nan
Nv/=np.linalg.norm(Nv,axis=1,keepdims=True)
d["beta"]=np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i",S,Nv),-1,1)))
Sp=S-Nv*np.einsum("ij,ij->i",S,Nv)[:,None]; Sp/=np.linalg.norm(Sp,axis=1,keepdims=True)
Ph=Pi/np.linalg.norm(Pi,axis=1,keepdims=True)
d["mu"]=(np.degrees(np.arctan2(np.einsum("ij,ij->i",np.cross(Nv,Sp),Ph),
                               np.einsum("ij,ij->i",Ph,Sp)))+180)%360
rr=np.linalg.norm(Pi,axis=1); cs=np.einsum("ij,ij->i",Ph,S)
d["eclipse"]=((cs<0)&(rr*np.sqrt(np.maximum(1-cs**2,0))<RE)).astype(float)
d["beta"]=d.groupby("scid")["beta"].ffill().bfill()
d["mu"]=d.groupby("scid")["mu"].ffill().bfill()

# --- handover ------------------------------------------------------------------
h=pd.read_parquet("handover.parquet")
for c in ["ho_R","ho_A","ho_C","ho_clk_m"]: h[c]=h[c].clip(-20,20)
d=d.merge(h,on=["sat","arc_toe"],how="left")

# --- broadcast-only feature columns --------------------------------------------
u=d.u_arg.to_numpy()
for k in (1,2,3): d[f"sin{k}u"],d[f"cos{k}u"]=np.sin(k*u),np.cos(k*u)
d["r_km"]=rr/1e3; d["latf"]=Z/rr
sod=d.epoch.dt.hour*3600+d.epoch.dt.minute*60
d["sin_tod"],d["cos_tod"]=np.sin(2*np.pi*sod/86400),np.cos(2*np.pi*sod/86400)
d["sin_mu"],d["cos_mu"]=np.sin(np.deg2rad(d.mu)),np.cos(np.deg2rad(d.mu))
d["absbeta"]=d.beta.abs()
BC=["sin1u","cos1u","sin2u","cos2u","sin3u","cos3u","r_km","latf","age_s",
    "sin_tod","cos_tod","sin_mu","cos_mu","beta","absbeta","eclipse",
    "ho_R","ho_A","ho_C","ho_clk_m","ho_gap_s"]
for c in BC: d[c]=d[c].astype("float32")

# --- dense grids ----------------------------------------------------------------
d=d.sort_values(["scid","epoch"]).reset_index(drop=True)
t0=d.epoch.min(); NT=int((d.epoch.max()-t0).total_seconds()//STEP)+1
sats=sorted(d.scid.unique()); si={s:i for i,s in enumerate(sats)}
E=np.full((len(sats),NT,4),np.nan,np.float32)          # truth (SP3-derived)
B=np.full((len(sats),NT,len(BC)),np.nan,np.float32)    # broadcast-only
k=((d.epoch-t0).dt.total_seconds()//STEP).astype(int).to_numpy()
ii=d.scid.map(si).to_numpy()
E[ii,k]=d[T].to_numpy(np.float32); B[ii,k]=d[BC].to_numpy(np.float32)
meta=pd.DataFrame({"scid":sats})
meta["sys"]=[s[0] for s in sats]
meta["cls"]=[d[d.scid==s].orbit_class.iloc[0] for s in sats]
np.savez_compressed("taskA.npz",E=E,B=B,sats=np.array(sats),
    bc_cols=np.array(BC),tgt_cols=np.array(T),t0=str(t0),NT=NT,LAG=LAG,
    sys=meta.sys.to_numpy(),cls=meta.cls.to_numpy())
print(f"grid {E.shape}  broadcast {B.shape}  span {t0} .. {d.epoch.max()}")
print(f"LAG = {LAG} steps ({LAG_D} d).  truth valid {np.isfinite(E[:,:,0]).mean()*100:.1f}%"
      f"  broadcast valid {np.isfinite(B[:,:,0]).mean()*100:.1f}%")
