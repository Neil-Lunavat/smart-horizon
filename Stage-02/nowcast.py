#!/usr/bin/env python3
"""Can the error at the CURRENT epoch be predicted from broadcast alone?

Ladder of feature sets, each strictly a superset of the last:
  F0 geometry   pure broadcast, no identity   (where is it in its orbit)
  F1 +handover  pure broadcast, real-time     (how much did the last upload jump)
  F2 +lagbias   SP3 older than 14 days        (operationally available: 2-week lag)
  F3 +recent    error 15 min ago              (NOT available -- ceiling only)
Scored as RMS reduction vs applying no correction at all, on a held-out later period.
"""
import numpy as np, pandas as pd, lightgbm as lgb
CL=299792458.0; T=["dR","dA","dC","dClk_m"]

d=pd.read_parquet("../full2.parquet",columns=["epoch","sat","sys","orbit_class","dR","dA","dC",
      "dClk_ns","dClk_debias_ns","age_s","u_arg","arc_id","arc_toe","px","py","pz","anom_z"])
d["dClk_m"]=d.dClk_ns*1e-9*CL
d=d[(d.anom_z<=10)&(d.dClk_m.abs()<100)].copy()
# PRN reuse -> distinct spacecraft
r=np.sqrt(d.px**2+d.py**2+d.pz**2)
sid=d.sat.to_numpy().astype(object); m=d.sat.isin(["C13","C38","C39"]).to_numpy()
sid[m]=np.where(r.to_numpy()[m]>3.5e7,d.sat.to_numpy()[m]+"_a",d.sat.to_numpy()[m]+"_b")
d["scid"]=sid
h=pd.read_parquet("handover.parquet")
d=d.merge(h,on=["sat","arc_toe"],how="left")
print(f"rows {len(d):,}   handover matched {d.ho_R.notna().mean()*100:.1f}%")

# ---- F0 geometry (pure broadcast) -----------------------------------------
u=d.u_arg.to_numpy()
for k in (1,2,3):
    d[f"sin{k}u"]=np.sin(k*u); d[f"cos{k}u"]=np.cos(k*u)
d["r_km"]=r/1e3; d["latf"]=d.pz/r
sod=d.epoch.dt.hour*3600+d.epoch.dt.minute*60
d["sin_tod"]=np.sin(2*np.pi*sod/86400); d["cos_tod"]=np.cos(2*np.pi*sod/86400)
d["sys_c"]=d.sys.astype("category"); d["cls_c"]=d.orbit_class.astype("category")
F0=["sin1u","cos1u","sin2u","cos2u","sin3u","cos3u","r_km","latf","age_s",
    "sin_tod","cos_tod","sys_c","cls_c"]
# ---- F1 handover (pure broadcast, causal) ---------------------------------
d=d.sort_values(["scid","epoch"])
HO=["ho_R","ho_A","ho_C","ho_clk_m","ho_gap_s"]
for c in HO: d[c]=d[c].clip(-50,50) if c!="ho_gap_s" else d[c]
g=d.groupby("scid",observed=True)
for c in ["ho_R","ho_A","ho_C","ho_clk_m"]:
    d[c+"_p"]=g[c].shift(96)                                   # previous day's arc jump
    d[c+"_m7"]=g[c].transform(lambda s:s.shift(1).rolling(672,min_periods=48).mean())
    d[c+"_s7"]=g[c].transform(lambda s:s.shift(1).rolling(672,min_periods=48).std())
F1=F0+HO+[c+s for c in ["ho_R","ho_A","ho_C","ho_clk_m"] for s in ("_p","_m7","_s7")]
# ---- F2 lagged SP3 bias (>=14 days old) -----------------------------------
d["date"]=d.epoch.dt.floor("D")
dy=d.groupby(["scid","date"],observed=True)[T].mean().reset_index()
dy=dy.sort_values(["scid","date"])
for c in T:
    dy["lb_"+c]=(dy.groupby("scid",observed=True)[c]
                   .transform(lambda s:s.shift(14).rolling(14,min_periods=3).mean()))
d=d.merge(dy[["scid","date"]+["lb_"+c for c in T]],on=["scid","date"],how="left")
F2=F1+["lb_"+c for c in T]
# ---- F3 recent error (NOT operationally available) ------------------------
d=d.sort_values(["scid","epoch"]); g=d.groupby("scid",observed=True)
for c in T: d["rec_"+c]=g[c].shift(1)
F3=F2+["rec_"+c for c in T]

cut=d.epoch.quantile(0.70); gap=pd.Timedelta(days=8)
tr=d.epoch<cut; te=d.epoch>=cut+gap
print(f"train {tr.sum():,}  test {te.sum():,}  split at {cut:%Y-%m-%d}\n")
P={"objective":"l2","learning_rate":0.05,"num_leaves":127,"min_data_in_leaf":200,
   "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":1,"verbose":-1,"num_threads":8}
res={}
for name,F in [("F0 geom",F0),("F1 +handover",F1),("F2 +lagbias",F2),("F3 +recent",F3)]:
    for c in T:
        y=d[c]; ok=y.notna()
        mdl=lgb.train(P,lgb.Dataset(d.loc[tr&ok,F],y[tr&ok]),num_boost_round=400)
        p=mdl.predict(d.loc[te&ok,F]); yt=y[te&ok].to_numpy()
        res[(name,c)]=(np.sqrt((yt**2).mean()),np.sqrt(((yt-p)**2).mean()))
        if name=="F2 +lagbias" and c=="dR":
            imp=pd.Series(mdl.feature_importance("gain"),index=F).sort_values(ascending=False)
            top=imp.head(10)
print(f"{'features':14s} "+"".join(f"{c:>22s}" for c in T))
print(f"{'':14s} "+"".join(f"{'raw -> corrected  %':>22s}" for c in T))
for name in ["F0 geom","F1 +handover","F2 +lagbias","F3 +recent"]:
    cells=[]
    for c in T:
        a,b=res[(name,c)]; cells.append(f"{a:.3f} -> {b:.3f} {100*(1-b/a):5.1f}%")
    print(f"{name:14s} "+"".join(f"{x:>22s}" for x in cells))
print("\ntop gain, F2 model on dR:"); print((top/top.sum()*100).round(1).to_string())
d.to_parquet("features.parquet",index=False)
