#!/usr/bin/env python3
"""How much does SP3 LATENCY cost us?

F2 used SP3 that was 14-28 days old, because JAXA FINAL has a ~2-week lag.
But rapid MGEX products (GFZ0MGXRAP, already in fetch_all.py's fallback chain)
land in ~1-3 days. If skill is steep in latency, switching product tier is worth
far more than any modelling change. This measures that curve on data we already have.
"""
import numpy as np, pandas as pd, xgboost as xgb
T=["dR","dA","dC","dClk_m"]
d=pd.read_parquet("features.parquet")
F0=["sin1u","cos1u","sin2u","cos2u","sin3u","cos3u","r_km","latf","age_s",
    "sin_tod","cos_tod","sys_c","cls_c"]
HO=["ho_R","ho_A","ho_C","ho_clk_m","ho_gap_s"]
F1=F0+HO+[c+s for c in ["ho_R","ho_A","ho_C","ho_clk_m"] for s in ("_p","_m7","_s7")]
for c in ["sys_c","cls_c"]: d[c]=d[c].astype("category")
d=d.sort_values(["scid","epoch"]).reset_index(drop=True)
g=d.groupby("scid",observed=True)

LAT=[0.25,1,2,3,5,7,14,28]          # SP3 latency in days
newf={}
for L in LAT:
    s=int(round(L*96))
    for c in T:
        newf[f"v{L}_{c}"]=g[c].shift(s)                       # value at t-L
        newf[f"m{L}_{c}"]=g[c].transform(                     # 7-day mean ending t-L
            lambda x,s=s: x.shift(s).rolling(672,min_periods=96).mean())
d=pd.concat([d,pd.DataFrame(newf,index=d.index)],axis=1)

cut=d.epoch.quantile(0.70); gap=pd.Timedelta(days=8)
tr=d.epoch<cut; te=d.epoch>=cut+gap
P=dict(tree_method="hist",device="cuda",max_depth=8,learning_rate=0.05,subsample=0.8,
       colsample_bytree=0.8,n_estimators=400,enable_categorical=True,verbosity=0)
print(f"train {tr.sum():,}  test {te.sum():,}\n")
print(f"{'SP3 latency':>12s} "+"".join(f"{c:>17s}" for c in T))
base={}
for c in T:
    y=d.loc[te&d[c].notna(),c]; base[c]=np.sqrt((y**2).mean())
print(f"{'raw (none)':>12s} "+"".join(f"{base[c]:17.3f}" for c in T))
for L in LAT:
    row=[]
    for c in T:
        F=F1+[f"v{L}_{k}" for k in T]+[f"m{L}_{k}" for k in T]
        ok=d[c].notna()
        m=xgb.XGBRegressor(**P).fit(d.loc[tr&ok,F],d.loc[tr&ok,c])
        p=m.predict(d.loc[te&ok,F]); y=d.loc[te&ok,c].to_numpy()
        r=np.sqrt(((y-p)**2).mean())
        row.append(f"{r:.3f} ({100*(1-r/base[c]):4.1f}%)")
    print(f"{str(L)+' d':>12s} "+"".join(f"{x:>17s}" for x in row))
