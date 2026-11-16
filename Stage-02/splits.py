#!/usr/bin/env python3
"""Does the model learn physics, or memorise satellites?

Three holdouts of increasing strictness:
  TIME  unseen future, satellites seen in training      (the earlier test)
  SAT   unseen satellites, time seen in training
  BOTH  unseen satellites AND unseen future             (the honest claim)
Plus an ablation dropping r_km/latf, which could act as a satellite fingerprint.
"""
import numpy as np, pandas as pd, xgboost as xgb
T=["dR","dA","dC","dClk_m"]
d=pd.read_parquet("features.parquet")
F0=["sin1u","cos1u","sin2u","cos2u","sin3u","cos3u","r_km","latf","age_s",
    "sin_tod","cos_tod","sys_c","cls_c"]
HO=["ho_R","ho_A","ho_C","ho_clk_m","ho_gap_s"]
F1=F0+HO+[c+s for c in ["ho_R","ho_A","ho_C","ho_clk_m"] for s in ("_p","_m7","_s7")]
F2=F1+["lb_"+c for c in T]
F2nofp=[f for f in F2 if f not in ("r_km","latf")]
for c in ["sys_c","cls_c"]: d[c]=d[c].astype("category")

sats=np.array(sorted(d.scid.unique()))
hold=set(sats[::3])                                   # every 3rd spacecraft held out
print(f"{len(sats)} spacecraft, {len(hold)} held out")
cut=d.epoch.quantile(0.70); gap=pd.Timedelta(days=8)
early,late = d.epoch<cut, d.epoch>=cut+gap
seen = ~d.scid.isin(hold); unseen = d.scid.isin(hold)
SPL={"TIME":(early&seen, late&seen), "SAT":(early&seen, early&unseen),
     "BOTH":(early&seen, late&unseen)}
P=dict(tree_method="hist",device="cuda",max_depth=8,learning_rate=0.05,
       subsample=0.8,colsample_bytree=0.8,n_estimators=400,
       enable_categorical=True,verbosity=0)
rows=[]
for sname,(trm,tem) in SPL.items():
    for fname,F in [("F0 geom",F0),("F1 +handover",F1),("F2 +lagbias",F2),("F2 no-r/lat",F2nofp)]:
        for c in T:
            ok=d[c].notna()
            tr,te=trm&ok,tem&ok
            if te.sum()<5000: continue
            m=xgb.XGBRegressor(**P).fit(d.loc[tr,F],d.loc[tr,c])
            p=m.predict(d.loc[te,F]); y=d.loc[te,c].to_numpy()
            rows.append(dict(split=sname,feats=fname,target=c,
                raw=np.sqrt((y**2).mean()),cor=np.sqrt(((y-p)**2).mean())))
R=pd.DataFrame(rows); R["pct"]=(100*(1-R.cor/R.raw)).round(1)
for sname in SPL:
    s=R[R.split==sname]
    print(f"\n=== {sname}  (train {SPL[sname][0].sum():,} / test {SPL[sname][1].sum():,})")
    piv=s.pivot(index="feats",columns="target",values=["raw","cor","pct"])
    for f in ["F0 geom","F1 +handover","F2 +lagbias","F2 no-r/lat"]:
        cells=[]
        for c in T:
            r=s[(s.feats==f)&(s.target==c)]
            cells.append(f"{r.raw.iloc[0]:.3f}->{r.cor.iloc[0]:.3f} {r.pct.iloc[0]:5.1f}%" if len(r) else "     .     ")
        print(f"  {f:13s} "+"  ".join(f"{x:>21s}" for x in cells))
print("\n(columns: "+"  ".join(T)+")")
R.to_csv("split_results.csv",index=False)
