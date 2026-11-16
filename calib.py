"""Measure the sigma calibration factor: actual day-8 RMS / reported 1-sigma."""
import numpy as np, pandas as pd
from taskB_final import XYZ, CLS, S, NT, WIN, HOR, STEP, thin
from smart_horizon import predict
T0 = pd.Timestamp("2025-01-01")
COLS = ["x_error","y_error","z_error","satclockerror"]
rg = np.random.default_rng(77)
acc = {True:[[],[]], False:[[],[]]}
got=0
while got < 700:
    s=int(rg.integers(0,S)); o=int(rg.integers(WIN,NT-HOR))
    w,f = XYZ[s,o-WIN:o], XYZ[s,o:o+HOR]
    if np.isfinite(w[:,0]).mean()<.5 or np.isfinite(f[:,0]).mean()<.5: continue
    iw=np.nonzero(np.isfinite(w[:,0]))[0]
    tw,vw = thin((iw-WIN)*float(STEP), w[iw], 0.30, rg)
    if len(tw)<30: continue
    iq=np.nonzero(np.isfinite(f[:,0]))[0]
    if len(iq)<8: continue
    tr=pd.DataFrame({"utc_time":T0+pd.to_timedelta(tw,"s")})
    for j,c in enumerate(COLS): tr[c]=vw[:,j]
    tq=iq*float(STEP); geo=CLS[s] in ("GEO","GSO")
    out=predict(tr, T0+pd.to_timedelta(tq,"s"), orbit="GEO" if geo else "MEO")
    P=out[COLS].to_numpy(float); sg=out[[c+"_sigma" for c in COLS]].to_numpy(float)[0]
    r=np.sqrt(((f[iq]-P)**2).mean(0))
    acc[geo][0].append(r); acc[geo][1].append(sg); got+=1
for geo,nm in ((True,"GEO"),(False,"MEO")):
    R=np.array(acc[geo][0]); G=np.array(acc[geo][1])
    if len(R)<10: continue
    # pooled ratio: sqrt(mean r^2 / mean sigma^2) -- variance-correct, not mean-of-ratios
    k=np.sqrt((R**2).mean(0)/ (G**2).mean(0))
    med=np.median(R/np.maximum(G,1e-9),axis=0)
    print(f"{nm}  n={len(R)}  pooled k = " + "  ".join(f"{v:.3f}" for v in k)
          + "   | median ratio " + "  ".join(f"{v:.2f}" for v in med))
