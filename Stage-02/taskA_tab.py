#!/usr/bin/env python3
"""Task A model ladder, part 1: baselines + linear + gradient boosting.
Every model sees EXACTLY the same information: broadcast up to t, truth to t-LAG."""
import numpy as np, pandas as pd, xgboost as xgb
from scipy import stats
z=np.load("taskA.npz",allow_pickle=True)
E,B=z["E"],z["B"]; LAG=int(z["LAG"]); NT=int(z["NT"])
BC=list(z["bc_cols"]); T=list(z["tgt_cols"]); sats=z["sats"]; SYS=z["sys"]; CLS=z["cls"]
S=len(sats); W=672                                     # 7-day history window
TR_END,VA_S,VA_E,TE_S=18720,19488,20832,21600          # step indices, gapped

def build(stride):
    """One row per (satellite, epoch). Features are broadcast-at-t plus
    statistics of the truth window that ends LAG steps before t."""
    rows=[]
    for s in range(S):
        e,b=E[s],B[s]
        t=np.arange(LAG+W,NT,stride)
        y=e[t]
        ok=np.isfinite(y).all(1)&np.isfinite(b[t]).all(1)
        t=t[ok]; y=y[ok]
        if len(t)<50: continue
        hi=t-LAG                                        # last usable truth index
        idx=hi[:,None]-np.arange(W)[None,:]             # (n,W) window indices
        win=e[idx]                                      # (n,W,4)
        with np.errstate(all="ignore"):
            wm=np.nanmean(win,axis=1); ws=np.nanstd(win,axis=1)
            wmed=np.nanmedian(win,axis=1)
            wm1=np.nanmean(win[:,:96],axis=1)           # most recent day
            wm3=np.nanmean(win[:,:288],axis=1)
        last=e[hi]
        lag96=e[np.maximum(hi-96,0)]; lag192=e[np.maximum(hi-192,0)]
        # same-orbital-phase copy: nearest earlier epoch with matching u
        f=np.column_stack([b[t],wm,ws,wmed,wm1,wm3,last,lag96,lag192,
                           np.isfinite(win[:,:,0]).mean(1)[:,None]])
        rows.append((np.full(len(t),s),t,f,y))
    si=np.concatenate([r[0] for r in rows]); tt=np.concatenate([r[1] for r in rows])
    F=np.vstack([r[2] for r in rows]); Y=np.vstack([r[3] for r in rows])
    return si,tt,F,Y
names=(BC+[f"{p}_{c}" for p in ["wm","ws","wmed","wm1","wm3","last","l96","l192"] for c in T]
       +["cov"])
si,tt,F,Y=build(2)
F=np.nan_to_num(F,nan=0.0,posinf=0.0,neginf=0.0)
sysn=np.array([{"G":0,"E":1,"C":2,"J":3}[SYS[i]] for i in si],np.float32)
clsn=np.array([{"MEO":0,"GSO":1,"GEO":2}[CLS[i]] for i in si],np.float32)
F=np.column_stack([F,sysn,clsn]); names=names+["sysn","clsn"]
tr=tt<TR_END; va=(tt>=VA_S)&(tt<VA_E); te=tt>=TE_S
print(f"samples {len(tt):,}   train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}")
print(f"features {F.shape[1]}   test span steps {tt[te].min()}..{tt[te].max()}\n")

W_R={"MEO":0.98,"GSO":0.99,"GEO":0.99}; W_A={"MEO":0.139,"GSO":0.123,"GEO":0.123}
wR=np.array([W_R[CLS[i]] for i in si]); wA=np.array([W_A[CLS[i]] for i in si])
def sisre(P,m):
    R,A,C,K=P[:,0],P[:,1],P[:,2],P[:,3]
    return np.sqrt(np.mean((wR[m]*R-K)**2+wA[m]**2*(A**2+C**2)))
def report(tag,P,m=te):
    res=Y[m]-P
    out=[f"{tag:22s}"]
    for j,c in enumerate(T): out.append(f"{np.sqrt((res[:,j]**2).mean()):7.3f}")
    out.append(f"{sisre(Y[m]-res,m):8.3f}")
    print("  ".join(out)); return res
print(f"{'model':22s} "+"  ".join(f"{c:>7s}" for c in T)+"      SISRE")
print(f"{'-'*22} "+"  ".join(["-"*7]*4)+"  "+"-"*8)
report("raw broadcast",np.zeros_like(Y[te]))
i_wm=[names.index(f"wm_{c}") for c in T]; i_med=[names.index(f"wmed_{c}") for c in T]
i_last=[names.index(f"last_{c}") for c in T]; i_wm1=[names.index(f"wm1_{c}") for c in T]
report("7d-window mean",F[te][:,i_wm])
report("7d-window median",F[te][:,i_med])
report("1d mean @lag",F[te][:,i_wm1])
report("persistence @lag",F[te][:,i_last])
mu=F[tr].mean(0); sd=F[tr].std(0)+1e-9
Xs=(F-mu)/sd
A=np.column_stack([np.ones(len(Xs)),Xs])
lam=1e-2*len(A[tr])
Pl=np.column_stack([A[te]@np.linalg.solve(A[tr].T@A[tr]+lam*np.eye(A.shape[1]),A[tr].T@Y[tr,j])
                    for j in range(4)])
report("ridge",Pl)
P=dict(tree_method="hist",device="cuda",max_depth=8,learning_rate=0.04,subsample=0.8,
       colsample_bytree=0.8,n_estimators=900,early_stopping_rounds=40,verbosity=0)
Pg=np.zeros_like(Y[te]); imp={}
for j,c in enumerate(T):
    m=xgb.XGBRegressor(**P).fit(F[tr],Y[tr,j],eval_set=[(F[va],Y[va,j])],verbose=False)
    Pg[:,j]=m.predict(F[te])
    imp[c]=pd.Series(m.feature_importances_,index=names).sort_values(ascending=False).head(6)
res=report("XGBoost",Pg)
np.savez("tab_pred.npz",Pg=Pg,Yte=Y[te],si=si[te],tt=tt[te],wR=wR[te],wA=wA[te])
print("\nresidual normality of the XGBoost model (test set):")
print(f"{'target':8s} {'mean':>8s} {'std':>8s} {'skew':>7s} {'kurt':>8s} {'ShapiroW':>9s}")
for j,c in enumerate(T):
    r=res[:,j]; sub=np.random.default_rng(0).choice(r,4000,replace=False)
    print(f"{c:8s} {r.mean():8.4f} {r.std():8.4f} {stats.skew(r):7.2f} "
          f"{stats.kurtosis(r):8.1f} {stats.shapiro(sub).statistic:9.4f}")
print("\ntop features:")
for c in T: print(f"  {c:8s} "+", ".join(f"{k}={v:.2f}" for k,v in imp[c].items()))
