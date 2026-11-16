#!/usr/bin/env python3
"""Task A model ladder, part 2: neural sequence models on the RAW series.

Unlike the tabular models these get no hand-built summaries. Each sample is the
strip t-960..t at stride 4 (241 steps), with 21 broadcast channels available at
every step and the 4 truth channels zeroed+masked after t-LAG. The net must work
out for itself how to bridge the 3-day gap.

Residual head: output = (7-day window mean of truth) + net(...). The constant is
handed over explicitly so capacity goes to the part that actually varies.
"""
import numpy as np, torch, torch.nn as nn, time, sys
from scipy import stats
dev="cuda"
z=np.load("taskA.npz",allow_pickle=True)
E,B=z["E"],z["B"]; LAG=int(z["LAG"]); NT=int(z["NT"])
T=list(z["tgt_cols"]); sats=z["sats"]; SYS=z["sys"]; CLS=z["cls"]; S=len(sats)
W=672; STRIDE=4; OFF=np.arange(-960,1,STRIDE); L=len(OFF); NB=B.shape[2]
TR_END,VA_S,VA_E,TE_S=18720,19488,20832,21600

Eg=torch.tensor(np.nan_to_num(E),device=dev)
Em=torch.tensor(np.isfinite(E).astype(np.float32),device=dev)
Bg=torch.tensor(np.nan_to_num(B),device=dev)
Bm=torch.tensor(np.isfinite(B[:,:,:1]).astype(np.float32),device=dev)
offs=torch.tensor(OFF,device=dev)
known=torch.tensor((OFF<=-LAG).astype(np.float32),device=dev)[None,:,None]

def samples(stride):
    si,tt=[],[]
    for s in range(S):
        t=np.arange(LAG+W,NT,stride)
        ok=np.isfinite(E[s,t]).all(1)&np.isfinite(B[s,t]).all(1)
        t=t[ok]
        if len(t)<50: continue
        si.append(np.full(len(t),s)); tt.append(t)
    return np.concatenate(si),np.concatenate(tt)
si,tt=samples(2)
tr=np.nonzero(tt<TR_END)[0]; va=np.nonzero((tt>=VA_S)&(tt<VA_E))[0]; te=np.nonzero(tt>=TE_S)[0]
print(f"samples {len(tt):,}  train {len(tr):,} val {len(va):,} test {len(te):,}  seq len {L}",flush=True)
sysn=torch.tensor(np.array([{"G":0,"E":1,"C":2,"J":3}[SYS[i]] for i in si]),device=dev)
clsn=torch.tensor(np.array([{"MEO":0,"GSO":1,"GEO":2}[CLS[i]] for i in si]),device=dev)
SI=torch.tensor(si,device=dev); TT=torch.tensor(tt,device=dev)

def batch(idx):
    s=SI[idx]; t=TT[idx]
    ix=t[:,None]+offs[None,:]                                   # (b,L)
    e=Eg[s[:,None],ix]; em=Em[s[:,None],ix]*known               # truth, masked after lag
    b=Bg[s[:,None],ix]; bm=Bm[s[:,None],ix]
    hi=t-LAG
    widx=hi[:,None]-torch.arange(W,device=dev)[None,:]
    wv=Eg[s[:,None],widx]; wmk=Em[s[:,None],widx]
    wmean=(wv*wmk).sum(1)/wmk.sum(1).clamp(min=1)               # (b,4) explicit constant
    x=torch.cat([e*em,em,b*bm,bm],dim=2)                        # (b,L,4+4+NB+1)
    y=Eg[s,t]
    return x,wmean,y,s

CIN=4+4+NB+1
class GRUNet(nn.Module):
    def __init__(s,h=128):
        super().__init__()
        s.inp=nn.Linear(CIN,h); s.gru=nn.GRU(h,h,2,batch_first=True,dropout=0.1)
        s.emb=nn.Embedding(4,16); s.emc=nn.Embedding(3,8)
        s.head=nn.Sequential(nn.Linear(h+16+8+4,128),nn.GELU(),nn.Linear(128,4))
    def forward(s,x,wm,sy,cl):
        o,_=s.gru(torch.relu(s.inp(x)))
        return s.head(torch.cat([o[:,-1],s.emb(sy),s.emc(cl),wm],1))
class TCN(nn.Module):
    def __init__(s,h=128):
        super().__init__()
        s.inp=nn.Conv1d(CIN,h,1); ls=[]
        for d in (1,2,4,8,16,32,64):
            ls+= [nn.Conv1d(h,h,3,padding=d,dilation=d),nn.GELU(),nn.BatchNorm1d(h)]
        s.body=nn.Sequential(*ls)
        s.emb=nn.Embedding(4,16); s.emc=nn.Embedding(3,8)
        s.head=nn.Sequential(nn.Linear(h+16+8+4,128),nn.GELU(),nn.Linear(128,4))
    def forward(s,x,wm,sy,cl):
        o=s.body(s.inp(x.transpose(1,2)))[:,:,-1]
        return s.head(torch.cat([o,s.emb(sy),s.emc(cl),wm],1))
class MLP(nn.Module):
    """Control: same information, no temporal structure -- sequence flattened."""
    def __init__(s,h=512):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(L*CIN+4,h),nn.GELU(),nn.Dropout(0.1),
                            nn.Linear(h,256),nn.GELU(),nn.Linear(256,4))
    def forward(s,x,wm,sy,cl): return s.net(torch.cat([x.flatten(1),wm],1))

with torch.no_grad():
    ix=torch.tensor(np.random.default_rng(0).choice(tr,20000,replace=False),device=dev)
    xs,wms,ys,_=batch(ix)
    xmu,xsd=xs.mean((0,1)),xs.std((0,1))+1e-6
    ysd=(ys-wms).std(0)+1e-6
print("target residual sd (m):",[f"{v:.3f}" for v in ysd.tolist()],flush=True)

def run(name,model,epochs=6,bs=256,lr=2e-3):
    m=model.to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    nb=len(tr)//bs
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=epochs*nb)
    best=1e9; bstate=None
    for ep in range(epochs):
        m.train(); perm=np.random.default_rng(ep).permutation(tr); t0=time.time()
        for k in range(nb):
            ix=torch.tensor(perm[k*bs:(k+1)*bs],device=dev)
            x,wm,y,_=batch(ix); x=(x-xmu)/xsd
            p=wm+m(x,wm,sysn[ix],clsn[ix])*ysd
            loss=nn.functional.huber_loss(p/ysd,y/ysd,delta=2.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
        m.eval(); v=[]
        with torch.no_grad():
            for k in range(0,len(va),512):
                ix=torch.tensor(va[k:k+512],device=dev)
                x,wm,y,_=batch(ix); x=(x-xmu)/xsd
                v.append(((wm+m(x,wm,sysn[ix],clsn[ix])*ysd-y)**2).mean(0))
        vm=torch.stack(v).mean(0).sqrt()
        sc=vm.mean().item()
        if sc<best: best=sc; bstate={k:v.clone() for k,v in m.state_dict().items()}
        print(f"  {name} ep{ep} val RMS "+" ".join(f"{c}={x:.3f}" for c,x in zip(T,vm.tolist()))
              +f"  [{time.time()-t0:.0f}s]",flush=True)
    m.load_state_dict(bstate); m.eval()
    P=[]
    with torch.no_grad():
        for k in range(0,len(te),512):
            ix=torch.tensor(te[k:k+512],device=dev)
            x,wm,y,_=batch(ix); x=(x-xmu)/xsd
            P.append((wm+m(x,wm,sysn[ix],clsn[ix])*ysd).cpu().numpy())
    return np.vstack(P)

Yte=E[si[te],tt[te]]
W_R={"MEO":0.98,"GSO":0.99,"GEO":0.99}; W_A={"MEO":0.139,"GSO":0.123,"GEO":0.123}
wR=np.array([W_R[CLS[i]] for i in si[te]]); wA=np.array([W_A[CLS[i]] for i in si[te]])
out={}
for nm,mk in [("GRU",GRUNet),("TCN",TCN)]:
    print(f"\n=== {nm}",flush=True)
    P=run(nm,mk()); out[nm]=P
    r=Yte-P
    ss=np.sqrt(np.mean((wR*P[:,0]-P[:,3])**2*0+ (wR*(Yte[:,0]-r[:,0])-(Yte[:,3]-r[:,3]))**2*0
        + (wR*r[:,0]-r[:,3])**2 + wA**2*(r[:,1]**2+r[:,2]**2)))
    print(f"  {nm} TEST  "+"  ".join(f"{c}={np.sqrt((r[:,j]**2).mean()):.3f}" for j,c in enumerate(T))
          +f"   SISRE={ss:.3f}",flush=True)
np.savez("nn_pred2.npz",Yte=Yte,si=si[te],tt=tt[te],**out)
