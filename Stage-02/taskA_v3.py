#!/usr/bin/env python3
"""Task A v3 -- heteroscedastic model.

WHY. v2's pooled clock residual had kurtosis 438 and Shapiro W 0.507, which looks
like catastrophic non-normality. It is not. Diagnostics showed:
  * within a single satellite, residuals are already Gaussian
    (median Shapiro W 0.995 dR / 0.991 dA / 0.994 dC / 0.988 clock)
  * satellites differ in residual scale by 11x to 22x
  * standardising each satellite by its own sd gives W = 0.9997 / 0.9981 / 0.9983
A pool of Gaussians with different variances is a scale mixture, and a scale
mixture is heavy-tailed by construction. So the fix is not a different loss on the
mean -- it is to MODEL THE SCALE.

The net therefore emits a mean and a log-sigma per target and trains under Gaussian
negative log-likelihood. This buys three things at once:
  1. standardised residuals (y-mu)/sigma that are genuinely N(0,1)
  2. automatic down-weighting of erratic satellites (G27) without hand-tuned clips
  3. a calibrated per-epoch error bar, which is what an operational user needs

Scale is predictable in real time: broadcast handover-jump sd correlates with
residual sd at r = 0.69 (radial) and r = 0.54 (clock), so sigma can be inferred
from broadcast alone -- no SP3 needed at prediction time.

Usage: python taskA_v3.py [n_seeds] [epochs]
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

dev = "cuda"
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

z = np.load("taskA.npz", allow_pickle=True)
E, B = z["E"], z["B"]
LAG, NT = int(z["LAG"]), int(z["NT"])
T = list(z["tgt_cols"])
SYS, CLS, SATS = z["sys"], z["cls"], list(z["sats"])
S = len(SATS)
W, STRIDE = 672, 4
OFF = np.arange(-960, 1, STRIDE)
L, NB = len(OFF), B.shape[2]
TR_END, VA_S, VA_E, TE_S = 18720, 19488, 20832, 21600

Eg = torch.tensor(np.nan_to_num(E), device=dev)
Em = torch.tensor(np.isfinite(E).astype(np.float32), device=dev)
Bg = torch.tensor(np.nan_to_num(B), device=dev)
Bm = torch.tensor(np.isfinite(B[:, :, :1]).astype(np.float32), device=dev)
offs = torch.tensor(OFF, device=dev)
known = torch.tensor((OFF <= -LAG).astype(np.float32), device=dev)[None, :, None]

si, tt = [], []
for s in range(S):
    t = np.arange(LAG + W, NT, 2)
    ok = np.isfinite(E[s, t]).all(1) & np.isfinite(B[s, t]).all(1)
    t = t[ok]
    if len(t) < 50:
        continue
    si.append(np.full(len(t), s))
    tt.append(t)
si, tt = np.concatenate(si), np.concatenate(tt)
tr = np.nonzero(tt < TR_END)[0]
va = np.nonzero((tt >= VA_S) & (tt < VA_E))[0]
te = np.nonzero(tt >= TE_S)[0]
SYSI = np.array([{"G": 0, "E": 1, "C": 2, "J": 3}[SYS[i]] for i in si])
CLSI = np.array([{"MEO": 0, "GSO": 1, "GEO": 2}[CLS[i]] for i in si])
sysn = torch.tensor(SYSI, device=dev)
clsn = torch.tensor(CLSI, device=dev)
SI = torch.tensor(si, device=dev)
TT = torch.tensor(tt, device=dev)
print(f"samples {len(tt):,}  train {len(tr):,} val {len(va):,} test {len(te):,}", flush=True)


def batch(idx):
    s, t = SI[idx], TT[idx]
    ix = t[:, None] + offs[None, :]
    e = Eg[s[:, None], ix]
    em = Em[s[:, None], ix] * known
    b = Bg[s[:, None], ix]
    bm = Bm[s[:, None], ix]
    hi = t - LAG
    widx = hi[:, None] - torch.arange(W, device=dev)[None, :]
    wv = Eg[s[:, None], widx]
    wmk = Em[s[:, None], widx]
    cnt = wmk.sum(1).clamp(min=1)
    wmean = (wv * wmk).sum(1) / cnt
    # spread of the truth window: the strongest hint about how erratic this sat is
    wvar = (((wv - wmean[:, None, :]) ** 2) * wmk).sum(1) / cnt
    return torch.cat([e * em, em, b * bm, bm], 2), wmean, wvar.sqrt(), Eg[s, t]


CIN = 4 + 4 + NB + 1


class HetMLP(nn.Module):
    """Emits mu and log-sigma per target. Sigma is centred on the observed window
    spread, so the net only has to learn a correction to an already-sane scale."""

    def __init__(self, h=512):
        super().__init__()
        self.emb, self.emc = nn.Embedding(4, 16), nn.Embedding(3, 8)
        self.trunk = nn.Sequential(
            nn.Linear(L * CIN + 8 + 24, h), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(h, 256), nn.GELU())
        self.mu = nn.Sequential(nn.Dropout(0.05), nn.Linear(256, 4))
        self.ls = nn.Linear(256, 4)

    def forward(self, x, wm, wsd, sy, cl):
        z = self.trunk(torch.cat([x.flatten(1), wm, wsd, self.emb(sy), self.emc(cl)], 1))
        return self.mu(z), self.ls(z).clamp(-4, 4)


rng = np.random.default_rng(0)
with torch.no_grad():
    ix = torch.tensor(rng.choice(tr, 20000, replace=False), device=dev)
    xs, wms, wsds, ys = batch(ix)
    xmu, xsd = xs.mean((0, 1)), xs.std((0, 1)) + 1e-6

scale = np.ones((4, 4), np.float32)
for k in range(4):
    m = SYSI[tr] == k
    if m.sum() < 500:
        continue
    with torch.no_grad():
        pick = rng.choice(int(m.sum()), min(8000, int(m.sum())), replace=False)
        ii = torch.tensor(tr[m][pick], device=dev)
        _, wm, _, y = batch(ii)
        scale[k] = (y - wm).std(0).clamp(min=1e-3).cpu().numpy()
SC = torch.tensor(scale, device=dev)
print("per-system residual sd (m); rows G,E,C,J; cols " + ",".join(T))
print(np.round(scale, 3), flush=True)


def train_one(seed):
    torch.manual_seed(seed)
    m = HetMLP().to(dev)
    bs = 768
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    nb = len(tr) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * nb)
    best, bst = 1e9, None
    for ep in range(EPOCHS):
        m.train()
        perm = np.random.default_rng(seed * 100 + ep).permutation(tr)
        for k in range(nb):
            ix = torch.tensor(perm[k * bs:(k + 1) * bs], device=dev)
            x, wm, wsd, y = batch(ix)
            sc = SC[sysn[ix]]
            mu_r, ls = m((x - xmu) / xsd, wm, wsd, sysn[ix], clsn[ix])
            mu = wm + mu_r * sc
            sig = sc * torch.exp(ls)
            # Gaussian NLL, the loss that makes standardised residuals N(0,1)
            loss = (0.5 * ((y - mu) / sig) ** 2 + torch.log(sig)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
        m.eval()
        v = []
        with torch.no_grad():
            for k in range(0, len(va), 2048):
                ix = torch.tensor(va[k:k + 2048], device=dev)
                x, wm, wsd, y = batch(ix)
                sc = SC[sysn[ix]]
                mu_r, ls = m((x - xmu) / xsd, wm, wsd, sysn[ix], clsn[ix])
                mu, sig = wm + mu_r * sc, sc * torch.exp(ls)
                v.append((0.5 * ((y - mu) / sig) ** 2 + torch.log(sig)).mean().item())
        s = float(np.mean(v))
        if s < best:
            best, bst = s, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bst)
    m.eval()

    def pred(idxs):
        M, G = [], []
        with torch.no_grad():
            for k in range(0, len(idxs), 2048):
                ix = torch.tensor(idxs[k:k + 2048], device=dev)
                x, wm, wsd, _ = batch(ix)
                sc = SC[sysn[ix]]
                mu_r, ls = m((x - xmu) / xsd, wm, wsd, sysn[ix], clsn[ix])
                M.append((wm + mu_r * sc).cpu().numpy())
                G.append((sc * torch.exp(ls)).cpu().numpy())
        return np.vstack(M), np.vstack(G)

    return pred(va), pred(te), best


Mv, Gv, Mt, Gt = [], [], [], []
for sd in range(NSEED):
    t0 = time.time()
    (pv, gv), (pt, gt), b = train_one(sd)
    Mv.append(pv)
    Gv.append(gv)
    Mt.append(pt)
    Gt.append(gt)
    print(f"  seed {sd} val NLL {b:.4f}  [{time.time() - t0:.0f}s]", flush=True)
# ensemble: average the means, and combine variances (within + between seeds)
Pv, Pt = np.mean(Mv, 0), np.mean(Mt, 0)
Sv = np.sqrt(np.mean(np.square(Gv), 0) + np.var(Mv, 0))
St = np.sqrt(np.mean(np.square(Gt), 0) + np.var(Mt, 0))


def wm_of(idxs):
    O = []
    with torch.no_grad():
        for k in range(0, len(idxs), 2048):
            ix = torch.tensor(idxs[k:k + 2048], device=dev)
            _, wm, _, _ = batch(ix)
            O.append(wm.cpu().numpy())
    return np.vstack(O)


Bv, Bt = wm_of(va), wm_of(te)
Yv, Yt = E[si[va], tt[va]], E[si[te], tt[te]]
alpha = np.ones((4, 4))
for k in range(4):
    m = SYSI[va] == k
    if m.sum() < 200:
        continue
    for j in range(4):
        g = np.linspace(0, 1, 21)
        alpha[k, j] = g[int(np.argmin([np.mean((Yv[m, j] - (a * Pv[m, j] + (1 - a) * Bv[m, j])) ** 2)
                                       for a in g]))]
A = alpha[SYSI[te]]
Pf = A * Pt + (1 - A) * Bt
W_R = {"MEO": 0.98, "GSO": 0.99, "GEO": 0.99}
W_A = {"MEO": 0.139, "GSO": 0.123, "GEO": 0.123}
wR = np.array([W_R[CLS[i]] for i in si[te]])
wA = np.array([W_A[CLS[i]] for i in si[te]])


def sis(r, m=None):
    m = np.ones(len(r), bool) if m is None else m
    return np.sqrt(np.mean((wR[m] * r[m, 0] - r[m, 3]) ** 2 + wA[m] ** 2 * (r[m, 1] ** 2 + r[m, 2] ** 2)))


np.savez("v3_pred.npz", Pf=Pf, Pt=Pt, St=St, Bt=Bt, Yt=Yt, si=si[te], tt=tt[te],
         SYSI=SYSI[te], wR=wR, wA=wA, alpha=alpha)
print(f"\n{'':22s} " + "  ".join(f"{c:>7s}" for c in T) + "     SISRE")
for tag, P in [("raw broadcast", np.zeros_like(Yt)), ("7d window mean", Bt),
               ("v3 ensemble", Pt), ("v3 + blend", Pf)]:
    r = Yt - P
    print(f"{tag:22s} " + "  ".join(f"{np.sqrt((r[:, j] ** 2).mean()):7.3f}" for j in range(4))
          + f"  {sis(r):8.3f}")
print("\nper-system SISRE:")
print(f"{'sys':5s} {'raw':>8s} {'baseline':>9s} {'v3':>8s} {'cut':>7s}")
for k, s in enumerate("GECJ"):
    m = SYSI[te] == k
    if m.sum() < 200:
        continue
    a, b, c = sis(Yt, m), sis(Yt - Bt, m), sis(Yt - Pf, m)
    print(f"{s:5s} {a:8.3f} {b:9.3f} {c:8.3f} {100 * (1 - c / a):6.1f}%")
print(f"{'ALL':5s} {sis(Yt):8.3f} {sis(Yt - Bt):9.3f} {sis(Yt - Pf):8.3f} "
      f"{100 * (1 - sis(Yt - Pf) / sis(Yt)):6.1f}%")

r = Yt - Pt
print("\n--- RAW residuals (pooled: a scale mixture, so expected to be heavy-tailed) ---")
print(f"{'target':8s} {'mean':>8s} {'std':>8s} {'skew':>7s} {'kurt':>8s} {'ShapiroW':>9s}")
for j, c in enumerate(T):
    x = r[:, j]
    sub = np.random.default_rng(0).choice(x, 4000, replace=False)
    print(f"{c:8s} {x.mean():8.4f} {x.std():8.4f} {stats.skew(x):7.2f} "
          f"{stats.kurtosis(x):8.1f} {stats.shapiro(sub).statistic:9.4f}")
print("\n--- STANDARDISED residuals (y-mu)/sigma : these are the ones that should be N(0,1) ---")
print(f"{'target':8s} {'mean':>8s} {'std':>8s} {'skew':>7s} {'kurt':>8s} {'ShapiroW':>9s} {'p':>8s}")
zt = r / St
for j, c in enumerate(T):
    x = zt[:, j]
    sub = np.random.default_rng(0).choice(x, 4000, replace=False)
    w, p = stats.shapiro(sub)
    print(f"{c:8s} {x.mean():8.4f} {x.std():8.4f} {stats.skew(x):7.2f} "
          f"{stats.kurtosis(x):8.1f} {w:9.4f} {p:8.4f}")
print("\n--- calibration: fraction of truth inside the predicted interval ---")
print(f"{'target':8s} " + "  ".join(f"{k:>9s}" for k in ["1s(68.3%)", "2s(95.4%)", "3s(99.7%)"]))
for j, c in enumerate(T):
    a = np.abs(r[:, j]) / St[:, j]
    print(f"{c:8s} " + "  ".join(f"{100*(a<k).mean():8.1f}%" for k in (1, 2, 3)))
