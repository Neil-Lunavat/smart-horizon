#!/usr/bin/env python3
"""Task A v2 -- fixes the three flaws diagnosed in v1.

 1. SCALE DOMINATION. v1 pooled all systems under plain MSE. BeiDou clock error
    is 12 m and Galileo's is 0.38 m, so BeiDou contributed ~1000x the squared
    error and Galileo was optimised as if it were noise: v1 made Galileo clock
    WORSE than doing nothing (0.383 -> 0.459 m). Fix: divide each target by its
    per-system residual sd inside the loss, so every system counts equally.
 2. HEAVY TAILS. v1 residual kurtosis reached 110 on dR. Fix: Huber loss.
 3. NO BASELINE FLOOR. The plain 7-day window mean beat v1 on clock. Fix: fit a
    per-system, per-target blend weight against that baseline on validation, so
    the delivered model can never be worse than doing nothing.
Plus seed ensembling, which lowers RMS and tightens the tails.

Usage: python taskA_v2.py [n_seeds] [epochs] [arch]
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

dev = "cuda"
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 4
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ARCH = sys.argv[3] if len(sys.argv) > 3 else "mlp"

z = np.load("taskA.npz", allow_pickle=True)
E, B = z["E"], z["B"]
LAG, NT = int(z["LAG"]), int(z["NT"])
T = list(z["tgt_cols"])
SYS, CLS = z["sys"], z["cls"]
S = len(z["sats"])
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
    wmean = (wv * wmk).sum(1) / wmk.sum(1).clamp(min=1)
    return torch.cat([e * em, em, b * bm, bm], 2), wmean, Eg[s, t]


CIN = 4 + 4 + NB + 1


class MLP(nn.Module):
    def __init__(self, h=512):
        super().__init__()
        self.emb, self.emc = nn.Embedding(4, 16), nn.Embedding(3, 8)
        self.net = nn.Sequential(
            nn.Linear(L * CIN + 4 + 24, h), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(h, 256), nn.GELU(), nn.Dropout(0.05), nn.Linear(256, 4))

    def forward(self, x, wm, sy, cl):
        return self.net(torch.cat([x.flatten(1), wm, self.emb(sy), self.emc(cl)], 1))


class TCN(nn.Module):
    """Dilated causal convolutions -- won on along/cross-track in the v1 ladder."""

    def __init__(self, h=128):
        super().__init__()
        self.inp = nn.Conv1d(CIN, h, 1)
        ls = []
        for d in (1, 2, 4, 8, 16, 32, 64):
            ls += [nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.GELU(), nn.BatchNorm1d(h)]
        self.body = nn.Sequential(*ls)
        self.emb, self.emc = nn.Embedding(4, 16), nn.Embedding(3, 8)
        self.head = nn.Sequential(nn.Linear(h + 16 + 8 + 4, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, x, wm, sy, cl):
        o = self.body(self.inp(x.transpose(1, 2)))[:, :, -1]
        return self.head(torch.cat([o, self.emb(sy), self.emc(cl), wm], 1))


rng = np.random.default_rng(0)
with torch.no_grad():
    ix = torch.tensor(rng.choice(tr, 20000, replace=False), device=dev)
    xs, wms, ys = batch(ix)
    xmu, xsd = xs.mean((0, 1)), xs.std((0, 1)) + 1e-6

# FIX 1 -- per-system residual scale, so the loss weights every system equally
scale = np.ones((4, 4), np.float32)
for k in range(4):
    m = SYSI[tr] == k
    if m.sum() < 500:
        continue
    with torch.no_grad():
        pick = rng.choice(m.sum(), min(8000, int(m.sum())), replace=False)
        ii = torch.tensor(tr[m][pick], device=dev)
        _, wm, y = batch(ii)
        scale[k] = (y - wm).std(0).clamp(min=1e-3).cpu().numpy()
print("per-system residual sd (m); rows G,E,C,J; cols " + ",".join(T))
print(np.round(scale, 3), flush=True)
SC = torch.tensor(scale, device=dev)


def train_one(seed):
    torch.manual_seed(seed)
    m = (MLP() if ARCH == "mlp" else TCN()).to(dev)
    bs = 768 if ARCH == "mlp" else 256
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    nb = len(tr) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * nb)
    best, bst = 1e9, None
    for ep in range(EPOCHS):
        m.train()
        perm = np.random.default_rng(seed * 100 + ep).permutation(tr)
        for k in range(nb):
            ix = torch.tensor(perm[k * bs:(k + 1) * bs], device=dev)
            x, wm, y = batch(ix)
            sc = SC[sysn[ix]]
            p = wm + m((x - xmu) / xsd, wm, sysn[ix], clsn[ix]) * sc
            loss = nn.functional.huber_loss(p / sc, y / sc, delta=1.5)  # FIX 2
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
                x, wm, y = batch(ix)
                sc = SC[sysn[ix]]
                p = wm + m((x - xmu) / xsd, wm, sysn[ix], clsn[ix]) * sc
                v.append((((p - y) / sc) ** 2).mean().item())
        s = float(np.mean(v))
        if s < best:
            best, bst = s, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bst)
    m.eval()

    def pred(idxs):
        P = []
        with torch.no_grad():
            for k in range(0, len(idxs), 2048):
                ix = torch.tensor(idxs[k:k + 2048], device=dev)
                x, wm, _ = batch(ix)
                sc = SC[sysn[ix]]
                P.append((wm + m((x - xmu) / xsd, wm, sysn[ix], clsn[ix]) * sc).cpu().numpy())
        return np.vstack(P)

    return pred(va), pred(te), best


Pv, Pt = [], []
for sd in range(NSEED):
    t0 = time.time()
    pv, pt, b = train_one(sd)
    Pv.append(pv)
    Pt.append(pt)
    print(f"  seed {sd} val {b:.4f}  [{time.time() - t0:.0f}s]", flush=True)
Pv, Pt = np.mean(Pv, 0), np.mean(Pt, 0)


def wmean_of(idxs):
    O = []
    with torch.no_grad():
        for k in range(0, len(idxs), 2048):
            ix = torch.tensor(idxs[k:k + 2048], device=dev)
            _, wm, _ = batch(ix)
            O.append(wm.cpu().numpy())
    return np.vstack(O)


Bv, Bt = wmean_of(va), wmean_of(te)
Yv, Yt = E[si[va], tt[va]], E[si[te], tt[te]]

# FIX 3 -- per-system, per-target blend against the baseline, fitted on validation
alpha = np.ones((4, 4))
for k in range(4):
    m = SYSI[va] == k
    if m.sum() < 200:
        continue
    for j in range(4):
        g = np.linspace(0, 1, 21)
        err = [np.mean((Yv[m, j] - (a * Pv[m, j] + (1 - a) * Bv[m, j])) ** 2) for a in g]
        alpha[k, j] = g[int(np.argmin(err))]
print("\nblend weight on the model (1.0 = pure model, 0.0 = pure baseline):")
print("      " + "  ".join(f"{c:>8s}" for c in T))
for k, s in enumerate("GECJ"):
    print(f"  {s}   " + "  ".join(f"{alpha[k, j]:8.2f}" for j in range(4)))

A = alpha[SYSI[te]]
Pf = A * Pt + (1 - A) * Bt
W_R = {"MEO": 0.98, "GSO": 0.99, "GEO": 0.99}
W_A = {"MEO": 0.139, "GSO": 0.123, "GEO": 0.123}
wR = np.array([W_R[CLS[i]] for i in si[te]])
wA = np.array([W_A[CLS[i]] for i in si[te]])


def sis(r, m=None):
    m = np.ones(len(r), bool) if m is None else m
    return np.sqrt(np.mean((wR[m] * r[m, 0] - r[m, 3]) ** 2 + wA[m] ** 2 * (r[m, 1] ** 2 + r[m, 2] ** 2)))


np.savez(f"v2_pred_{ARCH}.npz", Pf=Pf, Pt=Pt, Bt=Bt, Yt=Yt, si=si[te], tt=tt[te],
         SYSI=SYSI[te], wR=wR, wA=wA, alpha=alpha)
print(f"\n{'':22s} " + "  ".join(f"{c:>7s}" for c in T) + "     SISRE")
for tag, P in [("raw broadcast", np.zeros_like(Yt)), ("7d window mean", Bt),
               (f"v2 {ARCH} ensemble", Pt), (f"v2 {ARCH} + blend", Pf)]:
    r = Yt - P
    print(f"{tag:22s} " + "  ".join(f"{np.sqrt((r[:, j] ** 2).mean()):7.3f}" for j in range(4))
          + f"  {sis(r):8.3f}")
print("\nper-system SISRE:")
print(f"{'sys':5s} {'raw':>8s} {'baseline':>9s} {'v2+blend':>9s} {'cut':>7s}")
for k, s in enumerate("GECJ"):
    m = SYSI[te] == k
    if m.sum() < 200:
        continue
    a, b, c = sis(Yt, m), sis(Yt - Bt, m), sis(Yt - Pf, m)
    print(f"{s:5s} {a:8.3f} {b:9.3f} {c:9.3f} {100 * (1 - c / a):6.1f}%")
print(f"{'ALL':5s} {sis(Yt):8.3f} {sis(Yt - Bt):9.3f} {sis(Yt - Pf):9.3f} "
      f"{100 * (1 - sis(Yt - Pf) / sis(Yt)):6.1f}%")
r = Yt - Pf
print("\nresidual distribution (test):")
print(f"{'target':8s} {'mean':>8s} {'std':>8s} {'skew':>7s} {'kurt':>8s} {'ShapiroW':>9s} {'p':>8s}")
for j, c in enumerate(T):
    x = r[:, j]
    sub = np.random.default_rng(0).choice(x, 4000, replace=False)
    w, p = stats.shapiro(sub)
    print(f"{c:8s} {x.mean():8.4f} {x.std():8.4f} {stats.skew(x):7.2f} "
          f"{stats.kurtosis(x):8.1f} {w:9.4f} {p:8.4f}")
