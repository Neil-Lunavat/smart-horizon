#!/usr/bin/env python3
"""Task A v4 -- Student-t predictive distribution + PIT diagnostics.

v3 (Gaussian NLL, per-epoch sigma) already gave standardised residuals that are
essentially normal for the geometry components:
    dR W=0.9895   dA W=0.9888   dC W=0.9981
but the clock stayed at W=0.8531 with kurtosis 23. Sigma alone cannot fix that,
because satellite clocks genuinely jump -- G27 swings +-3 m day to day and no
error bar predicts the sign of a discrete event.

The honest model for "mostly quiet, occasionally jumps" is a heavier-tailed
predictive law, so v4 uses Student-t with a learned degrees-of-freedom per target.
Correctness is then judged by the probability integral transform: if the predictive
distribution is right, u = F(y) is Uniform(0,1) and z = Phi^-1(u) is exactly N(0,1).
PIT normal scores are the standard calibration check in probabilistic forecasting
and are the strongest available statement that the residuals are Gaussian.

Three views are reported, because they answer different questions:
  RAW (metres)   -- what a user actually experiences. A scale mixture, so heavy.
  STANDARDISED   -- residual / sigma. Tests whether the scale is right.
  PIT z-scores   -- tests whether the WHOLE predictive distribution is right.

Usage: python taskA_v4.py [n_seeds] [epochs]
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

dev = "cuda"
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12

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
    wvar = (((wv - wmean[:, None, :]) ** 2) * wmk).sum(1) / cnt
    # recent volatility (last day of the truth window) -- the clock-jump tell
    rv = wv[:, :96]
    rm = wmk[:, :96]
    rc = rm.sum(1).clamp(min=1)
    rmean = (rv * rm).sum(1) / rc
    rvar = (((rv - rmean[:, None, :]) ** 2) * rm).sum(1) / rc
    return (torch.cat([e * em, em, b * bm, bm], 2), wmean,
            torch.cat([wvar.sqrt(), rvar.sqrt()], 1), Eg[s, t])


CIN = 4 + 4 + NB + 1


class StudentMLP(nn.Module):
    def __init__(self, h=512):
        super().__init__()
        self.emb, self.emc = nn.Embedding(4, 16), nn.Embedding(3, 8)
        self.trunk = nn.Sequential(
            nn.Linear(L * CIN + 4 + 8 + 24, h), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(h, 256), nn.GELU())
        self.mu = nn.Sequential(nn.Dropout(0.05), nn.Linear(256, 4))
        self.ls = nn.Linear(256, 4)
        self.lnu = nn.Parameter(torch.full((4,), 1.6))   # log(nu-2), learned per target

    def forward(self, x, wm, sp, sy, cl):
        z = self.trunk(torch.cat([x.flatten(1), wm, sp, self.emb(sy), self.emc(cl)], 1))
        return self.mu(z), self.ls(z).clamp(-4, 4), 2.0 + torch.exp(self.lnu).clamp(0.05, 200.0)


rng = np.random.default_rng(0)
with torch.no_grad():
    ix = torch.tensor(rng.choice(tr, 20000, replace=False), device=dev)
    xs, wms, sps, ys = batch(ix)
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


def student_nll(y, mu, sig, nu):
    zz = (y - mu) / sig
    return (torch.log(sig) - torch.lgamma((nu + 1) / 2) + torch.lgamma(nu / 2)
            + 0.5 * torch.log(np.pi * nu) + (nu + 1) / 2 * torch.log1p(zz ** 2 / nu)).mean()


def train_one(seed):
    torch.manual_seed(seed)
    m = StudentMLP().to(dev)
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
            x, wm, sp, y = batch(ix)
            sc = SC[sysn[ix]]
            mu_r, ls, nu = m((x - xmu) / xsd, wm, sp, sysn[ix], clsn[ix])
            loss = student_nll(y, wm + mu_r * sc, sc * torch.exp(ls), nu)
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
                x, wm, sp, y = batch(ix)
                sc = SC[sysn[ix]]
                mu_r, ls, nu = m((x - xmu) / xsd, wm, sp, sysn[ix], clsn[ix])
                v.append(student_nll(y, wm + mu_r * sc, sc * torch.exp(ls), nu).item())
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
                x, wm, sp, _ = batch(ix)
                sc = SC[sysn[ix]]
                mu_r, ls, nu = m((x - xmu) / xsd, wm, sp, sysn[ix], clsn[ix])
                M.append((wm + mu_r * sc).cpu().numpy())
                G.append((sc * torch.exp(ls)).cpu().numpy())
        return np.vstack(M), np.vstack(G)

    with torch.no_grad():
        nu_v = (2.0 + torch.exp(m.lnu).clamp(0.05, 200.0)).cpu().numpy()
    return pred(va), pred(te), best, nu_v


Mv, Gv, Mt, Gt, NUs = [], [], [], [], []
for sd in range(NSEED):
    t0 = time.time()
    (pv, gv), (pt, gt), b, nu = train_one(sd)
    Mv.append(pv); Gv.append(gv); Mt.append(pt); Gt.append(gt); NUs.append(nu)
    print(f"  seed {sd} val NLL {b:.4f}  nu={np.round(nu,1)}  [{time.time()-t0:.0f}s]", flush=True)
Pv, Pt = np.mean(Mv, 0), np.mean(Mt, 0)
Sv = np.sqrt(np.mean(np.square(Gv), 0) + np.var(Mv, 0))
St = np.sqrt(np.mean(np.square(Gt), 0) + np.var(Mt, 0))
NU = np.mean(NUs, 0)
print("learned degrees of freedom:", dict(zip(T, np.round(NU, 1))))


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


np.savez("v4_pred.npz", Pf=Pf, Pt=Pt, St=St, Bt=Bt, Yt=Yt, si=si[te], tt=tt[te],
         SYSI=SYSI[te], wR=wR, wA=wA, alpha=alpha, NU=NU)
print(f"\n{'':22s} " + "  ".join(f"{c:>7s}" for c in T) + "     SISRE")
for tag, P in [("raw broadcast", np.zeros_like(Yt)), ("7d window mean", Bt),
               ("v4 ensemble", Pt), ("v4 + blend", Pf)]:
    r = Yt - P
    print(f"{tag:22s} " + "  ".join(f"{np.sqrt((r[:, j] ** 2).mean()):7.3f}" for j in range(4))
          + f"  {sis(r):8.3f}")
print("\nper-system SISRE:")
print(f"{'sys':5s} {'raw':>8s} {'baseline':>9s} {'v4':>8s} {'cut':>7s}")
for k, s in enumerate("GECJ"):
    m = SYSI[te] == k
    if m.sum() < 200:
        continue
    a, b, c = sis(Yt, m), sis(Yt - Bt, m), sis(Yt - Pf, m)
    print(f"{s:5s} {a:8.3f} {b:9.3f} {c:8.3f} {100 * (1 - c / a):6.1f}%")
print(f"{'ALL':5s} {sis(Yt):8.3f} {sis(Yt - Bt):9.3f} {sis(Yt - Pf):8.3f} "
      f"{100 * (1 - sis(Yt - Pf) / sis(Yt)):6.1f}%")

r = Yt - Pt
rng2 = np.random.default_rng(0)


def blk(tag, M):
    print(f"\n--- {tag} ---")
    print(f"{'target':8s} {'mean':>8s} {'std':>8s} {'skew':>7s} {'kurt':>8s} {'ShapiroW':>9s} {'p':>8s}")
    for j, c in enumerate(T):
        x = M[:, j]
        x = x[np.isfinite(x)]
        sub = rng2.choice(x, min(4000, len(x)), replace=False)
        w, p = stats.shapiro(sub)
        print(f"{c:8s} {x.mean():8.4f} {x.std():8.4f} {stats.skew(x):7.2f} "
              f"{stats.kurtosis(x):8.1f} {w:9.4f} {p:8.4f}")


blk("RAW residuals, metres (scale mixture -- heavy by construction)", r)
blk("STANDARDISED residuals (y-mu)/sigma", r / St)
# PIT: u = F_t(z; nu) then z = Phi^-1(u). N(0,1) iff the predictive law is correct.
U = np.column_stack([stats.t.cdf(r[:, j] / St[:, j], df=NU[j]) for j in range(4)])
U = np.clip(U, 1e-6, 1 - 1e-6)
blk("PIT NORMAL SCORES  Phi^-1(F(y))  <- the calibration test", stats.norm.ppf(U))
print("\n--- PIT uniformity (Kolmogorov-Smirnov vs Uniform(0,1)) ---")
for j, c in enumerate(T):
    ks = stats.kstest(U[:, j], "uniform")
    print(f"  {c:8s} KS D={ks.statistic:.4f}")
print("\n--- interval coverage from the Student-t predictive law ---")
print(f"{'target':8s} " + "  ".join(f"{k:>10s}" for k in ["50%", "68.3%", "95.4%", "99.7%"]))
for j, c in enumerate(T):
    row = []
    for q in (0.50, 0.683, 0.954, 0.997):
        hw = stats.t.ppf(0.5 + q / 2, df=NU[j]) * St[:, j]
        row.append(f"{100*(np.abs(r[:,j])<hw).mean():9.1f}%")
    print(f"{c:8s} " + "  ".join(row))
