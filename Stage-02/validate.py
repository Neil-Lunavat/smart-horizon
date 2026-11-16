#!/usr/bin/env python3
"""Task A: does the result survive honest validation? No cherry-picking.

Two modes, sharing one model and one data-loading path.

  --mode folds   (default)
    TEST 1  RANDOM 80/20 vs TIME SPLIT. Samples are taken every 2 steps (30 min)
            but each carries a 672-step (7 day) history window, so neighbouring
            samples share ~99.7% of their input. A random split therefore puts
            near-duplicates of test rows into training. Run both and print the
            gap: if random looks much better than time, it is leakage, not skill.
    TEST 2  ROLLING-ORIGIN FOLDS. One 8-day test window is thin. Retrain from
            scratch on 5 origins spread across the corpus and report every fold,
            so a lucky window cannot hide.
    TEST 3  SATELLITE HOLDOUT, unseen spacecraft x unseen future, plus the
            train-vs-test overfit gap on every fold.

  --mode mccv
    BLOCKED MONTE-CARLO CROSS-VALIDATION -- the definitive number. 24 random
    contiguous 2-week test blocks, model retrained from scratch per block,
    training data strictly before the block with an 8-day gap so no window
    straddles a boundary. Writes mccv.json.

This file was previously two scripts, validate.py and mccv.py, which were
byte-identical for their first 183 lines; mccv.py's docstring was a stale copy of
validate.py's and described experiments it did not run.

Usage: python validate.py [folds|mccv|all] [epochs]
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

dev = "cuda"
_args = [a for a in sys.argv[1:]]
MODE = _args[0] if _args and not _args[0].isdigit() else "folds"
_ep = [a for a in _args if a.isdigit()]
EPOCHS = int(_ep[0]) if _ep else 8
if MODE not in ("folds", "mccv", "all"):
    raise SystemExit(f"unknown mode {MODE!r}; expected folds, mccv or all")
z = np.load("taskA.npz", allow_pickle=True)
E, B = z["E"], z["B"]
LAG, NT = int(z["LAG"]), int(z["NT"])
T = list(z["tgt_cols"])
SYS, CLS, SATS = z["sys"], z["cls"], list(z["sats"])
S = len(SATS)
W, STRIDE = 672, 4
OFF = np.arange(-960, 1, STRIDE)
L, NB = len(OFF), B.shape[2]

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
SYSI = np.array([{"G": 0, "E": 1, "C": 2, "J": 3}[SYS[i]] for i in si])
CLSI = np.array([{"MEO": 0, "GSO": 1, "GEO": 2}[CLS[i]] for i in si])
sysn = torch.tensor(SYSI, device=dev)
clsn = torch.tensor(CLSI, device=dev)
SI = torch.tensor(si, device=dev)
TT = torch.tensor(tt, device=dev)
W_R = {"MEO": 0.98, "GSO": 0.99, "GEO": 0.99}
W_A = {"MEO": 0.139, "GSO": 0.123, "GEO": 0.123}
wRa = np.array([W_R[CLS[i]] for i in si])
wAa = np.array([W_A[CLS[i]] for i in si])
print(f"total samples {len(tt):,}  span steps {tt.min()}..{tt.max()}", flush=True)


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
    return torch.cat([e * em, em, b * bm, bm], 2), wmean, wvar.sqrt(), Eg[s, t]


CIN = 4 + 4 + NB + 1


class Net(nn.Module):
    def __init__(self, h=512):
        super().__init__()
        self.emb, self.emc = nn.Embedding(4, 16), nn.Embedding(3, 8)
        self.trunk = nn.Sequential(nn.Linear(L * CIN + 8 + 24, h), nn.GELU(),
                                   nn.Dropout(0.10), nn.Linear(h, 256), nn.GELU())
        self.mu = nn.Sequential(nn.Dropout(0.05), nn.Linear(256, 4))
        self.ls = nn.Linear(256, 4)

    def forward(self, x, wm, wsd, sy, cl):
        z = self.trunk(torch.cat([x.flatten(1), wm, wsd, self.emb(sy), self.emc(cl)], 1))
        return self.mu(z), self.ls(z).clamp(-4, 4)


def sisre(r, m):
    return np.sqrt(np.mean((wRa[m] * r[:, 0] - r[:, 3]) ** 2
                           + wAa[m] ** 2 * (r[:, 1] ** 2 + r[:, 2] ** 2)))


def fit_eval(tr, va, teidx, tag, seed=0):
    """Train on tr, early-stop on va, report on te AND on a train subsample."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        ix = torch.tensor(rng.choice(tr, min(20000, len(tr)), replace=False), device=dev)
        xs, _, _, _ = batch(ix)
        xmu, xsd = xs.mean((0, 1)), xs.std((0, 1)) + 1e-6
    scale = np.ones((4, 4), np.float32)
    for k in range(4):
        m = SYSI[tr] == k
        if m.sum() < 500:
            continue
        with torch.no_grad():
            pick = rng.choice(int(m.sum()), min(8000, int(m.sum())), replace=False)
            _, wm, _, y = batch(torch.tensor(tr[m][pick], device=dev))
            scale[k] = (y - wm).std(0).clamp(min=1e-3).cpu().numpy()
    SC = torch.tensor(scale, device=dev)
    m = Net().to(dev)
    bs = 768
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    nb = max(len(tr) // bs, 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * nb)
    best, bst = 1e9, None
    for ep in range(EPOCHS):
        m.train()
        perm = np.random.default_rng(seed * 100 + ep).permutation(tr)
        for k in range(nb):
            ix = torch.tensor(perm[k * bs:(k + 1) * bs], device=dev)
            if len(ix) < 8:
                continue
            x, wm, wsd, y = batch(ix)
            sc = SC[sysn[ix]]
            mu_r, ls = m((x - xmu) / xsd, wm, wsd, sysn[ix], clsn[ix])
            mu, sig = wm + mu_r * sc, sc * torch.exp(ls)
            loss = (0.5 * ((y - mu) / sig) ** 2 + torch.log(sig)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval(); v = []
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
    m.load_state_dict(bst); m.eval()

    def pr(idxs):
        P, Bm_ = [], []
        with torch.no_grad():
            for k in range(0, len(idxs), 2048):
                ix = torch.tensor(idxs[k:k + 2048], device=dev)
                x, wm, wsd, _ = batch(ix)
                sc = SC[sysn[ix]]
                mu_r, _ = m((x - xmu) / xsd, wm, wsd, sysn[ix], clsn[ix])
                P.append((wm + mu_r * sc).cpu().numpy()); Bm_.append(wm.cpu().numpy())
        return np.vstack(P), np.vstack(Bm_)

    trs = np.random.default_rng(1).choice(tr, min(len(teidx), len(tr)), replace=False)
    Pte, Bte = pr(teidx); Ptr, _ = pr(trs)
    Yte, Ytr = E[si[teidx], tt[teidx]], E[si[trs], tt[trs]]
    return dict(tag=tag, n_tr=len(tr), n_te=len(teidx),
                test=sisre(Yte - Pte, teidx), train=sisre(Ytr - Ptr, trs),
                base=sisre(Yte - Bte, teidx), raw=sisre(Yte, teidx),
                dR=np.sqrt(((Yte - Pte)[:, 0] ** 2).mean()),
                dClk=np.sqrt(((Yte - Pte)[:, 3] ** 2).mean()))



def run_folds():
    """TEST 1-3: random-vs-time, rolling origins, satellite holdout."""
    rows = []
    print("\n" + "=" * 92)
    print("TEST 1 -- RANDOM 80/20 (what you asked for) vs TIME SPLIT, same model, same data")
    print("=" * 92, flush=True)
    rs = np.random.default_rng(0).permutation(len(tt))
    n = len(tt)
    rows.append(fit_eval(rs[:int(0.7 * n)], rs[int(0.7 * n):int(0.8 * n)], rs[int(0.8 * n):],
                         "RANDOM 80/20"))
    rows.append(fit_eval(np.nonzero(tt < 18720)[0],
                         np.nonzero((tt >= 19488) & (tt < 20832))[0],
                         np.nonzero(tt >= 21600)[0], "TIME (held-out future)"))
    print(f"{'split':26s} {'n train':>9s} {'n test':>8s} {'raw':>7s} {'base':>7s} "
          f"{'TEST':>7s} {'train':>7s} {'gap':>7s}")
    for r in rows:
        print(f"{r['tag']:26s} {r['n_tr']:9,d} {r['n_te']:8,d} {r['raw']:7.3f} {r['base']:7.3f} "
              f"{r['test']:7.3f} {r['train']:7.3f} {r['test']-r['train']:+7.3f}", flush=True)

    print("\n" + "=" * 92)
    print("TEST 2 -- ROLLING-ORIGIN FOLDS (retrained from scratch at each origin)")
    print("=" * 92, flush=True)
    folds = []
    for i, cut in enumerate([12000, 14400, 16800, 19200, 21600]):
        tr = np.nonzero(tt < cut - 768)[0]
        va = np.nonzero((tt >= cut - 768) & (tt < cut))[0]
        teI = np.nonzero((tt >= cut + 768) & (tt < cut + 768 + 1920))[0]
        if len(teI) < 2000 or len(tr) < 20000:
            continue
        folds.append(fit_eval(tr, va, teI, f"fold{i} cut@{cut}"))
    print(f"{'fold':26s} {'n train':>9s} {'n test':>8s} {'raw':>7s} {'base':>7s} "
          f"{'TEST':>7s} {'train':>7s} {'gap':>7s} {'cut%':>6s}")
    for r in folds:
        print(f"{r['tag']:26s} {r['n_tr']:9,d} {r['n_te']:8,d} {r['raw']:7.3f} {r['base']:7.3f} "
              f"{r['test']:7.3f} {r['train']:7.3f} {r['test']-r['train']:+7.3f} "
              f"{100*(1-r['test']/r['raw']):5.1f}%", flush=True)
    if folds:
        a = np.array([r["test"] for r in folds])
        b = np.array([r["base"] for r in folds])
        print(f"\n  SISRE across folds: mean {a.mean():.3f}  sd {a.std():.3f}  "
              f"min {a.min():.3f}  max {a.max():.3f}")
        print(f"  baseline  across folds: mean {b.mean():.3f}   model beats baseline in "
              f"{int((a<b).sum())}/{len(a)} folds")

    print("\n" + "=" * 92)
    print("TEST 3 -- SATELLITE HOLDOUT (unseen spacecraft) x unseen future")
    print("=" * 92, flush=True)
    hold = set(np.unique(si)[::3])
    seen = ~np.isin(si, list(hold))
    uns = np.isin(si, list(hold))
    tr = np.nonzero((tt < 18720) & seen)[0]
    va = np.nonzero((tt >= 19488) & (tt < 20832) & seen)[0]
    for nm, msk in [("unseen sats, seen time", (tt < 18720) & uns),
                    ("unseen sats + future", (tt >= 21600) & uns),
                    ("seen sats + future", (tt >= 21600) & seen)]:
        teI = np.nonzero(msk)[0]
        if len(teI) < 2000:
            continue
        r = fit_eval(tr, va, teI, nm)
        print(f"{r['tag']:26s} {r['n_te']:8,d} raw {r['raw']:6.3f}  base {r['base']:6.3f}  "
              f"TEST {r['test']:6.3f}  train {r['train']:6.3f}  gap {r['test']-r['train']:+.3f}", flush=True)


def run_mccv():
    """Blocked Monte-Carlo CV over 24 contiguous 2-week test blocks."""
    # ---------------- BLOCKED MONTE-CARLO CROSS-VALIDATION ----------------
    # 24 random contiguous 2-week test blocks. Contiguous => no window overlaps a
    # boundary, so no leakage. Many folds => an actual confidence interval instead
    # of one lucky week. Train strictly BEFORE the block, with an 8-day gap.
    import json
    BLK = 1344          # 14 days
    GAP = 768           # 8 days, one window length
    rng = np.random.default_rng(7)
    lo, hi = 9000, int(tt.max()) - BLK
    starts = np.unique((rng.integers(lo, hi, 40) // 96) * 96)[:24]
    res = []
    print(f"{'block start':>12s} {'n test':>8s} {'raw':>7s} {'base':>7s} {'MODEL':>7s} "
          f"{'train':>7s} {'gap':>7s} {'vs base':>8s}", flush=True)
    for st in starts:
        teI = np.nonzero((tt >= st) & (tt < st + BLK))[0]
        tr  = np.nonzero(tt < st - GAP)[0]
        va  = np.nonzero((tt >= st - GAP - 1344) & (tt < st - GAP))[0]
        if len(teI) < 3000 or len(tr) < 60000 or len(va) < 3000:
            continue
        r = fit_eval(tr, va, teI, f"blk{st}")
        r["start"] = int(st)
        res.append(r)
        print(f"{st:12d} {r['n_te']:8,d} {r['raw']:7.3f} {r['base']:7.3f} {r['test']:7.3f} "
              f"{r['train']:7.3f} {r['test']-r['train']:+7.3f} "
              f"{100*(1-r['test']/r['base']):+7.1f}%", flush=True)
    a = np.array([r["test"] for r in res]); b = np.array([r["base"] for r in res])
    w = np.array([r["raw"] for r in res]);  g = np.array([r["test"]-r["train"] for r in res])
    print("")
    print(f"=== BLOCKED MC-CV over {len(res)} folds ===")
    print(f"  raw broadcast SISRE   mean {w.mean():6.3f}")
    print(f"  baseline SISRE        mean {b.mean():6.3f}  sd {b.std():.3f}")
    print(f"  MODEL SISRE           mean {a.mean():6.3f}  sd {a.std():.3f}  "
          f"median {np.median(a):.3f}  [{a.min():.3f}, {a.max():.3f}]")
    lo_ci, hi_ci = np.percentile(a, [2.5, 97.5])
    print(f"  MODEL 95% interval    [{lo_ci:.3f}, {hi_ci:.3f}]")
    print(f"  vs raw                {100*(1-a.mean()/w.mean()):5.1f}% reduction")
    print(f"  vs baseline           {100*(1-a.mean()/b.mean()):5.1f}% mean improvement")
    print(f"  beats baseline in     {int((a<b).sum())}/{len(a)} folds")
    d = b - a
    print(f"  paired diff (base-model) mean {d.mean():+.4f}  sd {d.std():.4f}  "
          f"t = {d.mean()/(d.std()/np.sqrt(len(d))+1e-12):.2f}")
    print(f"  overfit gap           mean {g.mean():+.3f}  sd {g.std():.3f}")
    json.dump([{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in r.items()} for r in res], open("mccv.json", "w"), indent=1)


if MODE in ("folds", "all"):
    run_folds()
if MODE in ("mccv", "all"):
    run_mccv()
