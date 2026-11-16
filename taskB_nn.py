#!/usr/bin/env python3
"""Task B: do neural sequence models beat the physics-derived fit?

CROSS-VALIDATED. Five rolling origins, model retrained from scratch at each, every
fold reported plus mean +- sd. A single fold is exactly how the Task A number got
overstated by 2x earlier in this project, so no single-fold result is quoted here.

The deterministic predictor (harmonic/level basis chosen by self-validation inside
each supplied window) scores GEO 0.244 / MEO 0.397. Two learned layers have already
failed to improve it: horizon shrinkage was slightly worse, and gradient boosting on
its residuals made GEO 7-14% worse. This is the thorough version of that question.

Every model receives exactly what the competition supplies: the 7-day window,
irregularly sampled, plus the orbit class.

  mlp     window flattened, no temporal structure   (control: architecture vs information)
  gru     recurrent, 2 layers
  lstm    recurrent, 2 layers
  tcn     dilated causal convolutions
  xfmr    transformer with continuous time encoding -- the only architecture that
          takes irregular sampling natively rather than via a resampled grid

Each is run twice: predicting the VALUE, and predicting the DETERMINISTIC MODEL'S
RESIDUAL (the fair test -- it only has to find what the physics missed).

Windows are built once, with their origin recorded, and re-split per fold, so the
expensive deterministic fit is not recomputed five times.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from taskB_final import (XYZ, CLS, S, NT, WIN, HOR, STEP,
                         predict_geo, predict_meo, thin)

dev = "cuda"
G, NQ = 672, 96
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NPOOL = int(sys.argv[2]) if len(sys.argv) > 2 else 14000
NFOLD = 5
GAPW = 768                    # one window length, keeps train and test disjoint
BLOCK = 2200                  # test block width in 15-min steps (~23 days)


def build_pool(n, seed):
    """One pool of windows spanning the whole timeline; origin kept for splitting."""
    rg = np.random.default_rng(seed)
    X = np.zeros((n, G, 8), np.float32)
    Q = np.zeros((n, NQ, 2), np.float32)
    Y = np.full((n, NQ, 4), np.nan, np.float32)
    D = np.zeros((n, NQ, 4), np.float32)
    MU = np.zeros((n, 4), np.float32)
    SD = np.ones((n, 4), np.float32)
    GE = np.zeros(n, np.int64)
    OR = np.zeros(n, np.int64)
    k, tried = 0, 0
    t0 = time.time()
    while k < n and tried < n * 15:
        tried += 1
        s = int(rg.integers(0, S))
        o = int(rg.integers(WIN, NT - HOR))
        w, f = XYZ[s, o - WIN:o], XYZ[s, o:o + HOR]
        if np.isfinite(w[:, 0]).mean() < 0.5 or np.isfinite(f[:, 0]).mean() < 0.5:
            continue
        iw = np.nonzero(np.isfinite(w[:, 0]))[0]
        tw, vw = thin((iw - WIN) * float(STEP), w[iw], 0.30, rg)
        if len(tw) < 30:
            continue
        iq = np.nonzero(np.isfinite(f[:, 0]))[0]
        if len(iq) < 8:
            continue
        geo = CLS[s] in ("GEO", "GSO")
        tq = iq * float(STEP)
        det = predict_geo(tw, vw, tq) if geo else predict_meo(tw, vw, tq)
        gi = np.clip(((tw / STEP) + WIN).astype(int), 0, G - 1)
        mu, sd = vw.mean(0), vw.std(0) + 1e-3
        X[k, gi, :4] = (vw - mu) / sd
        X[k, gi, 4:] = 1.0
        m = min(len(iq), NQ)
        Q[k, :m, 0] = tq[:m] / 86400.0
        Q[k, :m, 1] = 1.0
        Y[k, :m] = f[iq][:m]
        D[k, :m] = det[:m]
        MU[k], SD[k], GE[k], OR[k] = mu, sd, int(geo), o
        k += 1
        if k % 2000 == 0:
            print(f"    {k}/{n} windows  [{time.time()-t0:.0f}s]", flush=True)
    return dict(X=X[:k], Q=Q[:k], Y=Y[:k], D=D[:k], MU=MU[:k], SD=SD[:k],
                GEO=GE[:k], ORIG=OR[:k])


class Enc(nn.Module):
    def __init__(s, kind, h=128):
        super().__init__()
        s.kind, s.h = kind, h
        if kind == "mlp":
            s.f = nn.Sequential(nn.Linear(G * 8, 512), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(512, h), nn.GELU())
        elif kind in ("gru", "lstm"):
            s.inp = nn.Linear(8, h)
            s.r = (nn.GRU if kind == "gru" else nn.LSTM)(h, h, 2, batch_first=True, dropout=0.1)
        elif kind == "tcn":
            ls = [nn.Conv1d(8, h, 1)]
            for d in (1, 2, 4, 8, 16, 32, 64, 128):
                ls += [nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.GELU(), nn.BatchNorm1d(h)]
            s.f = nn.Sequential(*ls)
        else:
            s.inp = nn.Linear(16, h)
            lay = nn.TransformerEncoderLayer(h, 4, h * 2, 0.1, batch_first=True, norm_first=True)
            s.f = nn.TransformerEncoder(lay, 3)
            t = torch.arange(G).float()[:, None] * STEP / 86400.0
            fr = torch.exp(torch.arange(4).float() * np.log(10.0))
            s.register_buffer("pe", torch.cat([torch.sin(2 * np.pi * t * fr),
                                               torch.cos(2 * np.pi * t * fr)], 1))

    def forward(s, x):
        if s.kind == "mlp":
            return s.f(x.flatten(1))
        if s.kind in ("gru", "lstm"):
            o, _ = s.r(torch.relu(s.inp(x)))
            return o[:, -1]
        if s.kind == "tcn":
            return s.f(x.transpose(1, 2))[:, :, -1]
        return s.f(s.inp(torch.cat([x, s.pe[None].expand(x.shape[0], -1, -1)], 2))).mean(1)


class Net(nn.Module):
    def __init__(s, kind, h=128):
        super().__init__()
        s.enc = Enc(kind, h)
        s.head = nn.Sequential(nn.Linear(h + 3, 128), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(128, 4))

    def forward(s, x, q, g):
        c = s.enc(x)[:, None, :].expand(-1, q.shape[1], -1)
        gg = g[:, None, None].float().expand(-1, q.shape[1], 1)
        return s.head(torch.cat([c, q, gg], 2))


def fit_fold(kind, residual, P, tri, tei, seed=0):
    torch.manual_seed(seed)
    T = {k: torch.tensor(v, device=dev) for k, v in P.items()}
    m = Net(kind).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 64 if kind in ("gru", "lstm", "xfmr") else 128
    nb = max(len(tri) // bs, 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * nb)
    tri_t = torch.tensor(tri, device=dev)
    for ep in range(EPOCHS):
        m.train()
        perm = tri_t[torch.randperm(len(tri_t), device=dev)]
        for i in range(nb):
            ix = perm[i * bs:(i + 1) * bs]
            if len(ix) < 8:
                continue
            base = T["D"][ix] if residual else T["MU"][ix][:, None, :].expand(-1, NQ, -1)
            tgt = (T["Y"][ix] - base) / T["SD"][ix][:, None, :]
            good = (T["Q"][ix][:, :, 1] > 0) & torch.isfinite(tgt).all(2)
            if good.sum() < 8:
                continue
            p = m(T["X"][ix], T["Q"][ix], T["GEO"][ix])
            loss = nn.functional.huber_loss(p[good], tgt[good], delta=2.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    m.eval()
    out = {}
    with torch.no_grad():
        pr = []
        for i in range(0, len(tei), 256):
            ix = torch.tensor(tei[i:i + 256], device=dev)
            base = T["D"][ix] if residual else T["MU"][ix][:, None, :].expand(-1, NQ, -1)
            pr.append((base + m(T["X"][ix], T["Q"][ix], T["GEO"][ix])
                       * T["SD"][ix][:, None, :]).cpu().numpy())
    pr = np.concatenate(pr)
    for g, nm in ((1, "GEO"), (0, "MEO")):
        sel = P["GEO"][tei] == g
        if sel.sum() < 5:
            continue
        idx = tei[sel]
        v = (P["Q"][idx][:, :, 1] > 0) & np.isfinite(P["Y"][idx]).all(2)
        if v.sum() < 50:
            continue
        out[nm] = (float(np.sqrt(np.mean((P["Y"][idx] - pr[sel])[v] ** 2))),
                   float(np.sqrt(np.mean((P["Y"][idx] - P["D"][idx])[v] ** 2))))
    return out


if __name__ == "__main__":
    print(f"building pool of {NPOOL} windows (deterministic fit computed once) ...", flush=True)
    P = build_pool(NPOOL, 7)
    O = P["ORIG"]
    print(f"pool {len(O)}  GEO {int(P['GEO'].sum())}  MEO {int((1-P['GEO']).sum())}\n", flush=True)
    folds = []
    for i, cut in enumerate(np.linspace(NT * 0.42, NT * 0.80, NFOLD).astype(int)):
        tri = np.nonzero(O < cut - GAPW)[0]
        tei = np.nonzero((O >= cut) & (O < cut + BLOCK))[0]
        if len(tri) > 800 and len(tei) > 150:
            folds.append((cut, tri, tei))
    print(f"{len(folds)} folds: " + ", ".join(
        f"cut={c} tr={len(a)} te={len(b)}" for c, a, b in folds) + "\n", flush=True)

    for residual in (False, True):
        print(f"\n{'='*80}\n{'RESIDUAL of the deterministic model' if residual else 'VALUE directly'}"
              f"  -- net RMS vs deterministic RMS, per fold\n{'='*80}", flush=True)
        for kind in (["mlp", "gru", "lstm", "tcn", "xfmr"] if not residual
                     else ["mlp", "gru", "tcn", "xfmr"]):
            acc = {"GEO": [], "MEO": []}
            det = {"GEO": [], "MEO": []}
            t0 = time.time()
            try:
                for ci, (cut, tri, tei) in enumerate(folds):
                    r = fit_fold(kind, residual, P, tri, tei, seed=ci)
                    for nm, (a, b) in r.items():
                        acc[nm].append(a); det[nm].append(b)
            except Exception as e:
                print(f"  {kind:6s} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            for nm in ("GEO", "MEO"):
                if len(acc[nm]) < 2:
                    continue
                a, b = np.array(acc[nm]), np.array(det[nm])
                d = 100 * (a.mean() / b.mean() - 1)
                verdict = "BEATS" if d < -2 else ("ties" if d < 2 else "worse")
                print(f"  {kind:6s} {nm:4s} net {a.mean():6.3f}+-{a.std():.3f}  "
                      f"det {b.mean():6.3f}+-{b.std():.3f}  {d:+6.1f}%  {verdict:5s} "
                      f"wins {int((a<b).sum())}/{len(a)}  "
                      f"folds[" + " ".join(f"{x:.2f}" for x in a) + f"]  [{time.time()-t0:.0f}s]",
                      flush=True)
