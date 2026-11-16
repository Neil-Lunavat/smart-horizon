#!/usr/bin/env python3
"""TASK B -- benchmark every reasonable algorithm family, 24 blocked folds.

Same windows, same folds, same metric for all of them. Nothing is ever quoted
from a single fold; that mistake was already made once on the Task A side and
cost a factor of two.

Every method must satisfy the competition's signature: consume an irregularly
sampled 7-day window of (t, x, y, z, clock) plus the orbit class, and emit values
at arbitrary day-8 timestamps.

TWO STRUCTURAL FIXES over the first attempt
  1. GEO and MEO pools are built SEPARATELY. GEO/GSO is 15 of 103 spacecraft, so
     a proportionally sampled pool left ~12 GEO windows per fold -- below the
     minimum -- and every fold would have reported MEO only.
  2. The window-local methods (everything except the six tabular learners) do not
     look at the fold's training set at all, so they are scored ONCE per window,
     in parallel over cores, and the folds become pure aggregation. Only the
     tabular learners genuinely refit per fold.

FAMILIES
  trivial      zero, window mean/median, persistence, linear drift
  seasonal     seasonal-naive at the estimated period, and its multi-cycle mean
  spectral     top-k FFT continuation, fixed harmonic regression, Prony /
               matrix-pencil damped sinusoids, singular spectrum analysis
  smoothing    Holt-Winters exponential smoothing
  arima        ARIMA, seasonal ARIMA, Yule-Walker AR -- on an interpolated grid
  kernel       Gaussian process, periodic x RBF -- the textbook answer for
               irregularly sampled quasi-periodic data
  ml           random forest, extra trees, gradient boosting, kNN, SVR, ridge on
               window-derived features, retrained per fold
  ours         the self-validating harmonic/level predictor (SMART-HORIZON)

Methods needing a regular grid get one by linear interpolation; that is part of
the method, not a favour to it.
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

STEP, WIN, HOR = 900, 672, 96
T_SID = 86164.0
T4 = ["x_err", "y_err", "z_err", "clock_err"]
NFOLD, GAPW, BLOCK = 24, 768, 1344          # 2-week test blocks, 8-day gap

WLOCAL = ["zero", "mean", "median", "last", "drift", "snaive", "snaive-avg",
          "fft", "harm-fixed", "prony", "ssa", "gp", "holt-winters",
          "arima", "sarima", "ar-yw", "ours"]
MLNAMES = ["rf", "extratrees", "gbm", "knn", "svr", "ridge"]


# ------------------------------------------------------------------ helpers
def to_grid(tw, y):
    g = np.arange(-WIN, 0) * float(STEP)
    return g, np.interp(g, tw, y)


def snaive(tw, y, tq, P, average):
    """Predict from the same phase one (or every available) period earlier.

    The strongest naive baseline for a quasi-periodic series, and the one a
    reviewer will ask about first."""
    lo, hi = tw.min(), tw.max()
    out = np.empty(len(tq))
    for i, t in enumerate(tq):
        vals = []
        k = 1
        while t - k * P >= lo and k <= 40:
            if t - k * P <= hi:
                vals.append(float(np.interp(t - k * P, tw, y)))
                if not average:
                    break
            k += 1
        out[i] = float(np.mean(vals)) if vals else float(y.mean())
    return out


def fft_extrap(tw, y, tq, k=6):
    """Keep the k strongest FFT components of the gridded window and continue
    them periodically. Assumes the window length is itself the period."""
    g, yg = to_grid(tw, y)
    n = len(yg)
    F = np.fft.rfft(yg - yg.mean())
    keep = np.argsort(np.abs(F))[::-1][:k]
    ph = (tq - g[0]) / float(STEP)
    out = np.full(len(tq), float(yg.mean()))
    for m in keep:
        out += 2.0 / n * (F[m].real * np.cos(2 * np.pi * m * ph / n)
                          - F[m].imag * np.sin(2 * np.pi * m * ph / n))
    return out


def harm_fixed(tw, y, tq, P, K=2):
    """Harmonic regression at the estimated period, K=2, fitted on all 7 days
    with no model selection. The control that isolates what self-validation buys."""
    def des(t):
        c = [np.ones_like(t)]
        for j in range(1, K + 1):
            c += [np.sin(2 * np.pi * j * t / P), np.cos(2 * np.pi * j * t / P)]
        return np.column_stack(c)
    A = des(tw)
    lam = 1e-3 * np.trace(A.T @ A) / A.shape[1]
    try:
        c = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    except np.linalg.LinAlgError:
        return np.full(len(tq), float(y.mean()))
    return des(tq) @ c


def prony(tw, y, tq, K=3):
    """Matrix-pencil estimate of K damped sinusoids, then extrapolate. Unlike
    harmonic regression this lets amplitudes grow or decay."""
    g, yg = to_grid(tw, y)
    n = len(yg)
    L = n // 3
    Y = np.column_stack([yg[i:i + L] for i in range(n - L + 1)])
    try:
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        Us = U[:, :2 * K]
        A = np.linalg.pinv(Us[:-1]) @ Us[1:]
        zs = np.linalg.eigvals(A)
        zs = zs[np.abs(zs) < 1.02]
        if len(zs) == 0:
            return np.full(len(tq), float(y.mean()))
        idx = np.arange(n)[:, None]
        V = zs[None, :] ** idx
        c, *_ = np.linalg.lstsq(np.column_stack([V.real, V.imag]), yg, rcond=None)
        step = (tq - g[-1]) / float(STEP) + n - 1
        Vq = zs[None, :] ** step[:, None]
        return np.column_stack([Vq.real, Vq.imag]) @ c
    except Exception:
        return np.full(len(tq), float(y.mean()))


def ssa(tw, y, tq, L=192, r=8):
    """Singular spectrum analysis: reconstruct with r components, then continue
    with the linear recurrence the trajectory matrix implies."""
    g, yg = to_grid(tw, y)
    n = len(yg)
    L = min(L, n // 2)
    X = np.column_stack([yg[i:i + L] for i in range(n - L + 1)])
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(r, len(s))
        Xr = (U[:, :r] * s[:r]) @ Vt[:r]
        rec = np.array([np.mean(np.diag(Xr[:, ::-1], k))
                        for k in range(-Xr.shape[0] + 1, Xr.shape[1])])[::-1]
        Pm = U[:L - 1, :r]
        pi = U[L - 1, :r]
        nu2 = float(pi @ pi)
        if nu2 >= 1 - 1e-6:
            return np.full(len(tq), float(y.mean()))
        R = (Pm @ pi) / (1 - nu2)
        ext = list(rec)
        need = int(np.ceil((tq.max() - g[-1]) / STEP)) + 2
        for _ in range(max(need, 1)):
            ext.append(float(np.dot(R, ext[-(L - 1):])))
        ext = np.array(ext)
        tt = np.concatenate([g, g[-1] + np.arange(1, len(ext) - len(rec) + 1) * STEP])
        return np.interp(tq, tt[:len(ext)], ext)
    except Exception:
        return np.full(len(tq), float(y.mean()))


def gp_periodic(tw, y, tq, P):
    """GP with a periodic x RBF kernel, closed form, small grid search over the
    length scale, decay and noise. Handles irregular sampling natively."""
    t1, t2 = tw / 86400.0, tq / 86400.0
    mu = float(y.mean())
    yy = y - mu
    sd = float(y.std()) + 1e-9
    Pd = P / 86400.0

    def K(a, b, ls, dec):
        d = a[:, None] - b[None, :]
        return (np.exp(-2 * np.sin(np.pi * np.abs(d) / Pd) ** 2 / ls ** 2)
                * np.exp(-0.5 * (d / dec) ** 2))

    sub = np.linspace(0, len(t1) - 1, min(len(t1), 260)).astype(int)
    a1, ay = t1[sub], yy[sub]
    best, bp = None, np.inf
    for ls in (0.5, 1.0):
        for dec in (3.0, 10.0):
            for nz in (0.05, 0.2):
                Kaa = K(a1, a1, ls, dec) * sd ** 2 + (nz * sd) ** 2 * np.eye(len(a1))
                try:
                    L_ = np.linalg.cholesky(Kaa)
                except np.linalg.LinAlgError:
                    continue
                al = np.linalg.solve(L_.T, np.linalg.solve(L_, ay))
                nll = 0.5 * ay @ al + np.log(np.diag(L_)).sum()
                if nll < bp:
                    bp, best = nll, (ls, dec, al)
    if best is None:
        return np.full(len(tq), mu)
    ls, dec, al = best
    return mu + K(t2, a1, ls, dec) * sd ** 2 @ al


def holt_winters(tw, y, tq, P):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    g, yg = to_grid(tw, y)
    step = 4
    ys, gs = yg[::step], g[::step]
    sp = max(int(round(P / (STEP * step))), 2)
    need = int(np.ceil((tq.max() - gs[-1]) / (STEP * step))) + 1
    try:
        if len(ys) > 2 * sp:
            m = ExponentialSmoothing(ys, trend=None, seasonal="add",
                                     seasonal_periods=sp).fit(optimized=True)
        else:
            m = ExponentialSmoothing(ys, trend=None, seasonal=None).fit(optimized=True)
        f = np.asarray(m.forecast(need))
    except Exception:
        return np.full(len(tq), float(y.mean()))
    tt = gs[-1] + np.arange(1, need + 1) * STEP * step
    return np.interp(tq, tt, f)


def arima_fc(tw, y, tq, P=None, seasonal=False):
    from statsmodels.tsa.arima.model import ARIMA
    g, yg = to_grid(tw, y)
    step = 8                                   # thin to 2 h so ARIMA is tractable
    ys, gs = yg[::step], g[::step]
    need = int(np.ceil((tq.max() - gs[-1]) / (STEP * step))) + 1
    kw = {}
    if seasonal and P:
        sp = int(round(P / (STEP * step)))
        if 2 <= sp <= len(ys) // 3:
            kw["seasonal_order"] = (1, 0, 1, sp)
    try:
        f = np.asarray(ARIMA(ys, order=(2, 0, 2), **kw).fit().forecast(need))
    except Exception:
        return np.full(len(tq), float(y.mean()))
    tt = gs[-1] + np.arange(1, need + 1) * STEP * step
    return np.interp(tq, tt, f)


def ar_yw(tw, y, tq, p=48):
    """Yule-Walker AR on a 1-hour grid, iterated forward. The classical linear
    predictor, and the honest floor for 'just model it as a time series'."""
    g, yg = to_grid(tw, y)
    step = 4
    ys, gs = yg[::step], g[::step]
    mu = float(ys.mean())
    z = ys - mu
    p = min(p, len(z) // 3)
    try:
        r = np.correlate(z, z, "full")[len(z) - 1:] / len(z)
        R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
        a = np.linalg.solve(R + 1e-9 * np.eye(p) * r[0], r[1:p + 1])
    except Exception:
        return np.full(len(tq), mu)
    need = int(np.ceil((tq.max() - gs[-1]) / (STEP * step))) + 1
    buf = list(z[-p:])
    fc = []
    for _ in range(need):
        v = float(np.dot(a, buf[::-1][:p]))
        fc.append(v)
        buf.append(v)
    tt = gs[-1] + np.arange(1, need + 1) * STEP * step
    return np.interp(tq, tt, np.array(fc) + mu)


def ml_feats(tw, y, tq, P):
    """Window summary + query phase -> value. Shared by every tabular learner."""
    mu, sd = float(y.mean()), float(y.std()) + 1e-9
    ph = (tq % P) / P
    day = tw >= tw.max() - 86400.0
    base = [mu, sd, float(y[day].mean()) if day.sum() > 3 else mu,
            float(y[-1]), float(np.median(y)), float(stats.skew(y)),
            float(stats.kurtosis(y)), (tw[-1] % P) / P]
    return np.column_stack([np.repeat([base], len(tq), 0), tq / 86400.0,
                            np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph),
                            np.sin(4 * np.pi * ph), np.cos(4 * np.pi * ph)])


# ------------------------------------------------------------------ worker
def score_window(r):
    """Sum of squared errors per (method, target) for one window. Window-local
    methods only -- these never see anything outside the window they are given,
    so the result is fold-independent and computed exactly once."""
    tw, vw, tq, y, P = r["tw"], r["vw"], r["tq"], r["y"], r["P"]
    sse = {k: np.zeros(4) for k in WLOCAL}
    for j in range(4):
        yj, tj = vw[:, j], y[:, j]
        sse["zero"][j] = (tj ** 2).sum()
        sse["mean"][j] = ((tj - yj.mean()) ** 2).sum()
        sse["median"][j] = ((tj - np.median(yj)) ** 2).sum()
        sse["last"][j] = ((tj - yj[-1]) ** 2).sum()
        sse["drift"][j] = ((tj - np.polyval(np.polyfit(tw, yj, 1), tq)) ** 2).sum()
        sse["snaive"][j] = ((tj - snaive(tw, yj, tq, P, False)) ** 2).sum()
        sse["snaive-avg"][j] = ((tj - snaive(tw, yj, tq, P, True)) ** 2).sum()
        sse["fft"][j] = ((tj - fft_extrap(tw, yj, tq)) ** 2).sum()
        sse["harm-fixed"][j] = ((tj - harm_fixed(tw, yj, tq, P)) ** 2).sum()
        sse["prony"][j] = ((tj - prony(tw, yj, tq)) ** 2).sum()
        sse["ssa"][j] = ((tj - ssa(tw, yj, tq)) ** 2).sum()
        sse["gp"][j] = ((tj - gp_periodic(tw, yj, tq, P)) ** 2).sum()
        sse["holt-winters"][j] = ((tj - holt_winters(tw, yj, tq, P)) ** 2).sum()
        sse["arima"][j] = ((tj - arima_fc(tw, yj, tq)) ** 2).sum()
        sse["sarima"][j] = ((tj - arima_fc(tw, yj, tq, P, True)) ** 2).sum()
        sse["ar-yw"][j] = ((tj - ar_yw(tw, yj, tq)) ** 2).sum()
        sse["ours"][j] = ((tj - r["det"][:, j]) ** 2).sum()
    return sse, len(y)


def fit_fold_ml(args):
    """The six tabular learners, retrained from scratch on this fold's training
    windows only. The one family that genuinely needs the corpus."""
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                  GradientBoostingRegressor)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge
    R, T = args
    mk = {"rf": lambda: RandomForestRegressor(120, max_depth=12, n_jobs=1, random_state=0),
          "extratrees": lambda: ExtraTreesRegressor(120, max_depth=14, n_jobs=1, random_state=0),
          "gbm": lambda: GradientBoostingRegressor(random_state=0),
          "knn": lambda: KNeighborsRegressor(15),
          "svr": lambda: SVR(C=2.0, epsilon=0.05),
          "ridge": lambda: Ridge(1.0)}
    sse = {k: np.zeros(4) for k in MLNAMES}
    n = sum(len(r["y"]) for r in T)
    for j in range(4):
        Xtr = np.nan_to_num(np.vstack([ml_feats(r["tw"], r["vw"][:, j], r["tq"], r["P"])
                                       for r in R]))
        ytr = np.concatenate([r["y"][:, j] for r in R])
        Xte = [np.nan_to_num(ml_feats(r["tw"], r["vw"][:, j], r["tq"], r["P"])) for r in T]
        for name in MLNAMES:
            m = mk[name]().fit(Xtr, ytr)
            for r, X in zip(T, Xte):
                sse[name][j] += ((r["y"][:, j] - m.predict(X)) ** 2).sum()
    return sse, n


# ------------------------------------------------------------------ pool
def build(n, want_geo, seed):
    """Windows drawn only from spacecraft of the requested class, so both classes
    get a properly populated pool regardless of how few GEO satellites exist."""
    from taskB_final import (XYZ, CLS, S, NT, predict_geo, predict_meo, thin,
                             ls_period, MEO_LO, MEO_HI)
    ok = [i for i in range(S) if (CLS[i] in ("GEO", "GSO")) == want_geo]
    rg = np.random.default_rng(seed)
    out, tried = [], 0
    t0 = time.time()
    while len(out) < n and tried < n * 25:
        tried += 1
        s = int(ok[rg.integers(0, len(ok))])
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
        tq = iq * float(STEP)
        P = T_SID if want_geo else (ls_period(tw, vw[:, 2], MEO_LO, MEO_HI) or 43082.0)
        det = predict_geo(tw, vw, tq) if want_geo else predict_meo(tw, vw, tq)
        out.append(dict(tw=tw, vw=vw, tq=tq, y=f[iq], orig=o, P=float(P), det=det))
        if len(out) % 250 == 0:
            print(f"    {'GEO' if want_geo else 'MEO'} {len(out)}/{n}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
    return out


def make_folds(O, NT):
    rgf = np.random.default_rng(9)
    folds = []
    for cut in np.unique((rgf.integers(int(NT * 0.30), NT - BLOCK, NFOLD * 6) // 96) * 96):
        tri = np.nonzero(O < cut - GAPW)[0]
        tei = np.nonzero((O >= cut) & (O < cut + BLOCK))[0]
        if len(tri) > 150 and len(tei) > 20:
            folds.append((int(cut), tri, tei))
        if len(folds) >= NFOLD:
            break
    return folds


if __name__ == "__main__":
    NG = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
    NM = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
    NW = min(14, os.cpu_count() or 4)
    from taskB_final import NT

    RES = {}
    for cls, want, npool in (("GEO", True, NG), ("MEO", False, NM)):
        print(f"\n### {cls}: building {npool} windows ...", flush=True)
        pool = build(npool, want, 3 if want else 4)
        O = np.array([r["orig"] for r in pool])
        folds = make_folds(O, NT)
        print(f"{cls} pool {len(pool)}  {len(folds)} folds "
              f"(test {np.mean([len(t) for _, _, t in folds]):.0f} windows each)", flush=True)

        t0 = time.time()
        with ProcessPoolExecutor(NW) as ex:
            scored = list(ex.map(score_window, pool, chunksize=8))
        print(f"  window-local methods scored in {time.time()-t0:.0f}s "
              f"({len(WLOCAL)} methods x {len(pool)} windows)", flush=True)
        SSE = {k: np.array([s[0][k] for s in scored]) for k in WLOCAL}
        CNT = np.array([s[1] for s in scored])

        t0 = time.time()
        jobs = [([pool[i] for i in tri[:250]], [pool[i] for i in tei])
                for _, tri, tei in folds]
        with ProcessPoolExecutor(NW) as ex:
            mlres = list(ex.map(fit_fold_ml, jobs))
        print(f"  tabular learners fitted in {time.time()-t0:.0f}s "
              f"({len(MLNAMES)} models x 4 targets x {len(folds)} folds)", flush=True)

        per = {}
        for fi, (_, tri, tei) in enumerate(folds):
            n = int(CNT[tei].sum())
            for k in WLOCAL:
                per.setdefault(k, []).append(np.sqrt(SSE[k][tei].sum(0) / n))
            msse, mn = mlres[fi]
            for k in MLNAMES:
                per.setdefault(k, []).append(np.sqrt(msse[k] / mn))
        RES[cls] = {k: np.array(v) for k, v in per.items()}

    np.save("bench_results.npy", RES, allow_pickle=True)
    for cls in ("GEO", "MEO"):
        rows = sorted(RES[cls].items(), key=lambda kv: kv[1].mean())
        nf = len(next(iter(RES[cls].values())))
        print(f"\n{'='*104}\n{cls}  -- RMS in metres, mean over x/y/z/clock, "
              f"{nf} blocked folds\n{'='*104}")
        print(f"{'method':14s} {'mean':>7s} {'sd':>6s} {'best':>6s} {'worst':>6s} | "
              f"{'x':>6s} {'y':>6s} {'z':>6s} {'clock':>6s} | vs ours")
        base = RES[cls]["ours"].mean(1)
        for k, v in rows:
            m = v.mean(1)
            d = 100 * (m.mean() / base.mean() - 1)
            print(f"{k:14s} {m.mean():7.3f} {m.std():6.3f} {m.min():6.3f} {m.max():6.3f} | "
                  + " ".join(f"{x:6.3f}" for x in v.mean(0))
                  + f" | {d:+7.1f}%  wins {int((m < base).sum())}/{len(m)}")
        print(f"\n{cls} per-fold for ours: " + " ".join(f"{x:.3f}" for x in base))
