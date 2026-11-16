#!/usr/bin/env python3
"""Task A residual diagnostics -- the five investigations that explained the tail.

Previously five scripts (fix_sisre.py, persys.py, tails.py, g27.py, mixture.py).
They run in the order the investigation actually happened, and that order is the
argument:

  sisre    corrected SISRE for the XGBoost stage, raw vs model     [tab_pred.npz]
  persys   the same, broken out per constellation -- BeiDou's 12 m clock swamps
           Galileo's 0.38 m in any pooled metric                   [tab_pred.npz]
  tails    v2's clock residual has skew -7.11 and kurtosis 438. Locate the tail
           before trying to fix it: which rows, which satellites, which days, and
           does the BASELINE share it?                          [v2_pred_mlp.npz]
  g27      it does not -- raw clock kurtosis is 11.5 and the baseline's 12.2, so
           the MODEL manufactures the tail. What is it reacting to on G27/C12,
           and is the trigger visible in broadcast (i.e. fixable in real time)?
  mixture  the answer. Residuals are Gaussian WITHIN a satellite (median Shapiro
           W ~ 0.99) but satellite scales differ 11-22x, so the pooled histogram
           is a SCALE MIXTURE -- heavy-tailed by construction, with every
           component normal. The cure is to model the scale, not change the loss.
           Also: broadcast handover-jump sd predicts residual sd at r = 0.69
           (radial) / 0.54 (clock), so sigma is inferable with no SP3.

That last finding is the one that transferred to Task B.

Usage: python taskA_diagnostics.py [sisre|persys|tails|g27|mixture|all]

Needs taskA.npz plus tab_pred.npz (sisre, persys) and v2_pred_mlp.npz (the rest).
Both prediction dumps are regenerable -- see REBUILD.md.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

SECTIONS = ["sisre", "persys", "tails", "g27", "mixture"]
WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
if WHICH not in SECTIONS + ["all"]:
    raise SystemExit(f"unknown section {WHICH!r}; expected {', '.join(SECTIONS)} or all")


def _taskA():
    return np.load("taskA.npz", allow_pickle=True)


def _tab():
    d = np.load("tab_pred.npz")
    return d["Yte"], d["Pg"], d["si"], d["wR"], d["wA"]


def _v2():
    d = np.load("v2_pred_mlp.npz")
    return d["Yt"], d["Pf"], d["Bt"], d["si"], d["tt"]


def _hdr(title):
    print(f"\n{'#' * 78}\n# {title}\n{'#' * 78}")


# ------------------------------------------------------------------ 1. sisre
def sec_sisre():
    _hdr("SISRE -- corrected, XGBoost stage")
    Y, P, si, wR, wA = _tab()

    def sis(res):
        return np.sqrt(np.mean((wR * res[:, 0] - res[:, 3]) ** 2
                               + wA ** 2 * (res[:, 1] ** 2 + res[:, 2] ** 2)))

    print("CORRECTED SISRE (m RMS) on test = 24 Aug - 1 Sep\n")
    print(f"  raw broadcast (no correction) {sis(Y):8.3f}")
    print(f"  XGBoost                       {sis(Y - P):8.3f}")
    print(f"  -> reduction                  {100 * (1 - sis(Y - P) / sis(Y)):7.1f}%")


# ----------------------------------------------------------------- 2. persys
def sec_persys():
    _hdr("PER-SYSTEM -- BeiDou's clock swamps Galileo's in any pooled metric")
    z = _taskA()
    SYS, T = z["sys"], list(z["tgt_cols"])
    Y, P, si, wR, wA = _tab()
    sy = np.array([SYS[i] for i in si])

    def sis(res, m):
        return np.sqrt(np.mean((wR[m] * res[m, 0] - res[m, 3]) ** 2
                               + wA[m] ** 2 * (res[m, 1] ** 2 + res[m, 2] ** 2)))

    print("PER-SYSTEM, test = 24 Aug - 1 Sep (m RMS)\n")
    print(f"{'sys':4s} {'n':>7s} {'SISRE raw':>10s} {'SISRE XGB':>10s} {'cut':>6s}   "
          + "  ".join(f"{c + ' raw':>10s} {c + ' xgb':>9s}" for c in T))
    for s in ["G", "E", "C", "J"]:
        m = sy == s
        if m.sum() < 500:
            continue
        cells = []
        for j in range(4):
            cells.append(f"{np.sqrt((Y[m,j]**2).mean()):10.3f} "
                         f"{np.sqrt(((Y[m,j]-P[m,j])**2).mean()):9.3f}")
        print(f"{s:4s} {m.sum():7d} {sis(Y,m):10.3f} {sis(Y-P,m):10.3f} "
              f"{100*(1-sis(Y-P,m)/sis(Y,m)):5.1f}%   " + "  ".join(cells))
    m = np.ones(len(Y), bool)
    print(f"{'ALL':4s} {m.sum():7d} {sis(Y,m):10.3f} {sis(Y-P,m):10.3f} "
          f"{100*(1-sis(Y-P,m)/sis(Y,m)):5.1f}%")


# ------------------------------------------------------------------ 3. tails
def sec_tails():
    _hdr("TAILS -- where does the clock residual kurtosis of 438 come from?")
    z = _taskA()
    SATS, SYS = z["sats"], z["sys"]
    T = list(z["tgt_cols"])
    Y, P, B, si, tt = _v2()
    r = Y - P
    sc = np.array([SATS[i] for i in si])
    sy = np.array([SYS[i] for i in si])

    print("=== residual quantiles (m) ===")
    q = [0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99]
    print(f"{'target':8s} " + "  ".join(f"{x:>9}" for x in q))
    for j, c in enumerate(T):
        print(f"{c:8s} " + "  ".join(f"{v:9.3f}" for v in np.percentile(r[:, j], q)))

    print("\n=== how concentrated is the clock tail? ===")
    a = np.abs(r[:, 3])
    tot = (a ** 2).sum()
    for k in [10, 50, 100, 500, 1000]:
        idx = np.argsort(-a)[:k]
        print(f"  worst {k:5d} of {len(a):6d} rows ({100*k/len(a):5.2f}%) hold "
              f"{100*(a[idx]**2).sum()/tot:5.1f}% of squared clock error")

    print("\n=== which satellites own the clock tail? ===")
    big = a > np.percentile(a, 99.5)
    vc = pd.Series(sc[big]).value_counts().head(10)
    n_by = pd.Series(sc).value_counts()
    print(f"{'sat':8s} {'sys':4s} {'tail rows':>10s} {'total rows':>11s} {'% of its rows':>14s}")
    for s, n in vc.items():
        print(f"{s:8s} {SYS[list(SATS).index(s)]:4s} {n:10d} {n_by[s]:11d} "
              f"{100*n/n_by[s]:13.2f}%")

    print("\n=== clock residual by system, robust vs raw ===")
    print(f"{'sys':5s} {'n':>7s} {'RMS':>8s} {'robust sd':>10s} {'ratio':>7s} "
          f"{'skew':>8s} {'kurt':>9s}")
    for s in "GECJ":
        m = sy == s
        if m.sum() < 200:
            continue
        x = r[m, 3]
        rob = 1.4826 * np.median(np.abs(x - np.median(x)))
        print(f"{s:5s} {m.sum():7d} {np.sqrt((x**2).mean()):8.3f} {rob:10.3f} "
              f"{np.sqrt((x**2).mean())/max(rob,1e-9):7.2f} {stats.skew(x):8.2f} "
              f"{stats.kurtosis(x):9.1f}")

    print("\n=== are the tail rows clustered in TIME? (would mean an event, not noise) ===")
    tail = tt[big]
    h = pd.Series(pd.to_datetime("2026-01-01") + pd.to_timedelta(tail * 900, "s"))
    print(h.dt.floor("D").value_counts().sort_index().to_string())

    print("\n=== if the worst 0.5% of clock rows were excluded ===")
    keep = ~big
    for j, c in enumerate(T):
        x = r[keep, j]
        sub = np.random.default_rng(0).choice(x, 4000, replace=False)
        print(f"  {c:8s} RMS {np.sqrt((x**2).mean()):7.3f}  skew {stats.skew(x):6.2f}  "
              f"kurt {stats.kurtosis(x):8.1f}  ShapiroW {stats.shapiro(sub).statistic:.4f}")

    print("\n=== does the BASELINE have the same tail? (model's fault or data's?) ===")
    rb = Y - B
    for j, c in enumerate(T):
        print(f"  {c:8s} model kurt {stats.kurtosis(r[:,j]):8.1f}   "
              f"baseline kurt {stats.kurtosis(rb[:,j]):8.1f}   "
              f"raw-error kurt {stats.kurtosis(Y[:,j]):8.1f}")


# -------------------------------------------------------------------- 4. g27
def sec_g27():
    _hdr("G27 / C12 -- the model manufactures the tail; what triggers it?")
    z = _taskA()
    SATS, T = list(z["sats"]), list(z["tgt_cols"])
    E, B = z["E"], z["B"]
    BC = list(z["bc_cols"])
    Y, P, Bl, si, tt = _v2()
    r = Y - P
    t0 = pd.Timestamp("2026-01-01")

    for sat in ["G27", "C12"]:
        k = SATS.index(sat)
        m = si == k
        print(f"\n{'='*78}\n{sat}: {m.sum()} test rows")
        print(f"  clock  RMS model {np.sqrt((r[m,3]**2).mean()):.3f}   "
              f"baseline {np.sqrt(((Y[m,3]-Bl[m,3])**2).mean()):.3f}   "
              f"raw {np.sqrt((Y[m,3]**2).mean()):.3f}")
        worst = np.argsort(-np.abs(r[m, 3]))[:6]
        tw = tt[m][worst]
        print("  worst epochs:")
        print(f"    {'time':17s} {'truth':>8s} {'model':>8s} {'baseline':>9s} {'resid':>8s}")
        for i, w in zip(worst, tw):
            print(f"    {str(t0+pd.Timedelta(seconds=int(w)*900)):17s} {Y[m][i,3]:8.3f} "
                  f"{P[m][i,3]:8.3f} {Bl[m][i,3]:9.3f} {r[m][i,3]:8.3f}")
        # full clock history on the grid: is there a jump?
        ser = E[k, :, 3]
        fin = np.isfinite(ser)
        idx = np.nonzero(fin)[0]
        daily = pd.Series(ser[fin],
                          index=(t0 + pd.to_timedelta(idx * 900, "s"))).resample("D").median()
        tailwin = daily.loc["2026-08-05":]
        print("  daily median clock error (m), Aug 5 onward:")
        print("    " + "  ".join(f"{v:.2f}" for v in tailwin.values))
        jump = daily.diff().abs()
        print(f"  largest day-to-day jump in the whole record: {jump.max():.3f} m "
              f"on {jump.idxmax().date() if pd.notna(jump.max()) else '-'}")

    print(f"\n{'='*78}\nIs the trigger visible in BROADCAST (i.e. usable in real time)?")
    ihc = BC.index("ho_clk_m")
    for sat in ["G27", "C12", "G04", "E16"]:
        k = SATS.index(sat)
        h = B[k, :, ihc]
        h = h[np.isfinite(h)]
        late = B[k, 21600:, ihc]
        late = late[np.isfinite(late)]
        print(f"  {sat:6s} handover clock jump: whole-record sd {h.std():7.3f}  "
              f"test-period sd {late.std() if len(late) else float('nan'):7.3f}  "
              f"max|.| test {np.abs(late).max() if len(late) else float('nan'):7.3f}")

    print(f"\n{'='*78}\nHow well would a SIMPLE GUARD work?")
    print("Rule: if |model - baseline| is large, the model is extrapolating; fall back.")
    diff = np.abs(P[:, 3] - Bl[:, 3])
    for thr in [0.3, 0.5, 0.75, 1.0, 1.5, 99]:
        g = diff > thr
        Pg = P[:, 3].copy()
        Pg[g] = Bl[g, 3]
        rr = Y[:, 3] - Pg
        sub = np.random.default_rng(0).choice(rr, 4000, replace=False)
        print(f"  fallback when |diff|>{thr:5.2f} ({100*g.mean():5.2f}% of rows): "
              f"RMS {np.sqrt((rr**2).mean()):.3f}  skew {stats.skew(rr):7.2f}  "
              f"kurt {stats.kurtosis(rr):8.1f}  ShapiroW {stats.shapiro(sub).statistic:.4f}")


# ---------------------------------------------------------------- 5. mixture
def sec_mixture():
    _hdr("MIXTURE -- it is a scale mixture, not non-Gaussianity")
    z = _taskA()
    SATS, T = list(z["sats"]), list(z["tgt_cols"])
    B, BC = z["B"], list(z["bc_cols"])
    Y, P, Bl, si, _tt = _v2()
    r = Y - P

    print("=== 1) are residuals Gaussian WITHIN a single satellite? ===")
    print(f"{'sat':7s} {'n':>5s} " + "  ".join(f"{c + ' W':>10s}" for c in T))
    rows = []
    for k in np.unique(si):
        m = si == k
        if m.sum() < 300:
            continue
        ws = []
        for j in range(4):
            x = r[m, j]
            ws.append(stats.shapiro(x[:min(len(x), 2000)]).statistic)
        rows.append((SATS[k], m.sum(), *ws))
    R = pd.DataFrame(rows, columns=["sat", "n"] + T)
    for _, q in R.head(8).iterrows():
        print(f"{q['sat']:7s} {q['n']:5d} " + "  ".join(f"{q[c]:10.4f}" for c in T))
    print(f"{'MEDIAN':7s} {'':5s} " + "  ".join(f"{R[c].median():10.4f}" for c in T))
    print(f"{'POOLED':7s} {len(r):5d} " + "  ".join(
        f"{stats.shapiro(np.random.default_rng(0).choice(r[:,j],4000,replace=False)).statistic:10.4f}"
        for j in range(4)))

    print("\n=== 2) how much do per-satellite residual scales differ? ===")
    sd = pd.DataFrame({c: [np.std(r[si == k, j])
                           for k in np.unique(si) if (si == k).sum() >= 300]
                       for j, c in enumerate(T)})
    print(f"{'target':8s} {'min sd':>8s} {'median':>8s} {'max sd':>8s} {'max/min':>8s}")
    for c in T:
        print(f"{c:8s} {sd[c].min():8.3f} {sd[c].median():8.3f} {sd[c].max():8.3f} "
              f"{sd[c].max()/max(sd[c].min(),1e-9):8.1f}")

    print("\n=== 3) standardise each satellite by its OWN sd -> is the pool Gaussian? ===")
    print(f"{'target':8s} {'skew':>8s} {'kurt':>9s} {'ShapiroW':>10s}")
    for j, c in enumerate(T):
        zz = np.concatenate([(r[si == k, j] - r[si == k, j].mean())
                             / (r[si == k, j].std() + 1e-9)
                             for k in np.unique(si) if (si == k).sum() >= 300])
        sub = np.random.default_rng(0).choice(zz, 4000, replace=False)
        print(f"{c:8s} {stats.skew(zz):8.2f} {stats.kurtosis(zz):9.1f} "
              f"{stats.shapiro(sub).statistic:10.4f}")

    print("\n=== 4) can BROADCAST predict a satellite's residual scale? (real-time usable) ===")
    ihc = BC.index("ho_clk_m")
    ihr = BC.index("ho_R")
    feat, targ = [], []
    for k in np.unique(si):
        m = si == k
        if m.sum() < 300:
            continue
        hc = B[k, :, ihc]
        hc = hc[np.isfinite(hc)]
        hr = B[k, :, ihr]
        hr = hr[np.isfinite(hr)]
        if len(hc) < 100:
            continue
        feat.append([np.std(hc), np.percentile(np.abs(hc), 90), np.std(hr)])
        targ.append([np.std(r[m, j]) for j in range(4)])
    F, G = np.array(feat), np.array(targ)
    print(f"  n satellites = {len(F)}")
    print(f"{'target resid sd':18s} {'vs ho_clk sd':>13s} {'vs ho_clk p90':>14s} "
          f"{'vs ho_R sd':>12s}")
    for j, c in enumerate(T):
        cs = [np.corrcoef(F[:, i], G[:, j])[0, 1] for i in range(3)]
        print(f"  {c:16s} " + "  ".join(f"{v:13.3f}" for v in cs))


RUN = {"sisre": sec_sisre, "persys": sec_persys, "tails": sec_tails,
       "g27": sec_g27, "mixture": sec_mixture}

if __name__ == "__main__":
    for name in (SECTIONS if WHICH == "all" else [WHICH]):
        RUN[name]()
