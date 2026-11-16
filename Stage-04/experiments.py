#!/usr/bin/env python3
"""Every experiment in Stage-04, as reusable functions.

    python experiments.py            # runs all of them, prints tables

Inputs are the delivered SH-DST-03 files
(`data/`) and signals generated from orbital mechanics (`synth.py`).

  exp_recovery       does the model recover a day-8 truth it has never seen?
  exp_sensitivity    how does performance move as noise / gaps / artefact vary?
  exp_artifact       inject the GEO artefact into clean data, what breaks?
  exp_benchmark      rank methods using ONLY the delivered window (day 7 held out)
  exp_metric         what the Shapiro-Wilk score actually responds to
  exp_ramp           the day-by-day volatility ramp in GEO_Train
  exp_sih            the model vs baselines on the delivered files
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import clean
import synth
from clean import COLS
from sw_score import score, swtest
from model import predict

DAY = 86400.0


# ============================================================ 1. RECOVERY
def exp_recovery(n=60, orbits=("GEO", "MEO"), seed0=0, verbose=True):
    """Generate a window, hide day 8, ask the model for it, compare to truth.

    This is the central validation. The model is handed seven days of a signal
    built from orbital mechanics plus noise, and asked for day 8 at 96 times it
    has never seen. We know the true answer exactly.
    """
    rows = []
    for orbit in orbits:
        for k in range(n):
            tr, q, truth, meta = synth.make_window(orbit, seed=seed0 + k)
            if len(tr) < 25:
                continue
            P = predict(tr, q, orbit=orbit)[COLS].to_numpy(float)
            T = truth
            r_model = np.sqrt(((P - T) ** 2).mean(0))
            r_zero = np.sqrt((T ** 2).mean(0))
            r_mean = np.sqrt(((tr[COLS].mean().to_numpy() - T) ** 2).mean(0))
            rows.append({"orbit": orbit, "seed": seed0 + k, "n_train": len(tr),
                         "model": r_model.mean(), "zero": r_zero.mean(),
                         "window_mean": r_mean.mean(),
                         **{f"model_{c}": v for c, v in zip(COLS, r_model)}})
    R = pd.DataFrame(rows)
    if verbose:
        print("=" * 76)
        print(f"EXPERIMENT 1, recovery of a known day 8 ({n} windows per class)")
        print("=" * 76)
        g = R.groupby("orbit")[["zero", "window_mean", "model"]].agg(["mean", "std"])
        print(g.round(4).to_string())
        print()
        for o, gg in R.groupby("orbit"):
            imp = 100 * (1 - gg.model.mean() / gg.zero.mean())
            beat = 100 * (gg.model < gg.window_mean).mean()
            print(f"  {o:4s}  {imp:5.1f}% better than doing nothing   "
                  f"beats the 7-day average in {beat:.0f}% of windows")
    return R


# ========================================================= 2. SENSITIVITY
def exp_sensitivity(n=40, verbose=True):
    """Sweep noise, coverage and artefact level. Where does the model break?"""
    rows = []
    for noise in [0.01, 0.05, 0.15, 0.40, 1.0]:
        for k in range(n):
            tr, q, truth, _ = synth.make_window("MEO", seed=k, noise=noise)
            if len(tr) < 25:
                continue
            P = predict(tr, q, orbit="MEO")[COLS].to_numpy(float)
            rows.append({"knob": "noise", "value": noise,
                         "model": np.sqrt(((P - truth) ** 2).mean(0)).mean(),
                         "zero": np.sqrt((truth ** 2).mean(0)).mean()})
    for frac in [0.0, 0.1, 0.2, 0.35, 0.5]:
        for k in range(n):
            tr, q, truth, _ = synth.make_window("MEO", seed=k,
                                                artifact=frac > 0)
            if frac > 0:
                V, _ = synth.inject_artifact(tr[COLS].to_numpy(float), frac=frac, seed=k)
                for j, c in enumerate(COLS):
                    tr[c] = V[:, j]
            if len(tr) < 25:
                continue
            P = predict(tr, q, orbit="MEO")[COLS].to_numpy(float)
            rows.append({"knob": "artifact", "value": frac,
                         "model": np.sqrt(((P - truth) ** 2).mean(0)).mean(),
                         "zero": np.sqrt((truth ** 2).mean(0)).mean()})
    R = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 76)
        print("EXPERIMENT 2, sensitivity: what degrades the model, and how fast")
        print("=" * 76)
        for knob, g in R.groupby("knob"):
            print(f"\n  {knob}:")
            t = g.groupby("value")[["zero", "model"]].mean()
            t["% better"] = 100 * (1 - t.model / t.zero)
            print("   " + t.round(4).to_string().replace("\n", "\n   "))
    return R


# ============================================================ 3. ARTEFACT
def exp_artifact(n=40, verbose=True):
    """Inject the delivered GEO file's artefact into clean data.

    The delivered GEO file alternates sign on its large excursions, 24 of 24 in
    the test file. If we inject exactly that into a signal that was clean a
    moment ago, does a working model collapse the way it does on their file?
    """
    rows = []
    for k in range(n):
        tr, q, truth, _ = synth.make_window("GEO", seed=k)
        if len(tr) < 25:
            continue
        P0 = predict(tr, q, orbit="GEO")[COLS].to_numpy(float)
        tr2 = tr.copy()
        V, idx = synth.inject_artifact(tr2[COLS].to_numpy(float), frac=0.35, seed=k)
        for j, c in enumerate(COLS):
            tr2[c] = V[:, j]
        P1 = predict(tr2, q, orbit="GEO")[COLS].to_numpy(float)
        rows.append({"clean": np.sqrt(((P0 - truth) ** 2).mean(0)).mean(),
                     "with_artifact": np.sqrt(((P1 - truth) ** 2).mean(0)).mean(),
                     "zero": np.sqrt((truth ** 2).mean(0)).mean()})
    R = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 76)
        print("EXPERIMENT 3, inject the GEO artefact into clean synthetic data")
        print("=" * 76)
        m = R.mean()
        print(f"  clean signal          : model RMS {m.clean:.4f}   "
              f"({100*(1-m.clean/m.zero):.0f}% better than doing nothing)")
        print(f"  same signal + artefact: model RMS {m.with_artifact:.4f}   "
              f"({100*(1-m.with_artifact/m.zero):.0f}% better)")
        print(f"  -> the artefact alone destroys {100*(m.with_artifact/m.clean-1):.0f}% "
              f"of the model's accuracy on data that was clean a moment earlier.")
    return R


# =========================================================== 4. BENCHMARK
def _cands(t_tr, y_tr, t_q, orbit):
    """A menu of forecasting methods, all fitted on the same data."""
    out = {}
    out["zero"] = np.zeros(len(t_q))
    out["mean"] = np.full(len(t_q), y_tr.mean())
    out["median"] = np.full(len(t_q), np.median(y_tr))
    out["last value"] = np.full(len(t_q), y_tr[-1])
    for lbl, span in [("mean last 1d", DAY), ("mean last 2d", 2 * DAY),
                      ("mean last 3d", 3 * DAY)]:
        m = t_tr >= t_tr.max() - span
        out[lbl] = np.full(len(t_q), y_tr[m].mean() if m.sum() >= 3 else y_tr.mean())
    for tau, lbl in [(DAY, "ewma 1d"), (2 * DAY, "ewma 2d")]:
        w = np.exp(-(t_tr.max() - t_tr) / tau)
        out[lbl] = np.full(len(t_q), (w * y_tr).sum() / w.sum())
    # linear / quadratic extrapolation
    for deg, lbl in [(1, "linear trend"), (2, "quadratic trend")]:
        try:
            c = np.polyfit(t_tr, y_tr, deg)
            out[lbl] = np.polyval(c, t_q)
        except Exception:
            out[lbl] = np.full(len(t_q), y_tr.mean())
    # seasonal naive at the physically correct period
    P = synth.T_SID if str(orbit).upper().startswith("G") else 43082.0
    idx = np.array([np.argmin(np.abs(t_tr - (tq - P))) for tq in t_q])
    out["seasonal naive"] = y_tr[idx]
    # fixed harmonic regression, no selection
    from model import _design, _ridge
    for K in (1, 2, 3):
        A = _design(t_tr, [P], K)
        c = _ridge(A, y_tr)
        out[f"harmonic K={K}"] = _design(t_q, [P], K) @ c if c is not None \
            else np.full(len(t_q), y_tr.mean())
    return out


def exp_benchmark(verbose=True):
    """Rank methods using ONLY the delivered data, fit days 1-6, score day 7.

    The delivered files give three windows, which is far too few to benchmark on
    day 8. But every window contains its own held-out day: fit on days 1-6 and
    score on day 7. That is a fair comparison, uses no outside data, and is the
    same signal the model itself selects on.
    """
    rows = []
    for key, *_ in clean.PAIRS:
        tr, te, orbit = clean.load_pair(key)
        t = (tr.utc_time - tr.utc_time.min()).dt.total_seconds().to_numpy(float)
        cut = t.min() + (t.max() - t.min()) * 6.0 / 7.0
        a, b = t <= cut, t > cut
        if a.sum() < 12 or b.sum() < 4:
            continue
        for j, c in enumerate(COLS):
            y = tr[c].to_numpy(float)
            preds = _cands(t[a], y[a], t[b], orbit)
            for name, p in preds.items():
                rows.append({"pair": key, "channel": c, "method": name,
                             "rms": float(np.sqrt(np.mean((p - y[b]) ** 2)))})
        # our model, same protocol
        sub = tr[a].copy()
        if len(sub) >= 20:
            P = predict(sub, tr.loc[b, "utc_time"], orbit=orbit)
            for c in COLS:
                rows.append({"pair": key, "channel": c, "method": "SMART-HORIZON",
                             "rms": float(np.sqrt(np.mean(
                                 (P[c].to_numpy() - tr.loc[b, c].to_numpy()) ** 2)))})
    R = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 76)
        print("EXPERIMENT 4, method benchmark inside the delivered data")
        print("   (fit days 1-6, score day 7; no outside data used)")
        print("=" * 76)
        piv = R.pivot_table(index="method", columns="pair", values="rms")
        piv["MEO avg"] = piv[["MEO-1", "MEO-2"]].mean(axis=1)
        piv = piv.sort_values("MEO avg")
        print(piv.round(4).to_string())
        print("\n  Ranked on the MEO average, the GEO column is unpredictable for")
        print("  every method (see experiment 7), so it swamps any aggregate.")
    return R


# ============================================================== 5. METRIC
def exp_metric(verbose=True):
    """What does the Shapiro-Wilk score actually respond to?"""
    tr, te, orbit = clean.load_pair("GEO")
    T = te[COLS].reset_index(drop=True)
    rng = np.random.default_rng(0)
    sd = tr[COLS].to_numpy(float).std()
    rows = []

    def add(lbl, P):
        per, s = score(pd.DataFrame(P, columns=COLS) - T)
        rows.append({"prediction": lbl, "W": s["W"], "RMS": s["rms"]})

    add("predict zero", np.zeros(T.shape))
    add("predict the 7-day average", np.tile(tr[COLS].mean().to_numpy(), (len(T), 1)))
    add("predict the last value", np.tile(tr[COLS].to_numpy()[-1], (len(T), 1)))
    for f in [0.5, 0.9, 0.99]:
        add(f"CHEAT: {int(f*100)}% accurate", f * T.to_numpy())
    for a in [1, 2, 4, 12]:
        add(f"pure random noise x{a}", rng.normal(0, a * sd, T.shape))
    R = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 76)
        print("EXPERIMENT 5, what the score responds to (on the delivered GEO file)")
        print("=" * 76)
        print(R.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        print("\n  Higher W is better. Note that accuracy does not move it and noise does.")
    return R


# ================================================================ 6. RAMP
def exp_ramp(verbose=True):
    """The day-by-day volatility ramp inside GEO_Train."""
    tr = clean.load("DATA_GEO_Train.csv")
    tr = tr.copy()
    tr["day"] = ((tr.utc_time - tr.utc_time.min()).dt.total_seconds() // DAY).astype(int) + 1
    g = tr.groupby("day")[COLS].std()
    g.insert(0, "n", tr.groupby("day").size())
    te = clean.load("DATA_GEO_Test.csv")
    g.loc[8] = [len(te)] + list(te[COLS].std().to_numpy())
    if verbose:
        print("\n" + "=" * 76)
        print("EXPERIMENT 6, GEO_Train volatility, day by day (day 8 = the test file)")
        print("=" * 76)
        print(g.round(3).to_string())
        r = g.loc[8, "y_error"] / g.loc[1, "y_error"]
        print(f"\n  y_error spread grows {r:.1f}x from day 1 to day 8.")
    return g


# ================================================================= 7. SIH
def exp_sih(verbose=True):
    """The model vs baselines on the delivered files."""
    rows = []
    for key, *_ in clean.PAIRS:
        tr, te, orbit = clean.load_pair(key)
        T = te[COLS].reset_index(drop=True)
        preds = {
            "do nothing": np.zeros(T.shape),
            "7-day average": np.tile(tr[COLS].mean().to_numpy(), (len(T), 1)),
            "last value": np.tile(tr[COLS].to_numpy()[-1], (len(T), 1)),
            "SMART-HORIZON": predict(tr, te.utc_time, orbit=orbit)[COLS].to_numpy(),
            "CHEAT: test mean": np.tile(T.mean().to_numpy(), (len(T), 1)),
        }
        for name, P in preds.items():
            per, s = score(pd.DataFrame(P, columns=COLS) - T)
            rows.append({"pair": key, "orbit": orbit, "n_test": len(te),
                         "method": name, "RMS": s["rms"], "W": s["W"],
                         "p": s["p"], "rejects": s["n_reject"],
                         "mean": s["mean"], "sd": s["sd"]})
    R = pd.DataFrame(rows)
    if verbose:
        print("\n" + "=" * 76)
        print("EXPERIMENT 7, the model on the delivered SH-DST-03 files")
        print("=" * 76)
        print(R.pivot(index="method", columns="pair", values="RMS")
              .round(4).to_string())
        print("\nShapiro-Wilk W (their Priority 1):")
        print(R.pivot(index="method", columns="pair", values="W")
              .round(4).to_string())
    return R


if __name__ == "__main__":
    exp_ramp()
    exp_sih()
    exp_recovery(n=40)
    exp_artifact(n=30)
    exp_sensitivity(n=25)
    exp_benchmark()
    exp_metric()


# ============================================== 8. WHY PURE ML DOES NOT WORK
def exp_ml_fails(n_windows=400, verbose=True):
    """Train ordinary ML on the ACTUAL task, and watch it fail to beat a mean.

    THE TASK, STATED EXACTLY
    Given seven days of error values, predict day 8 at arbitrary timestamps.
    The crucial detail is that day 8 lies 0-24 hours BEYOND the last observation.
    A learner therefore cannot lean on "what happened five minutes ago", that
    information does not exist at prediction time. (An earlier version of this
    experiment let the model see the immediately preceding sample and it duly
    beat the baselines; that was next-step interpolation, not the competition's
    task, and it is why the distinction matters.)

    THE SETUP, done the way a practitioner would
    Build a training set of many windows so there is genuinely something to learn
    from. For each window, engineer the features available from five columns,
    the level, spread, trend and shape of the seven days, plus the query time and
    how far ahead it is. Train six standard regressors. Test on windows never
    seen in training.

    WHY THEY FAIL, and this is the whole motivation for what follows
    Supervised learning maps inputs to outputs. Here the inputs do not determine
    the output. What actually decides day 8 is where the satellite sits in its
    orbit, which constellation it belongs to, and when the ground segment last
    uploaded, and none of those are columns in the file. The learner is not
    underpowered; it is being asked to infer something that carries almost no
    mutual information with anything it can see.

    That is the argument for physics. The orbital rhythm is not a pattern we
    found inside the data, it is information we brought in from outside it,
    which is exactly why it works where learning cannot.
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    def window_feats(t, y):
        """Everything summarisable from one 7-day channel."""
        last1 = y[t >= t.max() - DAY]
        last2 = y[t >= t.max() - 2 * DAY]
        sl = np.polyfit(t, y, 1)[0] if len(t) > 3 else 0.0
        return [y.mean(), y.std(), np.median(y), y.min(), y.max(),
                last1.mean() if len(last1) else y.mean(),
                last1.std() if len(last1) > 1 else y.std(),
                last2.mean() if len(last2) else y.mean(),
                y[-1], y[-1] - y[0], sl * DAY,
                float(stats_skew(y)), float(len(y))]

    def stats_skew(v):
        m = v.mean(); s = v.std()
        return 0.0 if s <= 0 else float(((v - m) ** 3).mean() / s ** 3)

    X, Y, G = [], [], []
    for w in range(n_windows):
        orbit = "MEO" if w % 2 else "GEO"
        tr, q, truth, meta = synth.make_window(orbit, seed=2000 + w)
        t = meta["t_train_s"]
        tq = meta["t_query_s"]
        if len(t) < 25:
            continue
        for j, c in enumerate(COLS):
            y = tr[c].to_numpy(float)
            wf = window_feats(t, y)
            for k in range(0, len(tq), 4):                # subsample day-8 times
                ahead = (tq[k] - t.max()) / 3600.0        # hours beyond last data
                X.append(wf + [ahead, (tq[k] % DAY) / DAY,
                               np.sin(2 * np.pi * tq[k] / DAY),
                               np.cos(2 * np.pi * tq[k] / DAY),
                               1.0 if orbit == "GEO" else 0.0, float(j)])
                Y.append(truth[k, j]); G.append(w)
    X, Y, G = np.array(X), np.array(Y), np.array(G)
    ok = np.isfinite(X).all(1) & np.isfinite(Y)
    X, Y, G = X[ok], Y[ok], G[ok]

    cut = int(G.max() * 0.7)
    trm, tem = G < cut, G >= cut
    Xtr, Ytr, Xte, Yte = X[trm], Y[trm], X[tem], Y[tem]

    rms = lambda p: float(np.sqrt(np.mean((p - Yte) ** 2)))
    rows = [{"model": "predict the 7-day mean", "kind": "baseline", "rms": rms(Xte[:, 0])},
            {"model": "predict the last value", "kind": "baseline", "rms": rms(Xte[:, 8])},
            {"model": "predict zero", "kind": "baseline", "rms": rms(np.zeros(len(Yte)))}]
    models = {
        "Ridge regression": Ridge(alpha=1.0),
        "k-nearest neighbours": KNeighborsRegressor(n_neighbors=25),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                               random_state=0, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(random_state=0),
        "Support vector regression": SVR(C=1.0),
        "Neural network (MLP)": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600,
                                             random_state=0),
    }
    for name, m in models.items():
        try:
            m.fit(Xtr, Ytr)
            rows.append({"model": name, "kind": "learned", "rms": rms(m.predict(Xte))})
        except Exception:
            rows.append({"model": name, "kind": "learned", "rms": np.nan})

    # and our physics model, on the same held-out windows
    ph = []
    for w in range(cut, int(G.max()) + 1):
        orbit = "MEO" if w % 2 else "GEO"
        tr, q, truth, meta = synth.make_window(orbit, seed=2000 + w)
        if len(tr) < 25:
            continue
        P = predict(tr, q, orbit=orbit)[COLS].to_numpy(float)
        ph.append(np.sqrt(((P - truth) ** 2).mean()))
    rows.append({"model": "SMART-HORIZON (physics)", "kind": "physics",
                 "rms": float(np.mean(ph))})

    R = pd.DataFrame(rows).sort_values("rms")
    if verbose:
        print("\n" + "=" * 76)
        print("EXPERIMENT 8, ordinary ML on the actual task (predict day 8, not the next sample)")
        print(f"   {len(np.unique(G))} windows, {len(Xtr):,} training rows, 19 features")
        print("=" * 76)
        print(R.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        best = R[R.kind == "learned"].rms.min()
        bl = R[R.kind == "baseline"].rms.min()
        print(f"\n  best learner {best:.4f}   best trivial baseline {bl:.4f}"
              f"   ->  learning {'HELPS' if best < bl else 'DOES NOT HELP'}")
    return R
