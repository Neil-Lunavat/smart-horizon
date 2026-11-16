#!/usr/bin/env python3
"""Stage-03 headline: run the Stage-02 predictor on the actual PS-08 data.

    python apply_model.py

THE THING TO BE CLEAR ABOUT FIRST
There is no "trained model" to transfer.  `smart_horizon.py` has ZERO trained
parameters -- no weights, no corpus, no network.  The 244-day corpus decided
WHICH basis is correct for each orbit class (sidereal harmonics for GEO,
Earth-rotation beat frequencies for MEO, level estimators only for the clock);
the coefficients themselves are re-fitted inside whatever 7-day window the caller
supplies, every call.  So "applying the corpus model to the SIH data" is not a
transfer step at all -- it is one function call, and there is no distribution
shift to worry about because there is nothing fitted to be shifted.  The single
empirical constant, SIGMA_CAL = 1.25, scales the reported uncertainty only.

That is the whole reason this is even runnable: a weights-based Task-A-style
model could not be evaluated on these files at all (no PRN, no broadcast, no
SP3), which is exactly why Stage-02 was abandoned as a deliverable.

SCORED TWO WAYS, BECAUSE THEY DISAGREE
  RMS       what we optimised in Stage-02 and what a user feels
  swtest W  what Note.pdf actually grades on -- normality of the residual,
            scale-invariant, so it says nothing about accuracy

Baselines are the same ones Stage-02 used, so the numbers are comparable:
zero (do nothing), the 7-day window mean, and last-value carry-forward.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sih
from sw_score import score, ci_W, COLS
from smart_horizon import predict

OUT = "out"


# ------------------------------------------------------------------ baselines
def bl_zero(tr, q):
    return pd.DataFrame({c: np.zeros(len(q)) for c in COLS})


def bl_mean(tr, q):
    return pd.DataFrame({c: np.full(len(q), tr[c].mean()) for c in COLS})


def bl_last(tr, q):
    last = tr.sort_values("utc_time").iloc[-1]
    return pd.DataFrame({c: np.full(len(q), last[c]) for c in COLS})


def bl_model(tr, q, orbit):
    r = predict(tr, q, orbit=orbit)
    return r[COLS].reset_index(drop=True)


METHODS = ["zero", "window mean", "last value", "SMART-HORIZON"]


def run_pair(key, verbose=True):
    tr, te, orbit = sih.load_pair(key)
    q = te["utc_time"]
    truth = te[COLS].reset_index(drop=True)

    preds = {"zero": bl_zero(tr, q), "window mean": bl_mean(tr, q),
             "last value": bl_last(tr, q),
             "SMART-HORIZON": bl_model(tr, q, orbit)}

    rows, pers = [], {}
    for name, P in preds.items():
        resid = P - truth                      # Note.pdf: predicted - test
        per, s = score(resid)
        pers[name] = per
        rows.append({"pair": key, "orbit": orbit, "n_test": len(te),
                     "method": name, "RMS": s["rms"], "W": s["W"], "p": s["p"],
                     "H_rejects": s["n_reject"], "mean": s["mean"], "sd": s["sd"]})
    res = pd.DataFrame(rows)

    if verbose:
        print(f"\n{'='*78}\n{key}  (orbit={orbit}, train n={len(tr)}, test n={len(te)}, "
              f"horizon {(te.utc_time.min()-tr.utc_time.max()).total_seconds()/3600:.1f} h)\n{'='*78}")
        print(res[["method", "RMS", "W", "p", "H_rejects", "mean", "sd"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        print("\n  per-parameter, SMART-HORIZON:")
        pm = pers["SMART-HORIZON"]
        print("   " + pm[["param", "n", "rms", "W", "p", "H", "which"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.4f}").replace("\n", "\n   "))
    return res, pers, preds, truth, te


def main():
    os.makedirs(OUT, exist_ok=True)
    allres, dumps = [], []
    for key, *_ in sih.PAIRS:
        res, pers, preds, truth, te = run_pair(key)
        allres.append(res)
        d = te[["utc_time"]].copy()
        for c in COLS:
            d["truth_" + c] = truth[c].to_numpy()
            d["pred_" + c] = preds["SMART-HORIZON"][c].to_numpy()
        d.insert(0, "pair", key)
        dumps.append(d)

    R = pd.concat(allres, ignore_index=True)
    R.to_csv(f"{OUT}/scores.csv", index=False)
    pd.concat(dumps, ignore_index=True).to_csv(f"{OUT}/predictions.csv", index=False)

    print(f"\n{'='*78}\nSUMMARY -- all three pairs\n{'='*78}")
    piv = R.pivot(index="method", columns="pair", values="RMS").reindex(METHODS)
    pivW = R.pivot(index="method", columns="pair", values="W").reindex(METHODS)
    print("RMS (m), lower is better:")
    print(piv.to_string(float_format=lambda v: f"{v:8.4f}"))
    print("\nswtest W, HIGHER is better (this is Priority 1):")
    print(pivW.to_string(float_format=lambda v: f"{v:8.4f}"))

    print("\nMEO only (the two pairs whose data survives diagnosis):")
    meo = R[R.pair.str.startswith("MEO")]
    print(meo.groupby("method")[["RMS", "W"]].mean().reindex(METHODS)
          .to_string(float_format=lambda v: f"{v:8.4f}"))

    # Priority 1 asks for a confidence interval on W.
    print("\n95% bootstrap CI on W, SMART-HORIZON residual (Note.pdf asks for this):")
    for key, *_ in sih.PAIRS:
        tr, te, orbit = sih.load_pair(key)
        P = bl_model(tr, te["utc_time"], orbit)
        r = (P - te[COLS].reset_index(drop=True))
        for c in COLS:
            lo, hi = ci_W(r[c].to_numpy())
            print(f"  {key:6s} {c:14s} n={len(r):3d}  CI = [{lo:.3f}, {hi:.3f}]"
                  if np.isfinite(lo) else f"  {key:6s} {c:14s} n={len(r):3d}  CI = n/a")
    print(f"\nwrote {OUT}/scores.csv and {OUT}/predictions.csv")


if __name__ == "__main__":
    main()
