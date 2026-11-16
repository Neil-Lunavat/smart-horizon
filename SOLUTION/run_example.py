#!/usr/bin/env python3
"""SMART-HORIZON -- runnable example.

    python run_example.py

Loads two real 7-day windows (one GEO, one MEO) from example_data/, predicts
day 8 at the supplied timestamps, and scores the result against the held-back
truth. Needs nothing but numpy and pandas.
"""
import numpy as np
import pandas as pd

from smart_horizon import predict

COLS = ["x_error", "y_error", "z_error", "satclockerror"]


def run(tag, orbit):
    hist = pd.read_csv(f"example_data/{tag}_history.csv")
    query = pd.read_csv(f"example_data/{tag}_query.csv")
    truth = pd.read_csv(f"example_data/{tag}_truth.csv")

    pred = predict(hist, query["utc_time"], orbit=orbit)

    P = pred[COLS].to_numpy(float)
    T = truth[COLS].to_numpy(float)
    sigma = pred[[c + "_sigma" for c in COLS]].to_numpy(float)[0]
    rms = np.sqrt(((P - T) ** 2).mean(0))
    raw = np.sqrt((T ** 2).mean(0))

    print(f"\n{'=' * 78}")
    print(f"{orbit}   input {len(hist)} rows over 7 days   ->   {len(query)} day-8 predictions")
    print("=" * 78)
    hdr = ["x", "y", "z", "clock"]
    print(f"{'day-8 time':>12s}  " + "".join(f"{c:>11s}" for c in hdr))
    t0 = pd.to_datetime(query['utc_time']).iloc[0]
    for i in range(0, len(query), max(len(query) // 8, 1)):
        hh = (pd.to_datetime(query["utc_time"]).iloc[i] - t0).total_seconds() / 3600
        print(f"{hh:9.2f} h  " + "".join(f"{P[i, j]:11.3f}" for j in range(4)))
    print("-" * 78)
    for lbl, v in (("RMS error", rms), ("if you predicted 0", raw),
                   ("reported 1-sigma", sigma)):
        print(f"{lbl:>18s}  " + "".join(f"{x:11.3f}" for x in v))
    print(f"{'':>18s}  mean RMS {rms.mean():.3f} m  vs  {raw.mean():.3f} m "
          f"({100 * (1 - rms.mean() / raw.mean()):.0f}% better)")
    return pred


if __name__ == "__main__":
    run("geo", "GEO")
    out = run("meo", "MEO")
    out.to_csv("example_output.csv", index=False)
    print("\nfull prediction table written to example_output.csv")
