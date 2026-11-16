#!/usr/bin/env python3
"""Does the SHIPPED file reproduce the numbers we quote for it?

smart_horizon.py is a standalone re-implementation -- it deliberately imports
nothing from the corpus-side code. So it has to be checked against real corpus
windows through its PUBLIC API (a DataFrame of the competition's five columns),
not against the implementation it was derived from.

Also checks the thing the brief actually asks for: prediction at ARBITRARY
timestamps, not on the 15-minute grid. Query times are jittered off-grid and the
truth is interpolated to them.
"""
import sys
import time

import numpy as np
import pandas as pd

from taskB_final import XYZ, CLS, S, NT, WIN, HOR, STEP, thin
from smart_horizon import predict

T0 = pd.Timestamp("2025-01-01")
COLS = ["x_error", "y_error", "z_error", "satclockerror"]


def one(s, o, rg, offgrid):
    """Build one call exactly as a user would, and score what comes back."""
    w, f = XYZ[s, o - WIN:o], XYZ[s, o:o + HOR]
    if np.isfinite(w[:, 0]).mean() < 0.5 or np.isfinite(f[:, 0]).mean() < 0.5:
        return None
    iw = np.nonzero(np.isfinite(w[:, 0]))[0]
    tw, vw = thin((iw - WIN) * float(STEP), w[iw], 0.30, rg)
    if len(tw) < 30:
        return None
    iq = np.nonzero(np.isfinite(f[:, 0]))[0]
    if len(iq) < 8:
        return None

    tr = pd.DataFrame({"utc_time": T0 + pd.to_timedelta(tw, "s")})
    for j, c in enumerate(COLS):
        tr[c] = vw[:, j]

    tq = iq * float(STEP)
    if offgrid:
        # jitter off the grid and interpolate truth to the jittered times, so the
        # query epochs are ones the model has never been handed before
        j_ = rg.uniform(-400, 400, len(tq))
        tq2 = np.clip(tq + j_, 0.0, (HOR - 1) * float(STEP))
        truth = np.column_stack([np.interp(tq2, tq, f[iq][:, k]) for k in range(4)])
        tq = tq2
    else:
        truth = f[iq]

    geo = CLS[s] in ("GEO", "GSO")
    out = predict(tr, T0 + pd.to_timedelta(tq, "s"), orbit="GEO" if geo else "MEO")
    P = out[COLS].to_numpy(float)
    sig = out[[c + "_sigma" for c in COLS]].to_numpy(float)[0]
    return geo, ((truth - P) ** 2).sum(0), (truth ** 2).sum(0), len(tq), sig


def run(n, offgrid, seed):
    rg = np.random.default_rng(seed)
    acc = {True: [np.zeros(4), np.zeros(4), 0, []], False: [np.zeros(4), np.zeros(4), 0, []]}
    got = 0
    t0 = time.time()
    while got < n:
        r = one(int(rg.integers(0, S)), int(rg.integers(WIN, NT - HOR)), rg, offgrid)
        if r is None:
            continue
        geo, sse, s0, m, sig = r
        a = acc[geo]
        a[0] += sse; a[1] += s0; a[2] += m; a[3].append(sig)
        got += 1
    print(f"  ({got} windows in {time.time()-t0:.0f}s)")
    for geo, nm in ((True, "GEO/GSO"), (False, "MEO")):
        sse, s0, m, sg = acc[geo]
        if m == 0:
            continue
        rms = np.sqrt(sse / m)
        raw = np.sqrt(s0 / m)
        print(f"  {nm:8s} n={m:6d}  " + "  ".join(f"{c.split('_')[0]} {v:.3f}"
                                                  for c, v in zip(COLS, rms))
              + f"  | mean {rms.mean():.3f}  (do-nothing {raw.mean():.3f}, "
              f"{100*(1-rms.mean()/raw.mean()):.1f}% better)")
        sg = np.array(sg).mean(0)
        print(f"           reported 1-sigma  " + "  ".join(f"{v:.3f}" for v in sg)
              + f"   ratio actual/reported " + "  ".join(f"{a/b:.2f}"
                                                         for a, b in zip(rms, sg)))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print("=" * 78)
    print("ON-GRID query times (15-min), as in all earlier validation")
    print("=" * 78)
    run(N, False, 11)
    print()
    print("=" * 78)
    print("ARBITRARY off-grid query times (jittered +-400 s, truth interpolated)")
    print("=" * 78)
    run(N, True, 12)
