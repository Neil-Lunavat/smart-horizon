#!/usr/bin/env python3
"""What does it actually predict? One real window, side by side with truth."""
import numpy as np
import pandas as pd

from taskB_final import XYZ, CLS, SATS, S, NT, WIN, HOR, STEP, thin
from smart_horizon import predict

T0 = pd.Timestamp("2025-06-01")
COLS = ["x_error", "y_error", "z_error", "satclockerror"]


def demo(want_geo, seed):
    rg = np.random.default_rng(seed)
    ok = [i for i in range(S) if (CLS[i] in ("GEO", "GSO")) == want_geo]
    while True:
        s = int(ok[rg.integers(0, len(ok))])
        o = int(rg.integers(WIN, NT - HOR))
        w, f = XYZ[s, o - WIN:o], XYZ[s, o:o + HOR]
        if np.isfinite(w[:, 0]).mean() < 0.7 or np.isfinite(f[:, 0]).mean() < 0.7:
            continue
        iw = np.nonzero(np.isfinite(w[:, 0]))[0]
        tw, vw = thin((iw - WIN) * float(STEP), w[iw], 0.30, rg)
        iq = np.nonzero(np.isfinite(f[:, 0]))[0]
        if len(tw) >= 60 and len(iq) >= 40:
            break

    tr = pd.DataFrame({"utc_time": T0 + pd.to_timedelta(tw, "s")})
    for j, c in enumerate(COLS):
        tr[c] = vw[:, j]
    tq = iq * float(STEP)
    truth = f[iq]
    out = predict(tr, T0 + pd.to_timedelta(tq, "s"),
                  orbit="GEO" if want_geo else "MEO")

    cls = CLS[s]
    print(f"\n{'='*94}")
    print(f"{SATS[s]}  ({cls})   input: {len(tw)} rows over 7 days "
          f"(irregular, ~{100*len(tw)/WIN:.0f}% of a 15-min grid)")
    print(f"{'='*94}")
    print(f"{'day-8 time':>12s} |" + "".join(
        f"{c.replace('_error','').replace('satclock','clk'):>22s}" for c in COLS))
    print(f"{'':>12s} |" + "".join(f"{'pred':>8s}{'true':>7s}{'err':>7s}" for _ in COLS))
    step = max(len(iq) // 12, 1)
    for i in range(0, len(iq), step):
        hh = tq[i] / 3600.0
        row = f"{hh:9.2f} h |"
        for j in range(4):
            p, t = out[COLS[j]].iloc[i], truth[i, j]
            row += f"{p:8.3f}{t:7.3f}{p-t:7.3f}"
        print(row)
    rms = np.sqrt(((truth - out[COLS].to_numpy()) ** 2).mean(0))
    raw = np.sqrt((truth ** 2).mean(0))
    sig = out[[c + "_sigma" for c in COLS]].to_numpy()[0]
    print("-" * 94)
    print(f"{'RMS':>12s} |" + "".join(f"{v:22.3f}" for v in rms))
    print(f"{'do nothing':>12s} |" + "".join(f"{v:22.3f}" for v in raw))
    print(f"{'reported 1s':>12s} |" + "".join(f"{v:22.3f}" for v in sig))
    print(f"{'':>12s}  mean RMS {rms.mean():.3f} m   vs {raw.mean():.3f} m "
          f"doing nothing   ({100*(1-rms.mean()/raw.mean()):.0f}% better)")


if __name__ == "__main__":
    demo(True, 5)
    demo(False, 11)
