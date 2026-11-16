#!/usr/bin/env python3
"""Stage 3 (rebuilt). Memory-safe sliding windows.

Materialising stride-1 windows costs ~46 GB for 97 sats x 236 days
(97 * 22000 * 672 * 8 cols * 4 B). We never materialise them. Instead we store
one dense, regular-grid array per satellite (~70 MB total) and expose windows as
zero-copy views via sliding_window_view. Only the requested batch is ever real.
"""
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

IN_LEN, OUT_LEN, STEP_S = 672, 96, 900
TARGETS = ["dR","dA","dC","dClk_ns"]
FEATS   = ["age_s","sin_u","cos_u","sin_2u"]
COLS    = TARGETS + FEATS

class SeriesStore:
    """Dense regular-grid series per satellite, with a validity mask.

    anom_z is a per-satellite robust score ((|err|-median)/MAD). Epochs above
    `anom_z_max` are treated as missing, which also prevents any window from
    spanning them. Measured on 236 days, 24h-persistence dR residuals:
        no mask   kurt=+67645 std=0.917 m
        z>20      kurt=  +624 std=0.149 m   (0.098% dropped)
        z>10      kurt=  +120 std=0.140 m   (0.438% dropped)
        z> 6      kurt=   +43 std=0.143 m   (1.518% dropped)
    Tightening past z>4 over-trims: std rises again."""
    def __init__(self, df, anom_z_max=10.0):
        df = df.sort_values(["sat","epoch"])
        t0, t1 = df.epoch.min(), df.epoch.max()
        self.t0 = t0
        self.n  = int((t1 - t0).total_seconds() // STEP_S) + 1
        self.sats = sorted(df.sat.unique())
        self.si = {s:i for i,s in enumerate(self.sats)}
        S, C = len(self.sats), len(COLS)
        self.X = np.full((S, self.n, C), np.nan, np.float32)
        self.cls = []
        for s in self.sats:
            g = df[df.sat==s]
            k = ((g.epoch - t0).dt.total_seconds() // STEP_S).astype(int).to_numpy()
            self.X[self.si[s], k, :] = g[COLS].to_numpy(np.float32)
            self.cls.append(g.orbit_class.iloc[0])
        self.cls = np.array(self.cls)
        self.valid = np.isfinite(self.X[:,:,:len(TARGETS)]).all(axis=2)   # (S,n)
        if anom_z_max is not None and "anom_z" in df.columns:
            bad = np.zeros_like(self.valid)
            for s in self.sats:
                g = df[df.sat==s]
                k = ((g.epoch - t0).dt.total_seconds() // STEP_S).astype(int).to_numpy()
                bad[self.si[s], k] = (g["anom_z"].to_numpy() > anom_z_max)
            self.valid &= ~bad

    def nbytes(self): return self.X.nbytes

    def window_index(self):
        """All (sat_i, start) with a fully-valid IN+OUT span. Cheap: uses a
        cumulative sum instead of touching each window."""
        L = IN_LEN + OUT_LEN
        cs = np.cumsum(np.concatenate([np.zeros((self.valid.shape[0],1),int),
                                       self.valid.astype(int)],axis=1),axis=1)
        full = (cs[:, L:] - cs[:, :-L]) == L          # (S, n-L+1)
        si, st = np.nonzero(full)
        return si, st

    def batch(self, si, st):
        """Materialise only these windows. Returns X (B,IN,C), Y (B,OUT,T)."""
        B = len(si); L = IN_LEN + OUT_LEN
        idx = st[:,None] + np.arange(L)[None,:]
        blk = self.X[si[:,None], idx]                  # (B, L, C)
        return blk[:, :IN_LEN, :], blk[:, IN_LEN:, :len(TARGETS)]

    def origins(self, si, st):
        return self.t0 + pd.to_timedelta((st + IN_LEN - 1) * STEP_S, unit="s")

def add_features(df):
    df = df.copy()
    # u_arg is the true argument of latitude from the ICD evaluation (phi_k plus
    # harmonic corrections), NOT geocentric latitude -- latitude is ambiguous
    # between ascending and descending passes and so aliases as a phase variable.
    u = df["u_arg"].to_numpy()
    df["sin_u"], df["cos_u"] = np.sin(u), np.cos(u)
    df["sin_2u"] = np.sin(2*u)
    df["age_s"] = df["age_s"].fillna(0)
    return df

def temporal_split(orig, train_end, val_end, gap_days=8):
    o = pd.to_datetime(pd.Series(orig)); g = pd.Timedelta(days=gap_days)
    tr = (o <  pd.Timestamp(train_end)).to_numpy()
    va = ((o >= pd.Timestamp(train_end)+g) & (o < pd.Timestamp(val_end))).to_numpy()
    te = (o >= pd.Timestamp(val_end)+g).to_numpy()
    return tr, va, te
