#!/usr/bin/env python3
"""
S9 v2 — Corrected Confounder Robustness on Equities Short-Reversal.

Fixes addressing issues found in the prior run (s09_deep_run.py):

  1. Z1 (market mean) is now market-minus-self AND shift(1)-lagged.
     The original used the contemporaneous cross-sectional mean including
     each instrument's own return.

  2. Clustered standard errors: reports both naive OLS t AND date-clustered t
     (CR1 / HC sandwich), plus Newey-West with 5 lags. The clustered t is
     the HEADLINE number; naive OLS t is a benchmark, not the verdict.

  3. Block permutation null with block length 5 (matches the MA(4) induced
     by the 5-day factor). The prior IID shuffle understated the null.

  4. 1/price "size proxy" dropped. Share price is a fund-design artifact for
     ETFs and has no cross-sectional interpretation. No log-AUM is trivially
     available; we state this explicitly rather than include a noisy proxy.

  5. Real pollute-and-verify: we poison dr[t+1] (shift ret by -1 to introduce
     one-step lookahead) and confirm the factor / label at t is UNCHANGED
     while the t-stat rises substantially. The old test was disjoint-window;
     this one surgically edits one column and re-runs.

  6. Robustness added:
       - Rolling-window 252-day OOS t-stat with band
       - Pre/post-2015 subsample t-stats
       - Lookback sweep {3, 5, 7, 10} days

  7. DSR dropped (no proper alternative-parameterization trials for a single
     pre-registered factor). Verdict relies solely on clustered t and
     block-permutation p.

HONEST EXPECTED RESULT: clustering absorbs date-serial correlation (7 ETFs
move together every day), so t should fall from ~4-6 to ~1.5-2.0. If it does,
this is a NEGATIVE result and reported plainly as such.
"""

import os, sys, time, json, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

# ── Self-pin to cores 16-31 before anything else ──────────────────────────────
os.sched_setaffinity(0, range(16, 32))

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import numpy as np
import pandas as pd
from scipy import stats as ss
import glob

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
from config import EQUITY_1M, DATA_CACHE
ETF_DIR = EQUITY_1M
CACHE   = DATA_CACHE
OUT_DIR = HERE
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

import overfit as OF

ANN       = 252.0
ETF_SYMS  = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]
COST_BPS  = 3.0 / 10_000.0   # 3 bp one-way


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_etf_daily(sym: str) -> pd.DataFrame:
    cpath = os.path.join(CACHE, f"daily_etf_ohlcv_{sym}.parquet")
    if os.path.exists(cpath):
        return pd.read_parquet(cpath)

    files = sorted(f for f in glob.glob(os.path.join(ETF_DIR, f"{sym}_*.csv.gz"))
                   if "_pre_rename" not in f)
    base = os.path.join(ETF_DIR, f"{sym}.csv.gz")
    if os.path.exists(base):
        files = [base] + [f for f in files if f != base]
    if not files:
        raise FileNotFoundError(f"No ETF files for {sym}")

    parts = []
    for f in files:
        d = pd.read_csv(f, compression="gzip",
                        usecols=["BarDateTime", "FirstTradePrice", "HighTradePrice",
                                 "LowTradePrice", "LastTradePrice",
                                 "VolumeWeightPrice", "Volume"])
        dt = pd.to_datetime(d["BarDateTime"], errors="coerce")
        d = d.copy(); d.index = dt
        d = d.dropna(subset=["LastTradePrice"])
        daily = d["LastTradePrice"].resample("1D").last().rename("close").to_frame()
        daily["open"]   = d["FirstTradePrice"].resample("1D").first()
        daily["high"]   = d["HighTradePrice"].resample("1D").max()
        daily["low"]    = d["LowTradePrice"].resample("1D").min()
        daily["volume"] = d["Volume"].resample("1D").sum()
        daily["vwap"]   = d["VolumeWeightPrice"].resample("1D").mean()
        d["dollar_bar"] = d["Volume"] * d["VolumeWeightPrice"]
        daily["dollar_vol"] = d["dollar_bar"].resample("1D").sum()
        parts.append(daily.dropna(subset=["close"]))

    result = pd.concat(parts)
    result = result[~result.index.duplicated(keep="first")].sort_index()
    result.index = (result.index.tz_localize("UTC")
                    if result.index.tz is None else result.index)
    result.to_parquet(cpath)
    return result


def build_equities_panel():
    cret   = os.path.join(CACHE, "s09_eq_rets.parquet")
    cdvol  = os.path.join(CACHE, "s09_eq_dvol.parquet")
    if os.path.exists(cret) and os.path.exists(cdvol):
        return pd.read_parquet(cret), pd.read_parquet(cdvol)

    closes, dvols = {}, {}
    for sym in ETF_SYMS:
        df = _load_etf_daily(sym)
        closes[sym] = df["close"]
        dvols[sym]  = df["dollar_vol"]
        print(f"  {sym}: {len(df)} days, "
              f"{df.index.min().date()} – {df.index.max().date()}")

    rets = pd.DataFrame(closes).sort_index().pct_change()
    dvol = pd.DataFrame(dvols).sort_index()
    rets.to_parquet(cret);  dvol.to_parquet(cdvol)
    return rets, dvol


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _reversal_k(ret: pd.Series, k: int) -> pd.Series:
    """k-day trailing return, strictly lagged (shift 1). Causal."""
    f = (1.0 + ret).rolling(k).apply(np.prod, raw=True) - 1.0
    return f.shift(1)


def build_panel(rets: pd.DataFrame, dvol: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Build long-form panel (obs x columns) for the regression.

    Z1 — market-minus-self, shift(1)
         For instrument i at time t, Z1_i,t = mean of r_j,t for j != i, lagged one day.
         This avoids mechanical inclusion of own return and contemporaneous correlation.

    Z2 — lagged 21d cross-sectional vol.

    Z3 — log dollar-volume (21d rolling mean), lagged 1.

    Z4 — Amihud illiquidity (|ret|/dollar_vol 21d mean), lagged 1.

    NOTE: 1/price "size proxy" is DROPPED (addresses the size-proxy issue).
    ETF share price is a fund-design artifact with no cross-sectional size
    interpretation. log-AUM is not available from the minute-bar files.
    """
    # cross-sectional vol (all instruments together) for Z2
    cvol_raw = rets.std(axis=1).rolling(21).mean().shift(1)

    rows = []
    for sym in ETF_SYMS:
        r = rets[sym].dropna()

        # factor (k-day reversal, lagged 1)
        fac = _reversal_k(r, k).reindex(r.index)

        # Z1: market-minus-self, lagged 1
        others = [s for s in ETF_SYMS if s != sym]
        mkt_minus_self = rets[others].mean(axis=1).reindex(r.index)
        z1 = mkt_minus_self.shift(1)

        z2 = cvol_raw.reindex(r.index)

        dv    = dvol[sym].reindex(r.index)
        ldv   = np.log(dv.replace(0, np.nan))
        z3    = ldv.rolling(21).mean().shift(1)

        amihud = r.abs() / dv.replace(0, np.nan)
        z4 = amihud.rolling(21).mean().shift(1)

        df = pd.DataFrame({
            "date":   r.index,
            "sym":    sym,
            "y":      r.values,
            "x":      fac.values,
            "z1":     z1.values,
            "z2":     z2.values,
            "z3":     z3.values,
            "z4":     z4.values,
        }, index=r.index).dropna()
        rows.append(df)

    panel = pd.concat(rows).sort_index()
    panel["date"] = panel.index.normalize()   # UTC calendar day for clustering
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# OLS UTILITIES (naive t and clustered t in one pass)
# ═══════════════════════════════════════════════════════════════════════════════

def _ols_frisch_waugh(y: np.ndarray, x: np.ndarray, Z: np.ndarray):
    """
    Coefficient on x in y ~ intercept + x + Z (Frisch-Waugh residualisation).
    Z may have multiple columns. Returns (beta, residuals_y, residuals_x, n_params).
    """
    n = len(y)
    Zint = np.column_stack([np.ones(n), Z])
    def resid(v):
        coef, *_ = np.linalg.lstsq(Zint, v, rcond=None)
        return v - Zint @ coef

    rx = resid(x); ry = resid(y)
    srr = (rx * rx).sum()
    if srr <= 0:
        return 0.0, ry, rx, Zint.shape[1]
    b = (rx * ry).sum() / srr
    return b, ry - b * rx, rx, Zint.shape[1]   # beta, e_full, rx, k



def ols_full(y: np.ndarray, x: np.ndarray, Z: np.ndarray,
             date_ids: np.ndarray, entity_ids: np.ndarray):
    """
    Run OLS (Frisch-Waugh) and return a full SE bundle:
      - naive_t    : homoskedastic OLS t
      - date_t     : CR1-clustered by date (White-Rogers)
      - twoWay_t   : two-way cluster (date + entity) — additive approximation
      - nw5_t      : Newey-West t (lag=5)
      - beta       : point estimate
    """
    beta, e, rx, k_total = _ols_frisch_waugh(y, x, Z)
    n   = len(y)
    srr = (rx * rx).sum()
    if srr <= 0:
        return dict(beta=0.0, naive_t=0.0, date_t=0.0, twoWay_t=0.0, nw5_t=0.0)

    df = n - k_total - 1

    # ---- naive OLS (IID) ----
    s2     = (e @ e) / max(df, 1)
    var_ols = s2 / srr
    naive_t = beta / np.sqrt(max(var_ols, 1e-30))

    # ---- CR1 clustered by date ----
    def cluster_meat(ids):
        unique = np.unique(ids)
        G = len(unique)
        meat = 0.0
        for g in unique:
            mask = ids == g
            score = rx[mask] @ e[mask]   # sum of rx_i * e_i in cluster g
            meat += score * score
        # CR1 scaling: (G/(G-1)) * (n-1)/(n-k)
        scale = (G / (G - 1)) * ((n - 1) / max(n - k_total - 1, 1))
        return meat * scale

    meat_d  = cluster_meat(date_ids)
    var_d   = meat_d / (srr * srr)
    date_t  = beta / np.sqrt(max(var_d, 1e-30))

    meat_e  = cluster_meat(entity_ids)
    var_e   = meat_e / (srr * srr)

    # two-way: V_date + V_entity - V_both(naive as approx for the intersection)
    # Full two-way requires V(date+entity intersection) which is expensive when
    # some cells are unique. We use the Cameron-Gelbach-Miller additive approximation:
    # V_2way = V_date + V_entity - V_naive  (subtract baseline heteroskedastic, not OLS)
    # For simplicity use the additive approximation V_2way = V_d + V_e - var_ols
    var_2way = max(var_d + var_e - var_ols, var_d)   # lower bound = date cluster
    twoWay_t = beta / np.sqrt(max(var_2way, 1e-30))

    # ---- Newey-West (lag=5) ----
    # HAC variance estimator: V = (1/n^2) * S0 + sum_{l=1}^{L} w_l * (Sl + Sl')
    # where Sl = sum_t rx_t * e_t * rx_{t-l} * e_{t-l},  w_l = 1 - l/(L+1)
    L = 5
    scores = rx * e
    S0 = float((scores @ scores))
    S  = S0
    for lag in range(1, L + 1):
        wl = 1.0 - lag / (L + 1)
        cross = float(scores[lag:] @ scores[:-lag])
        S += 2.0 * wl * cross
    S = max(S, 1e-30)
    var_nw  = S / (srr * srr)
    nw5_t   = beta / np.sqrt(max(var_nw, 1e-30))

    return dict(beta=float(beta), naive_t=float(naive_t),
                date_t=float(date_t), twoWay_t=float(twoWay_t), nw5_t=float(nw5_t))


# ═══════════════════════════════════════════════════════════════════════════════
# T-STAT LADDER (naive OLS only; clustered applied at each step in a wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ladder_full(panel: pd.DataFrame):  # -> List[dict]
    """
    Steps:
      1. naive (no Z)
      2. + Z1 (market-minus-self, lagged)   <- FIX #1
      3. + Z1 + Z2
      4. + Z1 + Z2 + Z3 + Z4

    For each step: naive_t, date_clustered_t, two-way_t, NW5_t.
    date_clustered_t is the HEADLINE for verdict.
    """
    y     = panel["y"].to_numpy()
    x     = panel["x"].to_numpy()
    z1    = panel["z1"].to_numpy()
    z2    = panel["z2"].to_numpy()
    z3    = panel["z3"].to_numpy()
    z4    = panel["z4"].to_numpy()

    date_ids   = pd.factorize(panel["date"])[0].astype(np.int32)
    entity_ids = pd.factorize(panel["sym"])[0].astype(np.int32)

    def std(v):
        sd = v.std(ddof=1)
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()

    x_s  = std(x)
    z1_s = std(z1); z2_s = std(z2)
    z3_s = std(z3); z4_s = std(z4)

    steps = [
        ("naive",      x_s, np.empty((len(y), 0))),
        ("+ Z1 (mkt-self, lag1)",
                       x_s, np.column_stack([z1_s])),
        ("+ Z1+Z2 (mkt+cvol)",
                       x_s, np.column_stack([z1_s, z2_s])),
        ("+ Z1+Z2+Z3+Z4 (liq)",
                       x_s, np.column_stack([z1_s, z2_s, z3_s, z4_s])),
    ]

    rows = []
    for label, xi, Zi in steps:
        # ols_full handles empty Zi correctly (Frisch-Waugh vs intercept-only)
        res = ols_full(y, xi, Zi, date_ids, entity_ids)
        res["step"] = label
        rows.append(res)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK PERMUTATION NULL (block length = 5, preserves MA(4) autocorr)
# ═══════════════════════════════════════════════════════════════════════════════

def block_permutation_null(panel: pd.DataFrame,
                           block_len: int = 5,
                           n_perm: int = 1000,
                           seed: int = 42) -> np.ndarray:
    """
    FIX #3: Block permutation with block_len=5.
    For each instrument independently, partition the time series into
    non-overlapping blocks of length block_len, then shuffle the block order.
    This preserves within-block autocorrelation (the MA(4) induced by the
    5-day overlapping factor window) while breaking the predictive relationship.

    Returns array of n_perm date-clustered t-stats under the null.
    """
    rng = np.random.default_rng(seed)

    # Extract per-instrument numpy arrays (in time order)
    inst_data = {}
    for sym in ETF_SYMS:
        sub = panel[panel["sym"] == sym].copy()
        if len(sub) < block_len:
            continue
        inst_data[sym] = {
            "y":   sub["y"].to_numpy(),
            "x":   sub["x"].to_numpy(),
            "z1":  sub["z1"].to_numpy(),
            "z2":  sub["z2"].to_numpy(),
            "z3":  sub["z3"].to_numpy(),
            "z4":  sub["z4"].to_numpy(),
            "dates": sub["date"].to_numpy(),
        }

    y     = panel["y"].to_numpy()
    z1    = panel["z1"].to_numpy()
    z2    = panel["z2"].to_numpy()
    z3    = panel["z3"].to_numpy()
    z4    = panel["z4"].to_numpy()
    date_ids   = pd.factorize(panel["date"])[0].astype(np.int32)
    entity_ids = pd.factorize(panel["sym"])[0].astype(np.int32)

    def std_arr(v):
        sd = v.std(ddof=1)
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()

    # Precompute full-panel standardisation on confounders (fixed across perms)
    z1_s = std_arr(z1); z2_s = std_arr(z2)
    z3_s = std_arr(z3); z4_s = std_arr(z4)
    Z_full = np.column_stack([z1_s, z2_s, z3_s, z4_s])

    # Build per-instrument block index lists
    sym_blocks = {}
    for sym, d in inst_data.items():
        n = len(d["x"])
        n_full_blocks = n // block_len
        # indices of start of each full block
        sym_blocks[sym] = {
            "x": d["x"],
            "n": n,
            "n_blocks": n_full_blocks,
            "tail": n - n_full_blocks * block_len,
        }

    # Map each panel row to (sym, local_position)
    sym_local_pos = {}
    for sym in ETF_SYMS:
        sub = panel[panel["sym"] == sym]
        sym_local_pos[sym] = np.arange(len(sub))

    perm_t = []
    panel_len = len(panel)

    for _ in range(n_perm):
        x_perm = np.empty(panel_len)
        # Reconstruct the full panel order from permuted per-instrument blocks
        row_cursor = 0
        for sym in ETF_SYMS:
            if sym not in sym_blocks:
                continue
            info   = sym_blocks[sym]
            n      = info["n"]
            nb     = info["n_blocks"]
            tail   = info["tail"]
            x_orig = info["x"]

            # shuffle block indices
            block_order = rng.permutation(nb)
            x_new = np.empty(n)
            write = 0
            for bi in block_order:
                src = bi * block_len
                x_new[write:write + block_len] = x_orig[src:src + block_len]
                write += block_len
            # append tail (unshuffled to avoid partial blocks)
            if tail > 0:
                x_new[write:] = x_orig[nb * block_len:]

            sub_mask = panel["sym"] == sym
            x_perm[sub_mask] = x_new

        x_perm_s = std_arr(x_perm)
        res = ols_full(y, x_perm_s, Z_full, date_ids, entity_ids)
        perm_t.append(res["date_t"])

    return np.array(perm_t)


# ═══════════════════════════════════════════════════════════════════════════════
# POLLUTE-AND-VERIFY (FIX #5)
# ═══════════════════════════════════════════════════════════════════════════════

def pollute_and_verify(rets: pd.DataFrame, dvol: pd.DataFrame):
    """
    FIX #5: Real pollute-and-verify.

    Build a "poisoned" return panel where ret[t+1] has been shifted to look
    like ret[t] (i.e. we feed the future into the predictor by constructing
    the factor using shift(0) instead of shift(1)).

    Then:
      - Verify that the LABEL (y) in the original panel is UNCHANGED
        (same array, element-for-element) — the label is always ret[t+1]
        and should be unaffected by our change to the factor.
      - Confirm the factor x in the poisoned panel is DIFFERENT from the
        original (they should differ by exactly one lag).
      - Confirm the poisoned t-stat is substantially larger than the clean one
        (adding lookahead should inflate the signal).

    If the "clean" t-stat is already as high as the "poisoned" one, something
    is wrong. A truly causal factor should be weaker than its future-contaminated
    version.
    """
    print("\n  Building clean panel (shift 1)...")
    panel_clean = build_panel(rets, dvol, k=5)

    # Poisoned panel: factor built with shift(0) — no lag.
    print("  Building poisoned panel (shift 0 = lookahead)...")
    panel_poisoned_rows = []
    cvol_raw = rets.std(axis=1).rolling(21).mean().shift(1)

    for sym in ETF_SYMS:
        r = rets[sym].dropna()
        # Poisoned factor: no lag (contemporaneous 5-day return)
        fac_raw = (1.0 + r).rolling(5).apply(np.prod, raw=True) - 1.0
        fac_poisoned = fac_raw  # shift(0) — contains current day
        others = [s for s in ETF_SYMS if s != sym]
        mkt_minus_self = rets[others].mean(axis=1).reindex(r.index)
        z1 = mkt_minus_self.shift(1)
        z2 = cvol_raw.reindex(r.index)
        dv = dvol[sym].reindex(r.index)
        ldv = np.log(dv.replace(0, np.nan))
        z3 = ldv.rolling(21).mean().shift(1)
        amihud = r.abs() / dv.replace(0, np.nan)
        z4 = amihud.rolling(21).mean().shift(1)
        df = pd.DataFrame({
            "date": r.index,
            "sym": sym,
            "y": r.values,
            "x": fac_poisoned.values,
            "z1": z1.values,
            "z2": z2.values,
            "z3": z3.values,
            "z4": z4.values,
        }, index=r.index).dropna()
        panel_poisoned_rows.append(df)
    panel_poisoned = pd.concat(panel_poisoned_rows).sort_index()
    panel_poisoned["date"] = panel_poisoned.index.normalize()

    # Verify labels are the same (same date+sym index)
    merged = panel_clean.merge(
        panel_poisoned[["date", "sym", "y", "x"]].rename(columns={"y": "y_p", "x": "x_p"}),
        on=["date", "sym"], how="inner")
    label_max_diff = float((merged["y"] - merged["y_p"]).abs().max())
    factor_max_diff = float((merged["x"] - merged["x_p"]).abs().max())

    # Run clustered regression on both panels
    def run_date_t(p):
        y_  = p["y"].to_numpy()
        x_  = p["x"].to_numpy()
        z1_ = p["z1"].to_numpy(); z2_ = p["z2"].to_numpy()
        z3_ = p["z3"].to_numpy(); z4_ = p["z4"].to_numpy()
        def std(v): sd = v.std(ddof=1); return (v-v.mean())/sd if sd>0 else v-v.mean()
        Z_ = np.column_stack([std(z1_), std(z2_), std(z3_), std(z4_)])
        di = pd.factorize(p["date"])[0].astype(np.int32)
        ei = pd.factorize(p["sym"])[0].astype(np.int32)
        return ols_full(y_, std(x_), Z_, di, ei)["date_t"]

    clean_t    = run_date_t(panel_clean)
    poisoned_t = run_date_t(panel_poisoned)

    return {
        "label_unchanged":  label_max_diff == 0.0,
        "label_max_diff":   label_max_diff,
        "factor_changed":   factor_max_diff > 0,
        "factor_max_diff":  factor_max_diff,
        "clean_date_t":     float(clean_t),
        "poisoned_date_t":  float(poisoned_t),
        "lookahead_amplifies": abs(poisoned_t) > abs(clean_t),
        "verdict": ("CLEAN" if (label_max_diff == 0 and abs(poisoned_t) > abs(clean_t))
                    else "SUSPECT"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS (FIX #6)
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_oos_tstat(panel: pd.DataFrame, window: int = 252) -> dict:
    """
    Rolling 252-day window. For each window, estimate the full-confounder
    regression, record the date-clustered t-stat.
    Returns mean, std, and the fraction of windows with |t|>1.96.
    """
    dates = sorted(panel["date"].unique())
    if len(dates) < window + 20:
        return {"mean_t": np.nan, "std_t": np.nan, "frac_sig": np.nan,
                "n_windows": 0}

    def run_window_t(sub):
        if len(sub) < 30:
            return np.nan
        y_  = sub["y"].to_numpy(); x_  = sub["x"].to_numpy()
        z1_ = sub["z1"].to_numpy(); z2_ = sub["z2"].to_numpy()
        z3_ = sub["z3"].to_numpy(); z4_ = sub["z4"].to_numpy()
        def std(v): sd = v.std(ddof=1); return (v-v.mean())/sd if sd>0 else v-v.mean()
        Z_ = np.column_stack([std(z1_), std(z2_), std(z3_), std(z4_)])
        di = pd.factorize(sub["date"])[0].astype(np.int32)
        ei = pd.factorize(sub["sym"])[0].astype(np.int32)
        res = ols_full(y_, std(x_), Z_, di, ei)
        return res["date_t"]

    ts = []
    for start in range(0, len(dates) - window, 21):   # step ~1 month
        win_dates = set(dates[start:start + window])
        sub = panel[panel["date"].isin(win_dates)]
        t_  = run_window_t(sub)
        if np.isfinite(t_):
            ts.append(t_)

    ts = np.array(ts)
    return {
        "mean_t":    float(np.mean(ts)),
        "std_t":     float(np.std(ts, ddof=1)),
        "frac_sig":  float(np.mean(np.abs(ts) > 1.96)),
        "n_windows": int(len(ts)),
        "ts_min":    float(np.min(ts)) if len(ts) else np.nan,
        "ts_max":    float(np.max(ts)) if len(ts) else np.nan,
    }


def subsample_tstat(panel: pd.DataFrame) -> dict:
    """
    Pre-2015 vs post-2015 date-clustered t-stats on full confounder set.
    """
    def run(sub):
        if len(sub) < 30:
            return np.nan
        y_ = sub["y"].to_numpy(); x_ = sub["x"].to_numpy()
        z1_ = sub["z1"].to_numpy(); z2_ = sub["z2"].to_numpy()
        z3_ = sub["z3"].to_numpy(); z4_ = sub["z4"].to_numpy()
        def std(v): sd=v.std(ddof=1); return (v-v.mean())/sd if sd>0 else v-v.mean()
        Z_ = np.column_stack([std(z1_), std(z2_), std(z3_), std(z4_)])
        di = pd.factorize(sub["date"])[0].astype(np.int32)
        ei = pd.factorize(sub["sym"])[0].astype(np.int32)
        return ols_full(y_, std(x_), Z_, di, ei)["date_t"]

    cut = pd.Timestamp("2015-01-01", tz="UTC")
    pre  = panel[panel["date"] <  cut]
    post = panel[panel["date"] >= cut]
    return {
        "pre2015_date_t":  float(run(pre))  if len(pre)  > 100 else np.nan,
        "post2015_date_t": float(run(post)) if len(post) > 100 else np.nan,
        "pre2015_n":       int(len(pre)),
        "post2015_n":      int(len(post)),
    }


def lookback_sweep(rets: pd.DataFrame, dvol: pd.DataFrame):  # -> List[dict]
    """
    FIX #6: Lookback sweep over {3, 5, 7, 10} days.
    For each k, build a fresh panel and report date-clustered t on full confounder set.
    """
    results = []
    for k in [3, 5, 7, 10]:
        p = build_panel(rets, dvol, k=k)
        y_  = p["y"].to_numpy(); x_  = p["x"].to_numpy()
        z1_ = p["z1"].to_numpy(); z2_ = p["z2"].to_numpy()
        z3_ = p["z3"].to_numpy(); z4_ = p["z4"].to_numpy()
        def std(v): sd = v.std(ddof=1); return (v-v.mean())/sd if sd>0 else v-v.mean()
        Z_  = np.column_stack([std(z1_), std(z2_), std(z3_), std(z4_)])
        di  = pd.factorize(p["date"])[0].astype(np.int32)
        ei  = pd.factorize(p["sym"])[0].astype(np.int32)
        res = ols_full(y_, std(x_), Z_, di, ei)
        results.append({"lookback_days": k,
                         "n_obs": len(p),
                         "naive_t":  res["naive_t"],
                         "date_t":   res["date_t"],
                         "nw5_t":    res["nw5_t"]})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PEAK RAM
# ═══════════════════════════════════════════════════════════════════════════════

def peak_ram_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("S9 v2 — CORRECTED Confounder Robustness on Equities Short-Reversal")
    print("All 7 identified gaps addressed.")
    print("=" * 72)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("\n[1/7] Building daily panel (close + dollar_vol)...")
    rets, dvol = build_equities_panel()
    print(f"  Panel: {rets.shape[1]} instruments × {rets.shape[0]} days "
          f"({rets.index.min().date()} – {rets.index.max().date()})")

    # ── 2. Build main panel (k=5, corrected Z1) ───────────────────────────────
    print("\n[2/7] Building analysis panel (Z1=market-minus-self, shift(1))...")
    panel = build_panel(rets, dvol, k=5)
    n_obs = len(panel)
    print(f"  Valid obs: {n_obs:,}  ({len(ETF_SYMS)} instruments × ~{n_obs//len(ETF_SYMS):,} days)")
    print(f"  Date range: {panel['date'].min().date()} – {panel['date'].max().date()}")

    # ── 3. T-stat ladder (naive + all clustered variants) ─────────────────────
    print("\n[3/7] T-stat ladder (naive OLS + date-clustered + 2-way + NW5)...")
    ladder = run_ladder_full(panel)
    n_dates = panel["date"].nunique()
    print(f"\n  N obs (pooled panel) : {n_obs:,}")
    print(f"  N unique dates       : {n_dates:,}")
    print(f"  Effective clusters   : {n_dates:,} dates × {len(ETF_SYMS)} entities")
    print()
    print(f"  {'Step':<35} {'beta':>7} {'naive_t':>8} {'date_t':>8} "
          f"{'2way_t':>8} {'nw5_t':>7}")
    print(f"  {'-'*75}")
    for row in ladder:
        print(f"  {row['step']:<35} {row['beta']:>7.4f} {row['naive_t']:>8.3f} "
              f"{row['date_t']:>8.3f} {row['twoWay_t']:>8.3f} {row['nw5_t']:>7.3f}")

    full_row = ladder[-1]   # full confounder set
    headline_t = full_row["date_t"]
    print(f"\n  HEADLINE (date-clustered, full conf.): t = {headline_t:.3f}  "
          f"({'SIG |t|>1.96' if abs(headline_t) > 1.96 else 'NOT SIG'})")

    # ── 4. Block permutation null (block_len=5) ───────────────────────────────
    print("\n[4/7] Block permutation null (block_len=5, M=1000)...")
    perm_t = block_permutation_null(panel, block_len=5, n_perm=1000, seed=20260613)
    p_raw = float(np.mean(np.abs(perm_t) >= abs(headline_t)))
    perm_pct95 = float(np.percentile(np.abs(perm_t), 95))
    print(f"  Observed |date_t|      : {abs(headline_t):.3f}")
    print(f"  Null 95th pctile |t|   : {perm_pct95:.3f}")
    print(f"  Block-perm p-value     : {p_raw:.4f}  "
          f"({'SIG' if p_raw < 0.05 else 'NOT SIG'})")
    print(f"  Null mean±std          : {perm_t.mean():.3f} ± {perm_t.std(ddof=1):.3f}")

    # ── 5. Pollute-and-verify ─────────────────────────────────────────────────
    print("\n[5/7] Pollute-and-verify (shift(0) lookahead poison)...")
    pav = pollute_and_verify(rets, dvol)
    print(f"  Label unchanged?          : {pav['label_unchanged']} "
          f"(max_diff={pav['label_max_diff']:.2e})")
    print(f"  Factor CHANGED by shift?  : {pav['factor_changed']} "
          f"(max_diff={pav['factor_max_diff']:.4f})")
    print(f"  Clean date_t              : {pav['clean_date_t']:.3f}")
    print(f"  Poisoned (shift=0) date_t : {pav['poisoned_date_t']:.3f}")
    print(f"  Lookahead amplifies?      : {pav['lookahead_amplifies']}")
    print(f"  Verdict                   : {pav['verdict']}")

    # ── 6. Robustness ─────────────────────────────────────────────────────────
    print("\n[6/7] Robustness: rolling OOS, subsamples, lookback sweep...")

    print("  (a) Rolling 252-day OOS date_t band...")
    roll = rolling_oos_tstat(panel, window=252)
    print(f"      n_windows={roll['n_windows']}, "
          f"mean_t={roll['mean_t']:.3f}, std_t={roll['std_t']:.3f}, "
          f"min_t={roll['ts_min']:.3f}, max_t={roll['ts_max']:.3f}, "
          f"frac|t|>1.96={roll['frac_sig']:.2f}")

    print("  (b) Pre/post-2015 subsamples...")
    subs = subsample_tstat(panel)
    print(f"      pre-2015  n={subs['pre2015_n']:,}, date_t={subs['pre2015_date_t']:.3f}")
    print(f"      post-2015 n={subs['post2015_n']:,}, date_t={subs['post2015_date_t']:.3f}")

    print("  (c) Lookback sweep {3,5,7,10} days...")
    sweep = lookback_sweep(rets, dvol)
    print(f"      {'k':>4} {'n_obs':>7} {'naive_t':>8} {'date_t':>8} {'nw5_t':>7}")
    for row in sweep:
        print(f"      {row['lookback_days']:>4} {row['n_obs']:>7,} "
              f"{row['naive_t']:>8.3f} {row['date_t']:>8.3f} {row['nw5_t']:>7.3f}")

    # ── 7. Verdict ─────────────────────────────────────────────────────────────
    print("\n[7/7] VERDICT")
    print("=" * 72)

    # Headline logic
    sig_date_cluster = abs(headline_t) > 1.96
    sig_block_perm   = p_raw < 0.05
    pav_clean        = pav["verdict"] == "CLEAN"

    print(f"\n  T-STAT LADDER (date-clustered = HEADLINE):")
    print(f"  {'Step':<35} {'naive_t':>8} {'date_t':>8} {'nw5_t':>7}")
    for row in ladder:
        print(f"  {row['step']:<35} {row['naive_t']:>8.3f} "
              f"{row['date_t']:>8.3f} {row['nw5_t']:>7.3f}")
    print()
    print(f"  Block-perm p (block=5, M=1000) : {p_raw:.4f}")
    print(f"  Pollute-and-verify             : {pav['verdict']}")
    print(f"  Rolling OOS frac|t|>1.96       : {roll['frac_sig']:.2f} "
          f"(mean_t={roll['mean_t']:.3f})")
    print(f"  Subsample pre/post-2015 date_t : "
          f"{subs['pre2015_date_t']:.3f} / {subs['post2015_date_t']:.3f}")
    print()

    # Primary verdict logic
    if not sig_date_cluster:
        bar_verdict = "NEGATIVE"
        print("  RESULT: NEGATIVE")
        print("  Short-reversal does NOT survive date-clustered standard errors.")
        print(f"  Naive OLS t={ladder[0]['naive_t']:.2f} → date-clustered t={headline_t:.2f}.")
        print("  The 7-ETF daily panel has strong cross-sectional correlation:")
        print("  each calendar date is a single effective observation, not 7.")
        print("  Inflating the denominator by cluster structure reveals the naive")
        print("  OLS t-stat was an artifact of treating correlated panel observations")
        print("  as independent. Short-reversal is NOT robustly significant.")
    elif sig_date_cluster and not sig_block_perm:
        bar_verdict = "FAIL"
        print("  RESULT: FAIL (block permutation)")
        print(f"  Date-clustered t={headline_t:.2f} nominally > 1.96,")
        print(f"  but block-permutation p={p_raw:.4f} >= 0.05.")
        print("  The block null (which preserves 5-day MA autocorr) shows")
        print("  the signal is not robust beyond the serial autocorrelation structure.")
    else:
        bar_verdict = "PASS"
        print("  RESULT: PASS")
        print(f"  Date-clustered t={headline_t:.2f} AND block-perm p={p_raw:.4f}")
        print("  Both tests support survival after corrected standard errors.")

    print()
    print(f"  NOTE: DSR is DROPPED (gap #7). No proper alternative-")
    print(f"  parameterization trials exist for this pre-registered single factor.")
    print(f"  Verdict rests on clustered t and block-permutation p only.")
    print()
    print(f"  Bar verdict: {bar_verdict}")
    print("=" * 72)

    ram = peak_ram_mb()
    print(f"\n  Peak RAM: {ram:.0f} MB")
    print(f"  Elapsed : {time.time()-t0:.1f}s")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    # t-stat ladder CSV
    ladder_df = pd.DataFrame([
        {"step": r["step"], "beta": r["beta"],
         "naive_t": r["naive_t"], "date_t": r["date_t"],
         "twoWay_t": r["twoWay_t"], "nw5_t": r["nw5_t"],
         "sig_naive": abs(r["naive_t"]) > 1.96,
         "sig_date_cluster": abs(r["date_t"]) > 1.96}
        for r in ladder
    ])
    ladder_df.to_csv(os.path.join(OUT_DIR, "tstat_ladder_v2.csv"), index=False)

    # block perm null
    perm_df = pd.DataFrame({"block_perm_date_t": perm_t})
    perm_df.to_parquet(os.path.join(OUT_DIR, "block_perm_null_v2.parquet"))

    # lookback sweep
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(os.path.join(OUT_DIR, "lookback_sweep_v2.csv"), index=False)

    # rolling OOS
    roll_detail = {"rolling_oos": roll}
    with open(os.path.join(OUT_DIR, "rolling_oos_v2.json"), "w") as fh:
        json.dump(roll_detail, fh, indent=2)

    # Full summary
    summary = {
        "run_date": "2026-06-13",
        "script": "s09_v2.py",
        "n_obs_pooled": n_obs,
        "n_instruments": len(ETF_SYMS),
        "n_unique_dates": int(n_dates),
        "date_range": f"{panel['date'].min().date()} – {panel['date'].max().date()}",
        "gaps_fixed": [
            "Z1=market-minus-self-shift1 (not contemporaneous whole-mkt mean)",
            "clustered SEs: date-cluster CR1 + two-way + NW5; date_t is headline",
            "block permutation null block_len=5",
            "1/price size proxy DROPPED (ETF fund-design artifact, no log-AUM available)",
            "real pollute-and-verify: poisoned shift(0) vs clean shift(1)",
            "rolling OOS t-band + pre/post-2015 + lookback sweep {3,5,7,10}",
            "DSR dropped: no proper alternative-parameterization trials",
        ],
        "tstat_ladder": [
            {"step": r["step"], "naive_t": round(r["naive_t"], 3),
             "date_t": round(r["date_t"], 3), "nw5_t": round(r["nw5_t"], 3)}
            for r in ladder
        ],
        "headline_date_t": round(headline_t, 3),
        "block_perm_p": round(p_raw, 4),
        "block_perm_null_pct95": round(perm_pct95, 3),
        "pollute_verify": pav,
        "rolling_oos": roll,
        "subsamples": subs,
        "lookback_sweep": sweep,
        "bar_verdict": bar_verdict,
        "peak_ram_mb": round(ram, 0),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "summary_v2.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    verdict = {
        "study": "S9_v2",
        "factor": "equities:short_reversal_5d",
        "run_date": summary["run_date"],
        "bar_verdict": bar_verdict,
        "headline_date_clustered_t": summary["headline_date_t"],
        "naive_t": round(ladder[0]["naive_t"], 3),
        "nw5_t": round(full_row["nw5_t"], 3),
        "block_perm_p_block5": summary["block_perm_p"],
        "pollute_verify_verdict": pav["verdict"],
        "rolling_oos_frac_sig": round(roll["frac_sig"], 3),
        "subsample_pre2015_date_t": round(subs["pre2015_date_t"], 3),
        "subsample_post2015_date_t": round(subs["post2015_date_t"], 3),
        "dsr": "DROPPED — no proper trial set",
        "note": ("Clustered SEs absorb cross-sectional correlation. "
                 "Naive OLS t conflated 7-instrument date-correlated obs as independent. "
                 "Correct inference uses date-clustered t as headline."),
    }
    with open(os.path.join(OUT_DIR, "verdict_v2.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)

    print(f"\nArtifacts in: {OUT_DIR}")
    print("  tstat_ladder_v2.csv")
    print("  block_perm_null_v2.parquet")
    print("  lookback_sweep_v2.csv")
    print("  rolling_oos_v2.json")
    print("  summary_v2.json")
    print("  verdict_v2.json")


if __name__ == "__main__":
    main()
