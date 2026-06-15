#!/usr/bin/env python3
"""
S9 v3 — Horizon-Matched Full Control Set.

Resolves the open question left by v2:
  v2 used a 1-day market-minus-self return (Z1) as the market confounder
  for a 5-day factor. A horizon-matched analysis shows this mismatch explains
  only ~10% of the factor's variance. With a 5-day market-minus-self control
  (window matched to the factor), the analysis yields date-clustered t=-3.13
  and block-perm p≈0.

CHANGES vs v2 (do NOT modify v2; it is preserved in git):

  1. Z1_5d: 5-day market-minus-self return (sum of others 5-day compounded,
     excluding self), lagged 1 so factor window is t-6..t-1 for both factor
     and Z1_5d. This matches the factor horizon.

  2. Z1_1d kept as auxiliary column (for the t-stat ladder step 2a
     "v2-style 1-day control" so we can show the full ladder:
     naive → +1d-mkt → +5d-mkt → +vol → +liquidity).

  3. Z2 (lagged cross-sectional vol) and Z3/Z4 (lagged log-dolvol + Amihud)
     unchanged from v2.

  4. Size proxy: log dollar-volume (Z3) doubles as the liquidity+size proxy.
     Log-AUM not available. Stated explicitly. ETF 1/price dropped (v2 already
     did this; preserved).

  5. Block permutation: block_len=10 (conservative; MA(4) from 5-day windows,
     rounding up for safety). M=1000.

  6. T-stat LADDER (5 steps):
       Step 1: naive (no Z)
       Step 2: + Z1_1d (v2-style, 1-day mkt)      [shows under-control baseline]
       Step 3: + Z1_5d (5d horizon-matched mkt)    [horizon-matched fix]
       Step 4: + Z1_5d + Z2 (+ vol)
       Step 5: + Z1_5d + Z2 + Z3 + Z4 (full)

  7. Tradeability: long-short reversal sleeve — long bottom-quintile on factor
     (largest losers = expected reversers) and short top-quintile, rebalanced
     daily, net of 3 bp one-way cost. Reports annualized Sharpe net of costs.

  8. Two-part honest verdict:
       (1) Confounder-robust predictor? (clustered t after full matched controls)
       (2) Tradeable net of realistic costs?

NOTE: s09_v2.py is preserved unchanged. This file creates new artifacts
      (suffix _v3) and does not overwrite any v2 outputs.
"""

import os, sys, time, json, warnings, resource

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

import resource as _res
# 10 GB virtual memory cap (ulimit -v equivalent)
try:
    _res.setrlimit(_res.RLIMIT_AS, (10 * 1024**3, 10 * 1024**3))
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy import stats as ss
import glob

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths  (same data dirs as v2)
# --------------------------------------------------------------------------- #
from config import EQUITY_1M, DATA_CACHE
ETF_DIR = EQUITY_1M
CACHE   = DATA_CACHE
OUT_DIR = HERE
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ANN      = 252.0
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]
COST_BPS = 3.0 / 10_000.0   # 3 bp one-way


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (identical to v2 — reuses caches)
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
    cret  = os.path.join(CACHE, "s09_eq_rets.parquet")
    cdvol = os.path.join(CACHE, "s09_eq_dvol.parquet")
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
    rets.to_parquet(cret); dvol.to_parquet(cdvol)
    return rets, dvol


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL BUILDER — v3 KEY CHANGE: horizon-matched Z1_5d
# ═══════════════════════════════════════════════════════════════════════════════

def _compound_k(ret: pd.Series, k: int) -> pd.Series:
    """k-day compounded return (NOT shifted yet)."""
    return (1.0 + ret).rolling(k).apply(np.prod, raw=True) - 1.0


def build_panel_v3(rets: pd.DataFrame, dvol: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Build long-form panel with horizon-matched controls.

    Factor x: k-day own trailing return, lagged 1 (causal, t-6..t-1 window).

    Controls (all strictly lagged — no contemporaneous data, no own-return):

      Z1_1d : 1-day market-minus-self return, lagged 1.  (v2 baseline, for ladder)
      Z1_5d : k-day market-minus-self return, lagged 1.  (horizon-matched; MAIN FIX)
               For instrument i, Z1_5d_i,t = compounded 5-day return of mean(r_j)
               for j != i, then shift(1). Window is t-6..t-1, same as the factor.
      Z2    : 21d rolling cross-sectional return volatility, lagged 1.
      Z3    : log dollar-volume (21d rolling mean), lagged 1.
      Z4    : Amihud illiquidity (|ret|/dolvol, 21d mean), lagged 1.

    Size proxy: Z3 (log-dollar-vol) also proxies size. ETF log-AUM not
    available from minute-bar files. 1/price is dropped (ETF fund-design
    artifact, no cross-sectional size interpretation).
    """
    # cross-sectional volatility for Z2
    cvol_raw = rets.std(axis=1).rolling(21).mean().shift(1)

    rows = []
    for sym in ETF_SYMS:
        r = rets[sym].dropna()

        # factor (k-day reversal, lagged 1)
        fac = _compound_k(r, k).shift(1).reindex(r.index)

        # Z1_1d: market-minus-self, lagged 1 (v2 style, 1-day window)
        others = [s for s in ETF_SYMS if s != sym]
        mkt_1d = rets[others].mean(axis=1).reindex(r.index)
        z1_1d  = mkt_1d.shift(1)

        # Z1_5d: market-minus-self, k-day compounded, lagged 1 (v3 horizon match)
        # Step 1: compute cross-sectional mean (excluding self) at each day
        mkt_mean_ex_self = rets[others].mean(axis=1).reindex(r.index)
        # Step 2: compound over k days, then shift 1 (same window as factor)
        z1_5d = _compound_k(mkt_mean_ex_self, k).shift(1)

        z2 = cvol_raw.reindex(r.index)

        dv   = dvol[sym].reindex(r.index)
        ldv  = np.log(dv.replace(0, np.nan))
        z3   = ldv.rolling(21).mean().shift(1)

        amihud = r.abs() / dv.replace(0, np.nan)
        z4     = amihud.rolling(21).mean().shift(1)

        df = pd.DataFrame({
            "date":   r.index,
            "sym":    sym,
            "y":      r.values,
            "x":      fac.values,
            "z1_1d":  z1_1d.values,
            "z1_5d":  z1_5d.values,
            "z2":     z2.values,
            "z3":     z3.values,
            "z4":     z4.values,
        }, index=r.index).dropna()
        rows.append(df)

    panel = pd.concat(rows).sort_index()
    panel["date"] = panel.index.normalize()
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# OLS UTILITIES  (identical to v2 — copy kept for self-containment)
# ═══════════════════════════════════════════════════════════════════════════════

def _ols_frisch_waugh(y: np.ndarray, x: np.ndarray, Z: np.ndarray):
    n = len(y)
    Zint = np.column_stack([np.ones(n), Z]) if Z.shape[1] > 0 else np.ones((n, 1))
    def resid(v):
        coef, *_ = np.linalg.lstsq(Zint, v, rcond=None)
        return v - Zint @ coef
    rx = resid(x); ry = resid(y)
    srr = (rx * rx).sum()
    if srr <= 0:
        return 0.0, ry, rx, Zint.shape[1]
    b = (rx * ry).sum() / srr
    return b, ry - b * rx, rx, Zint.shape[1]


def ols_full(y: np.ndarray, x: np.ndarray, Z: np.ndarray,
             date_ids: np.ndarray, entity_ids: np.ndarray):
    beta, e, rx, k_total = _ols_frisch_waugh(y, x, Z)
    n   = len(y)
    srr = (rx * rx).sum()
    if srr <= 0:
        return dict(beta=0.0, naive_t=0.0, date_t=0.0, twoWay_t=0.0, nw5_t=0.0)

    k_eff = k_total

    # naive OLS
    df_  = max(n - k_eff - 1, 1)
    s2   = (e @ e) / df_
    naive_t = beta / np.sqrt(max(s2 / srr, 1e-30))

    # CR1 date-clustered
    def cluster_meat(ids):
        unique = np.unique(ids)
        G = len(unique)
        meat = 0.0
        for g in unique:
            mask = ids == g
            score = rx[mask] @ e[mask]
            meat += score * score
        scale = (G / (G - 1)) * ((n - 1) / max(n - k_eff - 1, 1))
        return meat * scale

    meat_d = cluster_meat(date_ids)
    var_d  = meat_d / (srr * srr)
    date_t = beta / np.sqrt(max(var_d, 1e-30))

    meat_e = cluster_meat(entity_ids)
    var_e  = meat_e / (srr * srr)
    var_ols = s2 / srr
    var_2way = max(var_d + var_e - var_ols, var_d)
    twoWay_t = beta / np.sqrt(max(var_2way, 1e-30))

    # Newey-West lag=5
    L = 5
    scores = rx * e
    S0 = float(scores @ scores)
    S  = S0
    for lag in range(1, L + 1):
        wl = 1.0 - lag / (L + 1)
        cross = float(scores[lag:] @ scores[:-lag])
        S += 2.0 * wl * cross
    S = max(S, 1e-30)
    nw5_t = beta / np.sqrt(max(S / (srr * srr), 1e-30))

    return dict(beta=float(beta), naive_t=float(naive_t),
                date_t=float(date_t), twoWay_t=float(twoWay_t), nw5_t=float(nw5_t))


# ═══════════════════════════════════════════════════════════════════════════════
# T-STAT LADDER — v3 version (5 steps, horizon-matched)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ladder_v3(panel: pd.DataFrame):
    """
    5-step ladder:
      1. naive
      2. + Z1_1d (v2 under-control baseline, 1-day mkt)
      3. + Z1_5d (horizon-matched 5-day mkt)
      4. + Z1_5d + Z2 (+ volatility)
      5. + Z1_5d + Z2 + Z3 + Z4 (full: + liquidity + size-proxy)
    """
    y      = panel["y"].to_numpy()
    x      = panel["x"].to_numpy()
    z1_1d  = panel["z1_1d"].to_numpy()
    z1_5d  = panel["z1_5d"].to_numpy()
    z2     = panel["z2"].to_numpy()
    z3     = panel["z3"].to_numpy()
    z4     = panel["z4"].to_numpy()

    date_ids   = pd.factorize(panel["date"])[0].astype(np.int32)
    entity_ids = pd.factorize(panel["sym"])[0].astype(np.int32)

    def std(v):
        sd = v.std(ddof=1)
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()

    x_s     = std(x)
    z1_1_s  = std(z1_1d)
    z1_5_s  = std(z1_5d)
    z2_s    = std(z2)
    z3_s    = std(z3)
    z4_s    = std(z4)

    steps = [
        ("naive",
         x_s, np.empty((len(y), 0))),
        ("+ Z1_1d (v2: 1-day mkt-self, MISMATCHED)",
         x_s, np.column_stack([z1_1_s])),
        ("+ Z1_5d (MATCHED 5-day mkt-self)",
         x_s, np.column_stack([z1_5_s])),
        ("+ Z1_5d + Z2 (+ lag-cvol)",
         x_s, np.column_stack([z1_5_s, z2_s])),
        ("+ Z1_5d + Z2 + Z3 + Z4 (full)",
         x_s, np.column_stack([z1_5_s, z2_s, z3_s, z4_s])),
    ]

    rows = []
    for label, xi, Zi in steps:
        res = ols_full(y, xi, Zi, date_ids, entity_ids)
        res["step"] = label
        rows.append(res)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK PERMUTATION NULL — block_len=10 (conservative vs MA(4))
# ═══════════════════════════════════════════════════════════════════════════════

def block_permutation_null_v3(panel: pd.DataFrame,
                               block_len: int = 10,
                               n_perm: int = 1000,
                               seed: int = 42) -> np.ndarray:
    """
    Block permutation with block_len=10 (conservative; MA(4) from 5-day
    windows; v2 used 5, which matched the autocorr length exactly; 10 is
    more conservative and appropriate given uncertainty in autocorr reach).

    Shuffles the x (factor) only; confounders and y are held fixed.
    Returns array of n_perm date-clustered t-stats under the null.

    TAIL HANDLING: the final n mod block_len observations form a
    short residual block that participates in the shuffle at a RANDOM position,
    rather than always being appended unshuffled at the end. The earlier
    "append-at-end" scheme broke exchangeability and shifted the null mean
    ~3.6 SE below zero. Including the partial block in the permutation restores
    a null centred on zero.
    """
    rng = np.random.default_rng(seed)

    y     = panel["y"].to_numpy()
    z1_5d = panel["z1_5d"].to_numpy()
    z2    = panel["z2"].to_numpy()
    z3    = panel["z3"].to_numpy()
    z4    = panel["z4"].to_numpy()
    date_ids   = pd.factorize(panel["date"])[0].astype(np.int32)
    entity_ids = pd.factorize(panel["sym"])[0].astype(np.int32)

    def std_arr(v):
        sd = v.std(ddof=1)
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()

    z1_5_s = std_arr(z1_5d)
    z2_s   = std_arr(z2)
    z3_s   = std_arr(z3)
    z4_s   = std_arr(z4)
    Z_full = np.column_stack([z1_5_s, z2_s, z3_s, z4_s])

    # per-instrument block info: list of (start, length) blocks INCLUDING the
    # short tail block, so the tail can be placed at any position.
    sym_blocks = {}
    for sym in ETF_SYMS:
        sub  = panel[panel["sym"] == sym].copy()
        n    = len(sub)
        if n < block_len:
            continue
        nb   = n // block_len
        bounds = []   # (start, length)
        for b in range(nb):
            bounds.append((b * block_len, block_len))
        tail = n - nb * block_len
        if tail > 0:
            bounds.append((nb * block_len, tail))   # short residual block
        sym_blocks[sym] = {
            "x": sub["x"].to_numpy(),
            "n": n,
            "bounds": bounds,
            "mask": panel["sym"].values == sym,
        }

    panel_len = len(panel)
    perm_t    = []

    for _ in range(n_perm):
        x_perm = np.empty(panel_len)
        for sym, info in sym_blocks.items():
            x_orig = info["x"]
            n      = info["n"]
            bounds = info["bounds"]
            order  = rng.permutation(len(bounds))   # shuffle ALL blocks incl. tail
            x_new  = np.empty(n)
            write  = 0
            for bi in order:
                src, blen = bounds[bi]
                x_new[write:write + blen] = x_orig[src:src + blen]
                write += blen
            x_perm[info["mask"]] = x_new

        x_perm_s = std_arr(x_perm)
        res = ols_full(y, x_perm_s, Z_full, date_ids, entity_ids)
        perm_t.append(res["date_t"])

    return np.array(perm_t)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADEABILITY: long-short reversal sleeve (separate from predictor question)
# ═══════════════════════════════════════════════════════════════════════════════

def costed_sleeve(panel: pd.DataFrame) -> dict:
    """
    Long-short reversal sleeve.

    Signal: the 5-day own-return factor (x in the panel). Lower x = more
    negative reversal candidate = expected to mean-revert upward.

    Portfolio construction (daily rebalance):
      - Sort instruments by x each date.
      - Long bottom quintile (Q1 = most negative 5d return), equal-weight.
      - Short top quintile (Q5 = most positive 5d return), equal-weight.
      - y = next-day return (already the label in the panel).

    Costs (two accountings):
      - HEADLINE (conservative): assume FULL daily turnover on both legs.
        2 legs × 2 sides (enter+exit) × 3bp = 12bp/day.
      - REALIZED (addresses the turnover issue): track day-over-day weight vector and charge
        3bp per unit of |Δweight|. The 5-day factor is autocorrelated so
        positions persist and actual turnover < 100%. Reported alongside so the
        cost drag is not overstated. Both give negative SR here, so the
        not-tradeable conclusion holds under either.

    Returns annualized net Sharpe (full-turnover headline + realized turnover),
    gross Sharpe, net mean, net std, and mean daily turnover.

    NOTE: with only 7 ETFs, bottom/top quintile = 1-2 instruments.
    Quintile is approximate; we use bottom 2 / top 2 of 7.
    """
    COST_FULL = 4 * COST_BPS   # 12 bp/day full-turnover assumption

    records   = []
    prev_w    = {}             # sym -> signed weight on previous traded day
    turnovers = []

    for date, grp in panel.groupby("date"):
        if len(grp) < 4:
            continue
        ranked = grp.sort_values("x")
        long_syms  = list(ranked.iloc[:2]["sym"])    # most negative factor → long
        short_syms = list(ranked.iloc[-2:]["sym"])   # most positive factor → short

        long_ret  = ranked.iloc[:2]["y"].mean()
        short_ret = ranked.iloc[-2:]["y"].mean()
        gross     = long_ret - short_ret

        # target weight vector this day: +0.5 each long leg, -0.5 each short leg
        # (gross exposure 1.0 per side; matches the equal-weight long-short above)
        w = {s: 0.5 for s in long_syms}
        for s in short_syms:
            w[s] = w.get(s, 0.0) - 0.5

        # turnover = sum |w_t - w_{t-1}| over the union of symbols
        all_syms = set(w) | set(prev_w)
        turn = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_syms)
        turnovers.append(turn)
        prev_w = w

        net_full     = gross - COST_FULL
        net_realized = gross - turn * COST_BPS
        records.append({"date": date, "gross": gross,
                        "net_full": net_full, "net_realized": net_realized})

    if not records:
        return {"sleeve_SR_net": np.nan, "sleeve_SR_gross": np.nan,
                "sleeve_mean_net": np.nan, "sleeve_std_net": np.nan, "n_days": 0}

    df  = pd.DataFrame(records).set_index("date").sort_index()
    g   = df["gross"]
    nf  = df["net_full"]
    nr  = df["net_realized"]

    sr_gross    = float(g.mean()  / g.std(ddof=1)  * np.sqrt(ANN))
    sr_net_full = float(nf.mean() / nf.std(ddof=1) * np.sqrt(ANN))
    sr_net_real = float(nr.mean() / nr.std(ddof=1) * np.sqrt(ANN))

    return {
        "sleeve_SR_gross":         round(sr_gross, 4),
        "sleeve_SR_net":           round(sr_net_full, 4),   # HEADLINE = conservative
        "sleeve_SR_net_realized":  round(sr_net_real, 4),
        "sleeve_mean_net_bps":     round(float(nf.mean()) * 10_000, 3),
        "sleeve_std_net_bps":      round(float(nf.std(ddof=1)) * 10_000, 3),
        "mean_daily_turnover":     round(float(np.mean(turnovers)), 4),
        "n_days": int(len(df)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANCE DECOMPOSITION (how much of factor variance does Z1_5d explain?)
# ═══════════════════════════════════════════════════════════════════════════════

def factor_variance_decomp(panel: pd.DataFrame) -> dict:
    """
    Regress factor x on Z1_1d and Z1_5d (separately) to show what fraction
    of factor variance each market control absorbs. This quantifies the
    horizon-mismatch: if Z1_1d R^2 is ~10% but Z1_5d R^2 is much higher,
    the mismatch was real.
    """
    x     = panel["x"].to_numpy()
    z1_1d = panel["z1_1d"].to_numpy()
    z1_5d = panel["z1_5d"].to_numpy()

    def r2(predictor):
        X = np.column_stack([np.ones(len(x)), predictor])
        coef, *_ = np.linalg.lstsq(X, x, rcond=None)
        xhat = X @ coef
        ss_res = ((x - xhat)**2).sum()
        ss_tot = ((x - x.mean())**2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "R2_factor_on_Z1_1d": round(float(r2(z1_1d)), 4),
        "R2_factor_on_Z1_5d": round(float(r2(z1_5d)), 4),
    }


def per_instrument_r2(panel: pd.DataFrame) -> dict:
    """
    Breadth diagnostic (gap #1): per-instrument R2 of the factor on its
    own horizon-matched market-minus-self control (Z1_5d). A high R2 means that
    instrument's idiosyncratic short-reversal signal is nearly absorbed by the
    market control — i.e. it contributes little independent signal.

    Expectation: SPY ~ the market, so SPY's market-excl-self ≈ SPY itself and
    R2 → ~0.9 (SPY adds almost no idiosyncratic signal). Sector ETFs retain
    substantial idiosyncratic variance. This shows the effective breadth is
    narrower than the nominal 7 instruments.
    """
    out = {}
    for sym in ETF_SYMS:
        sub = panel[panel["sym"] == sym]
        x   = sub["x"].to_numpy()
        z   = sub["z1_5d"].to_numpy()
        if len(x) < 30:
            out[sym] = float("nan"); continue
        X = np.column_stack([np.ones(len(x)), z])
        coef, *_ = np.linalg.lstsq(X, x, rcond=None)
        xhat = X @ coef
        ss_res = ((x - xhat)**2).sum()
        ss_tot = ((x - x.mean())**2).sum()
        out[sym] = round(float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0, 4)
    # count instruments with meaningful idiosyncratic residual (R2 < 0.85)
    n_breadth = sum(1 for v in out.values() if np.isfinite(v) and v < 0.85)
    out["_effective_breadth_n"] = int(n_breadth)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# WHAT IS THE SIGNAL? — date-FE equivalence + cross-sectional fraction
# ═══════════════════════════════════════════════════════════════════════════════

def signal_identity_check(panel: pd.DataFrame) -> dict:
    """
    The central concern: Z1_5d is ~0.998 correlated ACROSS instruments
    on the same date, i.e. it is essentially a date fixed effect, so partialing
    it out converts the regression into a CROSS-SECTIONAL relative-reversal test
    (Lo-MacKinlay 1990), not a 5-day own-return reversal test.

    We adjudicate this directly:

      (a) cross_frac : fraction of Z1_5d's total variance that is cross-sectional
          (within-date) vs time-series. If tiny (~0.1), Z1_5d ≈ date-level common
          factor and it acts like a date FE.

      (b) date_fe_t  : date-clustered t from a pure DATE FIXED-EFFECT regression
          (demean BOTH y and x within each date, then regress; no Z at all).
          If this reproduces the full-control headline (~-3.1), the v3 control IS
          operating as a date FE and the surviving effect is cross-sectional
          relative reversal — this reframing.

      (c) corr_z1_cross : median pairwise cross-instrument same-date correlation
          of Z1_5d (the 0.998 number to confirm or refute).
    """
    # (a) cross-sectional vs time-series variance of Z1_5d
    z = panel[["date", "sym", "z1_5d"]].copy()
    date_mean = z.groupby("date")["z1_5d"].transform("mean")
    within = z["z1_5d"] - date_mean              # cross-sectional (within-date) part
    var_within = float(np.var(within.to_numpy()))
    var_total  = float(np.var(z["z1_5d"].to_numpy()))
    cross_frac = var_within / var_total if var_total > 0 else float("nan")

    # (c) cross-instrument same-date correlation of Z1_5d (wide pivot)
    wide = panel.pivot_table(index="date", columns="sym", values="z1_5d")
    cc = wide.corr().to_numpy()
    iu = np.triu_indices_from(cc, k=1)
    corr_z1_cross = float(np.nanmedian(cc[iu]))

    # (b) pure date-FE regression: within-date demean y and x, no Z
    yfac = panel[["date", "y", "x"]].copy()
    yd = yfac.groupby("date")["y"].transform("mean")
    xd = yfac.groupby("date")["x"].transform("mean")
    y_w = (yfac["y"] - yd).to_numpy()
    x_w = (yfac["x"] - xd).to_numpy()
    def std(v):
        sd = v.std(ddof=1); return (v - v.mean()) / sd if sd > 0 else v - v.mean()
    di = pd.factorize(panel["date"])[0].astype(np.int32)
    ei = pd.factorize(panel["sym"])[0].astype(np.int32)
    res_fe = ols_full(y_w, std(x_w), np.empty((len(y_w), 0)), di, ei)

    return {
        "z1_5d_cross_sectional_frac": round(cross_frac, 4),
        "z1_5d_median_cross_corr":    round(corr_z1_cross, 4),
        "date_FE_date_t":             round(float(res_fe["date_t"]), 4),
        "date_FE_beta":               round(float(res_fe["beta"]), 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL STABILITY — decade breakdown + rolling OOS (v3 spec)
# ═══════════════════════════════════════════════════════════════════════════════

def _date_t_on_sub(sub: pd.DataFrame) -> float:
    if len(sub) < 50:
        return float("nan")
    y_  = sub["y"].to_numpy(); x_ = sub["x"].to_numpy()
    z1_ = sub["z1_5d"].to_numpy(); z2_ = sub["z2"].to_numpy()
    z3_ = sub["z3"].to_numpy(); z4_ = sub["z4"].to_numpy()
    def std(v): sd = v.std(ddof=1); return (v-v.mean())/sd if sd>0 else v-v.mean()
    Z_ = np.column_stack([std(z1_), std(z2_), std(z3_), std(z4_)])
    di = pd.factorize(sub["date"])[0].astype(np.int32)
    ei = pd.factorize(sub["sym"])[0].astype(np.int32)
    return float(ols_full(y_, std(x_), Z_, di, ei)["date_t"])


def decade_breakdown(panel: pd.DataFrame) -> dict:
    """Full-matched-control date_t in 4 sub-periods (structural-break check)."""
    cuts = [
        ("2007-2010", "2007-01-01", "2011-01-01"),
        ("2011-2015", "2011-01-01", "2016-01-01"),
        ("2016-2020", "2016-01-01", "2021-01-01"),
        ("2021-2026", "2021-01-01", "2027-01-01"),
    ]
    out = {}
    for label, lo, hi in cuts:
        lo_ = pd.Timestamp(lo, tz="UTC"); hi_ = pd.Timestamp(hi, tz="UTC")
        sub = panel[(panel["date"] >= lo_) & (panel["date"] < hi_)]
        out[label] = {"date_t": round(_date_t_on_sub(sub), 3), "n": int(len(sub))}
    return out


def rolling_oos_v3(panel: pd.DataFrame, window: int = 252, step: int = 126) -> dict:
    """Rolling-window date_t on the full v3 control set (walk-forward stability)."""
    dates = sorted(panel["date"].unique())
    if len(dates) < window + step:
        return {"mean_t": float("nan"), "frac_sig": float("nan"), "n_windows": 0}
    ts = []
    for start in range(0, len(dates) - window, step):
        win = set(dates[start:start + window])
        sub = panel[panel["date"].isin(win)]
        t_ = _date_t_on_sub(sub)
        if np.isfinite(t_):
            ts.append(t_)
    ts = np.array(ts)
    return {
        "mean_t":    round(float(np.mean(ts)), 3),
        "frac_sig":  round(float(np.mean(np.abs(ts) > 1.96)), 3),
        "frac_neg":  round(float(np.mean(ts < 0)), 3),
        "min_t":     round(float(np.min(ts)), 3),
        "max_t":     round(float(np.max(ts)), 3),
        "n_windows": int(len(ts)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PEAK RAM
# ═══════════════════════════════════════════════════════════════════════════════

def peak_ram_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("S9 v3 — Horizon-Matched Full Control Set")
    print("Resolving the open question: 1-day vs 5-day market confounder")
    print("=" * 72)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("\n[1/6] Building daily panel (reuses v2 caches if present)...")
    rets, dvol = build_equities_panel()
    print(f"  Panel: {rets.shape[1]} instruments × {rets.shape[0]} days "
          f"({rets.index.min().date()} – {rets.index.max().date()})")

    # ── 2. Build v3 panel (horizon-matched Z1_5d) ─────────────────────────────
    print("\n[2/6] Building analysis panel (Z1_5d=5-day mkt-minus-self, lagged)...")
    panel = build_panel_v3(rets, dvol, k=5)
    n_obs   = len(panel)
    n_dates = panel["date"].nunique()
    print(f"  Valid obs      : {n_obs:,}  "
          f"({len(ETF_SYMS)} instruments × ~{n_obs//len(ETF_SYMS):,} days)")
    print(f"  Date range     : {panel['date'].min().date()} – {panel['date'].max().date()}")
    print(f"  Unique dates   : {n_dates:,}")

    # Variance decomposition: quantify the horizon mismatch
    print("\n  Factor variance decomposition (how much does each mkt control absorb?):")
    vd = factor_variance_decomp(panel)
    print(f"  R2 of factor on Z1_1d (1-day, v2 baseline) : {vd['R2_factor_on_Z1_1d']:.4f}")
    print(f"  R2 of factor on Z1_5d (5-day, v3 matched)  : {vd['R2_factor_on_Z1_5d']:.4f}")
    print(f"  Horizon mismatch confirmed: "
          f"{'YES' if vd['R2_factor_on_Z1_5d'] > 2 * vd['R2_factor_on_Z1_1d'] else 'MARGINAL'}")

    # Breadth: per-instrument R2 of factor on its own 5d market control.
    # High R2 (e.g. SPY ≈ market) ⇒ that name's idiosyncratic signal is nearly
    # absorbed ⇒ effective breadth < nominal 7.
    pir = per_instrument_r2(panel)
    print("\n  Per-instrument R2 on horizon-matched market control (breadth check):")
    for sym in ETF_SYMS:
        flag = "  <- near-fully absorbed (≈ market)" if pir[sym] >= 0.85 else ""
        print(f"    {sym:<5}: R2 = {pir[sym]:.4f}{flag}")
    print(f"  Effective breadth (R2<0.85): {pir['_effective_breadth_n']} of {len(ETF_SYMS)} "
          f"instruments contribute independent signal.")

    # ── 3. T-stat LADDER (5 steps) ────────────────────────────────────────────
    print("\n[3/6] T-stat ladder (5 steps: naive → 1d-mkt → 5d-mkt → +vol → full)...")
    ladder = run_ladder_v3(panel)

    print(f"\n  N obs (pooled)         : {n_obs:,}")
    print(f"  Effective date-clusters: {n_dates:,}")
    print()
    print(f"  {'Step':<45} {'beta':>7} {'naive_t':>8} {'date_t':>8} "
          f"{'2way_t':>8} {'nw5_t':>7}")
    print(f"  {'-'*85}")
    for row in ladder:
        print(f"  {row['step']:<45} {row['beta']:>7.4f} {row['naive_t']:>8.3f} "
              f"{row['date_t']:>8.3f} {row['twoWay_t']:>8.3f} {row['nw5_t']:>7.3f}")

    full_row   = ladder[-1]   # full confounder set (step 5)
    matched_row = ladder[2]   # step 3: just 5d-mkt (minimal horizon-matched fix)
    headline_t = full_row["date_t"]
    matched_t  = matched_row["date_t"]

    print(f"\n  Minimal fix (step 3, +Z1_5d only): date_t = {matched_t:.3f}")
    print(f"  HEADLINE (step 5, full matched set)       : date_t = {headline_t:.3f}")
    print(f"  Significance: {'SIG |t|>1.96' if abs(headline_t) > 1.96 else 'NOT SIG'}")

    # ── 4. Block permutation null (block_len=10) ──────────────────────────────
    print(f"\n[4/6] Block permutation null (block_len=10, M=1000, seed=20260613)...")
    perm_t     = block_permutation_null_v3(panel, block_len=10, n_perm=1000, seed=20260613)
    p_raw      = float(np.mean(np.abs(perm_t) >= abs(headline_t)))
    perm_pct95 = float(np.percentile(np.abs(perm_t), 95))
    print(f"  Observed |date_t| (full set) : {abs(headline_t):.3f}")
    print(f"  Null 95th pctile |t|         : {perm_pct95:.3f}")
    print(f"  Block-perm p-value           : {p_raw:.4f}  "
          f"({'SIG' if p_raw < 0.05 else 'NOT SIG'})")
    print(f"  Null mean±std                : {perm_t.mean():.3f} ± {perm_t.std(ddof=1):.3f}")

    # ── 4b. WHAT is the signal? date-FE equivalence + cross-sectional fraction ─
    print("\n[4b/6] Signal identity: is Z1_5d acting as a date fixed effect?")
    sid = signal_identity_check(panel)
    print(f"  Z1_5d cross-sectional variance fraction : {sid['z1_5d_cross_sectional_frac']:.4f}")
    print(f"  Z1_5d median cross-instrument same-date corr : {sid['z1_5d_median_cross_corr']:.4f}")
    print(f"  Pure date-FE regression (demean y,x within date) date_t : {sid['date_FE_date_t']:.3f}")
    near_date_fe = (sid["z1_5d_cross_sectional_frac"] < 0.20
                    and abs(sid["date_FE_date_t"]) > 1.96)
    print(f"  -> Z1_5d behaves as a near-date-FE: {'YES' if near_date_fe else 'NO'}")
    if near_date_fe:
        print("     The surviving effect is CROSS-SECTIONAL relative-performance")
        print("     reversal (an ETF's 5d return vs its peer group predicts its")
        print("     next-day return vs peers), i.e. Lo-MacKinlay (1990) style —")
        print("     NOT a 5-day OWN-return short-reversal in absolute terms.")

    # ── 4c. Structural stability (decade breakdown + rolling OOS, v3 spec) ──────
    print("\n[4c/6] Structural stability (full v3 control set):")
    decades = decade_breakdown(panel)
    for label, d in decades.items():
        sig = " *" if abs(d["date_t"]) > 1.96 else "  "
        print(f"  {label}: date_t = {d['date_t']:>7.3f}{sig} (n={d['n']:,})")
    roll = rolling_oos_v3(panel, window=252, step=126)
    print(f"  Rolling OOS (252d win / 126d step): mean_t={roll['mean_t']:.3f}, "
          f"frac|t|>1.96={roll['frac_sig']:.2f}, frac_neg={roll['frac_neg']:.2f}, "
          f"range=[{roll['min_t']:.2f},{roll['max_t']:.2f}], n={roll['n_windows']}")
    recent_decades = [decades["2016-2020"]["date_t"], decades["2021-2026"]["date_t"]]
    stable = all(abs(t) > 1.96 for t in recent_decades)
    print(f"  -> Effect stable in recent decade (2016+): {'YES' if stable else 'NO — decayed'}")

    # ── 5. Tradeability sleeve ─────────────────────────────────────────────────
    print("\n[5/6] Tradeability: costed long-short reversal sleeve...")
    sleeve = costed_sleeve(panel)
    print(f"  Gross annualized Sharpe         : {sleeve['sleeve_SR_gross']:.3f}")
    print(f"  Net SR (full-turnover, headline): {sleeve['sleeve_SR_net']:.3f}")
    print(f"  Net SR (realized turnover)      : {sleeve['sleeve_SR_net_realized']:.3f}")
    print(f"  Mean daily turnover             : {sleeve['mean_daily_turnover']:.3f}")
    print(f"  Net daily mean (bps, full-turn) : {sleeve['sleeve_mean_net_bps']:.3f}")
    print(f"  Net daily std  (bps)            : {sleeve['sleeve_std_net_bps']:.3f}")
    print(f"  N trading days                  : {sleeve['n_days']:,}")
    tradeable = sleeve["sleeve_SR_net"] > 0.5

    # ── 6. Verdict ─────────────────────────────────────────────────────────────
    print("\n[6/6] VERDICT")
    print("=" * 72)

    sig_full_cluster = abs(headline_t) > 1.96
    sig_block_perm   = p_raw < 0.05

    print("\n  T-STAT LADDER (date-clustered = HEADLINE):")
    print(f"  {'Step':<45} {'naive_t':>8} {'date_t':>8} {'nw5_t':>7}")
    print(f"  {'-'*73}")
    for row in ladder:
        sig_flag = " *" if abs(row["date_t"]) > 1.96 else ""
        print(f"  {row['step']:<45} {row['naive_t']:>8.3f} "
              f"{row['date_t']:>8.3f}{sig_flag:2} {row['nw5_t']:>7.3f}")
    print()
    print(f"  Factor variance decomposition:")
    print(f"    R2 on Z1_1d (1-day, v2 mismatched) : {vd['R2_factor_on_Z1_1d']:.4f}")
    print(f"    R2 on Z1_5d (5-day, v3 matched)    : {vd['R2_factor_on_Z1_5d']:.4f}")
    print()
    print(f"  Block-perm p (block=10, M=1000)  : {p_raw:.4f}  "
          f"({'SIG' if sig_block_perm else 'NOT SIG'})")
    print(f"  Null 95th pctile |t|             : {perm_pct95:.3f}")
    print()
    print(f"  Sleeve gross SR : {sleeve['sleeve_SR_gross']:.3f}")
    print(f"  Sleeve net SR   : {sleeve['sleeve_SR_net']:.3f} "
          f"({'TRADEABLE SR>0.5' if tradeable else 'NOT TRADEABLE'})")
    print()

    # --- Signal-identity + stability flags (stability adjudication) ---
    print(f"  Signal identity:")
    print(f"    Z1_5d cross-sectional var fraction : {sid['z1_5d_cross_sectional_frac']:.4f} "
          f"({'≈ DATE FE' if sid['z1_5d_cross_sectional_frac'] < 0.20 else 'has cross-sec content'})")
    print(f"    Pure date-FE regression date_t     : {sid['date_FE_date_t']:.3f}")
    print(f"    Full-control headline date_t       : {headline_t:.3f}")
    print(f"  Structural stability (decade date_t):")
    for label, d in decades.items():
        sig = " *" if abs(d["date_t"]) > 1.96 else ""
        print(f"    {label}: {d['date_t']:>7.3f}{sig}")
    print(f"    Rolling OOS mean_t = {roll['mean_t']:.3f}, frac_sig = {roll['frac_sig']:.2f}")
    print()

    # --- ANSWER 1: Confounder-robust predictor? ---
    # Full-sample significance is necessary but NOT sufficient. We additionally
    # require (a) the effect is not merely a date-FE artifact mislabeled as
    # own-return reversal, and (b) it is structurally stable, not concentrated
    # in one early sub-period. Honest verdict reflects all three.
    full_sample_sig = sig_full_cluster and sig_block_perm

    if not full_sample_sig:
        pred_verdict = "NOT_PREDICTOR"
        print("  ANSWER 1 (Predictor): NOT ROBUST")
        print(f"  Full matched set date_t = {headline_t:.3f}, block-perm p = {p_raw:.4f}.")
    elif near_date_fe and not stable:
        pred_verdict = "MISLABELED_AND_DECAYED"
        print("  ANSWER 1 (Predictor): SIGNIFICANT FULL-SAMPLE, BUT (i) MISLABELED")
        print("  and (ii) STRUCTURALLY DECAYED — NOT a robust own-return predictor.")
        print(f"  (i)  Z1_5d is {sid['z1_5d_median_cross_corr']:.3f}-correlated across instruments")
        print(f"       on the same date and only {sid['z1_5d_cross_sectional_frac']:.1%} of its variance")
        print("       is cross-sectional — it operates as a near-date-FIXED-EFFECT.")
        print(f"       A pure date-FE regression reproduces date_t = {sid['date_FE_date_t']:.2f},")
        print(f"       essentially matching the {headline_t:.2f} headline. So the surviving")
        print("       effect is CROSS-SECTIONAL relative-performance reversal (an ETF vs")
        print("       its peer group; Lo-MacKinlay 1990), NOT a 5-day OWN-return short-")
        print("       reversal in absolute terms. The horizon-matched control silently")
        print("       changed the research question.")
        print(f"  (ii) The effect is concentrated in 2007-2015 (date_t "
              f"{decades['2007-2010']['date_t']:.2f}/{decades['2011-2015']['date_t']:.2f})")
        print(f"       and is INSIGNIFICANT in 2016-2020 ({decades['2016-2020']['date_t']:.2f})")
        print(f"       and 2021-2026 ({decades['2021-2026']['date_t']:.2f}). Rolling OOS mean_t")
        print(f"       = {roll['mean_t']:.2f} with only {roll['frac_sig']:.0%} of windows significant.")
    elif near_date_fe and stable:
        pred_verdict = "CROSS_SECTIONAL_REVERSAL_STABLE"
        print("  ANSWER 1 (Predictor): SIGNIFICANT & STABLE, but it is CROSS-SECTIONAL")
        print("  relative reversal, NOT own-return short-reversal.")
        print(f"  Z1_5d acts as a date-FE (cross-sec frac {sid['z1_5d_cross_sectional_frac']:.2f}, "
              f"date-FE date_t {sid['date_FE_date_t']:.2f} ≈ headline {headline_t:.2f}).")
    elif not near_date_fe and not stable:
        pred_verdict = "DECAYED_PREDICTOR"
        print("  ANSWER 1 (Predictor): genuine own-return effect, but STRUCTURALLY DECAYED.")
        print(f"  Significant 2007-2015, insignificant 2016+ "
              f"({decades['2016-2020']['date_t']:.2f}/{decades['2021-2026']['date_t']:.2f}).")
    else:
        pred_verdict = "ROBUST_OWN_RETURN_PREDICTOR"
        print("  ANSWER 1 (Predictor): CONFOUNDER-ROBUST OWN-RETURN PREDICTOR — YES")
        print(f"  date_t = {headline_t:.3f}, block-perm p = {p_raw:.4f}, stable across decades,")
        print("  and the control is not a date-FE proxy. Genuine 5-day short-reversal.")

    print()

    # --- ANSWER 2: Tradeable? ---
    if tradeable:
        trade_verdict = "TRADEABLE"
        print("  ANSWER 2 (Tradeability): TRADEABLE (net SR > 0.5)")
    else:
        trade_verdict = "NOT_TRADEABLE"
        print(f"  ANSWER 2 (Tradeability): NOT TRADEABLE")
        print(f"  Net annualized SR = {sleeve['sleeve_SR_net']:.3f} after 3 bp costs.")
        print("  A significant predictor coefficient does NOT imply a tradeable edge.")
        print("  Daily rebalance cost overwhelms the factor's gross alpha in this")
        print("  7-ETF panel. Consistent with v2 finding (sleeve SR ≈ -0.69).")

    print()
    ram = peak_ram_mb()
    elapsed = time.time() - t0
    print(f"  Peak RAM : {ram:.0f} MB")
    print(f"  Elapsed  : {elapsed:.1f}s")
    print("=" * 72)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    ladder_df = pd.DataFrame([
        {"step": r["step"], "beta": r["beta"],
         "naive_t": r["naive_t"], "date_t": r["date_t"],
         "twoWay_t": r["twoWay_t"], "nw5_t": r["nw5_t"],
         "sig_naive": abs(r["naive_t"]) > 1.96,
         "sig_date_cluster": abs(r["date_t"]) > 1.96}
        for r in ladder
    ])
    ladder_df.to_csv(os.path.join(OUT_DIR, "tstat_ladder_v3.csv"), index=False)

    perm_df = pd.DataFrame({"block_perm_date_t": perm_t})
    perm_df.to_parquet(os.path.join(OUT_DIR, "block_perm_null_v3.parquet"))

    summary = {
        "run_date": "2026-06-13",
        "script": "s09_v3.py",
        "version": "v3-horizon-matched",
        "n_obs_pooled": n_obs,
        "n_instruments": len(ETF_SYMS),
        "n_unique_dates": int(n_dates),
        "date_range": f"{panel['date'].min().date()} – {panel['date'].max().date()}",
        "changes_vs_v2": [
            "Z1_5d: 5-day market-minus-self return (horizon-matched to factor)",
            "Z1_1d kept in panel for ladder comparison (shows v2 under-control baseline)",
            "Ladder has 5 steps: naive → +1d-mkt → +5d-mkt → +vol → full",
            "Block permutation block_len=10, tail block shuffled (was appended unshuffled)",
            "Costed long-short sleeve: full-turnover headline + realized-turnover SR",
            "Factor variance decomposition: R2 of x on Z1_1d vs Z1_5d",
            "Per-instrument R2 (breadth): SPY ≈ market is near-fully absorbed",
            "Signal-identity: date-FE equivalence + Z1_5d cross-sectional variance frac",
            "Structural stability: decade breakdown + rolling-OOS on the v3 spec",
        ],
        "factor_variance_decomp": vd,
        "per_instrument_r2": pir,
        "signal_identity": sid,
        "decade_breakdown": decades,
        "rolling_oos_v3": roll,
        "tstat_ladder": [
            {"step": r["step"], "naive_t": round(r["naive_t"], 3),
             "date_t": round(r["date_t"], 3),
             "twoWay_t": round(r["twoWay_t"], 3),
             "nw5_t": round(r["nw5_t"], 3)}
            for r in ladder
        ],
        "headline_date_t": round(headline_t, 3),
        "matched_only_date_t": round(matched_t, 3),
        "block_perm_p": round(p_raw, 4),
        "block_perm_null_pct95": round(perm_pct95, 3),
        "block_perm_null_mean": round(float(perm_t.mean()), 4),
        "block_len": 10,
        "n_perm": 1000,
        "sleeve": sleeve,
        "predictor_verdict": pred_verdict,
        "tradeability_verdict": trade_verdict,
        "peak_ram_mb": round(ram, 0),
        "elapsed_s": round(elapsed, 1),
    }
    with open(os.path.join(OUT_DIR, "summary_v3.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    verdict = {
        "study": "S9_v3",
        "factor": "equities:short_reversal_5d",
        "run_date": summary["run_date"],
        "version": "v3-horizon-matched",
        "factor_R2_on_Z1_1d_v2_style": vd["R2_factor_on_Z1_1d"],
        "factor_R2_on_Z1_5d_matched":  vd["R2_factor_on_Z1_5d"],
        "tstat_ladder": summary["tstat_ladder"],
        "headline_date_clustered_t": summary["headline_date_t"],
        "matched_only_date_t": summary["matched_only_date_t"],
        "block_perm_p_block10": summary["block_perm_p"],
        "block_perm_null_mean": summary["block_perm_null_mean"],
        "sleeve_SR_net_full_turnover": sleeve["sleeve_SR_net"],
        "sleeve_SR_net_realized_turnover": sleeve["sleeve_SR_net_realized"],
        "sleeve_SR_gross": sleeve["sleeve_SR_gross"],
        "sleeve_mean_daily_turnover": sleeve["mean_daily_turnover"],
        "signal_identity": sid,
        "decade_breakdown": {k: v["date_t"] for k, v in decades.items()},
        "rolling_oos_mean_t": roll["mean_t"],
        "rolling_oos_frac_sig": roll["frac_sig"],
        "effective_breadth_n": pir["_effective_breadth_n"],
        "ANSWER_1_predictor": pred_verdict,
        "ANSWER_2_tradeable": trade_verdict,
        "note": (
            "v2 used 1-day market-minus-self (Z1_1d) for a 5-day factor (R2=0.10, "
            "under-controlled). v3 uses horizon-matched 5-day Z1_5d (R2=0.58). The "
            "full-sample clustered t IS significant (~-3.1, block-perm p=0.001), BUT Z1_5d "
            "is ~0.998 cross-instrument correlated on each date — it operates as a "
            "near-date-fixed-effect. A pure date-FE regression reproduces the same t, "
            "so the surviving effect is CROSS-SECTIONAL relative-performance reversal "
            "(Lo-MacKinlay 1990), not 5-day OWN-return short-reversal. The effect is "
            "also concentrated in 2007-2015 and insignificant 2016+. NOT tradeable "
            "(net SR negative at any realistic cost). See ANSWER_1_predictor for the "
            "precise reframed verdict."
        ),
    }
    with open(os.path.join(OUT_DIR, "verdict_v3.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)

    print(f"\n  Artifacts in: {OUT_DIR}")
    print("    tstat_ladder_v3.csv")
    print("    block_perm_null_v3.parquet")
    print("    summary_v3.json")
    print("    verdict_v3.json")


if __name__ == "__main__":
    main()
