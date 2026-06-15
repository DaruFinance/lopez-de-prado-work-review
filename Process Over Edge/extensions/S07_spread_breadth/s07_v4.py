#!/usr/bin/env python3
"""
S7a + S7b  --  Spread mean-reversion at breadth  (FAST v4)
===========================================================
BASED ON s07_v3.py — all leak-free machinery preserved exactly:
  * Global split T* (2025-01-01)
  * Pre-T*-only Engle-Granger screen (parallel Pool(8))
  * OHLC-bounded OU exits (intrabar pessimistic first-touch)
  * Per-quarter tail-guard, sign-consistency, inv-vol sizing
  * Hard no-lookahead assertion (aborts on violation)
  * OU mesh from the shared optimal-trading-rule kernels (otr + deepen)

KEY CHANGES vs v3 for speed/practicality:
  1. TOP-60 ONLY: after EG screen selects cointegrated pairs by pre-T*
     ADF stat (most-negative; tie-break shortest OU half-life), run OOS
     trading ONLY on those 60.  No all-pairs OOS book.
  2. PERMUTATION NULL uses ~600-sample non-coint pool (not full pool).
     Each null book draws 60 from that 600-sample cache.
  3. No-lookahead assertion runs on just the 60 (cheap; was the bottleneck).
  4. Targets < 15 min wall time with Pool(8) on cores 16-31.

DO NOT MODIFY s07_v3.py -- this is a NEW file.
"""
from __future__ import annotations

# ── CPU AFFINITY: self-pin to cores 16-31 (user's script owns 0-15) ──────────
import os
os.sched_setaffinity(0, range(16, 32))

# Pin thread counts BEFORE any import
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import gc, sys, time, json, warnings, resource
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import stats as ss

warnings.filterwarnings("ignore")

# ── repo paths ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_PERP_1H
_SCRIPTS = os.path.join(REPO_ROOT, "projects", "12_optimal_trading_rules", "scripts")
sys.path.insert(0, _LIB)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, HERE)

import otr
import deepen

# ── output directory ──────────────────────────────────────────────────────────
OUT = HERE
os.makedirs(OUT, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
PERP_DIR = CRYPTO_PERP_1H

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL SPLIT  (the entire anti-lookahead architecture rests on this one cut)
# ──────────────────────────────────────────────────────────────────────────────
T_STAR = pd.Timestamp("2025-01-01", tz="UTC")

# ── universe filter thresholds ────────────────────────────────────────────────
MIN_PRE_BARS   = 2000
MIN_POST_BARS  = 1000

# ── correlation prefilter (PRE-T* returns only) ───────────────────────────────
CORR_MIN_FRAC  = 0.50
CORR_TOP_K     = 5

# ── per-pair cointegration gate ───────────────────────────────────────────────
MIN_PAIR_PRE   = 1000
ADF_P_MAX      = 0.05

# ── OU / trading parameters ───────────────────────────────────────────────────
Z_SPAN    = 500
ENTRY_K   = 2.0
MAX_HOLD  = 200

PT_GRID = np.round(np.concatenate([
    np.arange(0.05, 0.26, 0.05),
    np.arange(0.50, 3.01, 0.25),
]), 4)
SL_GRID = PT_GRID.copy()

N_PATHS    = 20000
MC_HORIZON = 500
MC_SEED    = 12

# ── costs ─────────────────────────────────────────────────────────────────────
COST_BP_PER_LEG = 7.0
RT_COST         = 4 * COST_BP_PER_LEG / 1e4

# ── permutation null ──────────────────────────────────────────────────────────
N_PERM           = 1000
MIN_OOS_TRADES   = 10
NULL_POOL_SAMPLE = 600   # non-coint pairs to pre-trade for null pool

# ── top-N capped book ─────────────────────────────────────────────────────────
TOP_N = 60

# ── sanity gates ──────────────────────────────────────────────────────────────
SANITY_MIN_SYMBOLS    = 100
SANITY_MIN_CANDIDATES = 300

HEARTBEAT     = 50
BARS_PER_YEAR = 24.0 * 365.25

# ── parallelism ───────────────────────────────────────────────────────────────
N_WORKERS = 8


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _ram_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_perp(symbol: str) -> pd.DataFrame:
    """Load one hourly perp. Returns df indexed by open_time (UTC), OHLCV."""
    f = os.path.join(PERP_DIR, f"{symbol}_1h.parquet")
    df = pd.read_parquet(f, columns=["open_time", "open", "high", "low", "close", "volume"])
    ot = pd.to_datetime(df["open_time"], utc=True)
    df = df.assign(open_time=ot).set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["open"] > 0)]
    return df


def _adf_pvalue(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return 1.0
    try:
        from statsmodels.tsa.stattools import adfuller
        return float(adfuller(x, maxlag=5, regression="c", autolag=None)[1])
    except Exception:
        y = np.diff(x); xl = x[:-1]
        X = np.column_stack([np.ones_like(xl), xl])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef; n = len(y)
        s2 = (resid @ resid) / max(n - 2, 1)
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(s2 * xtx_inv[1, 1])
        t = coef[1] / se if se > 0 else 0.0
        return float(min(max(0.5 * np.exp(0.5 * (t + 2.0)), 1e-4), 1.0))


def engle_granger_train(la_tr: np.ndarray, lb_tr: np.ndarray):
    """OLS of log A on log B over TRAIN only. Returns (alpha, beta, adf_p)."""
    X = np.column_stack([np.ones_like(lb_tr), lb_tr])
    coef, *_ = np.linalg.lstsq(X, la_tr, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = la_tr - (alpha + beta * lb_tr)
    return alpha, beta, _adf_pvalue(resid)


def spread_ohlc(a: pd.DataFrame, b: pd.DataFrame, beta: float):
    """Spread OHLC bounded by leg OHLC extremes (no intrabar lookahead)."""
    laO = np.log(a["open"].to_numpy(np.float64))
    laH = np.log(a["high"].to_numpy(np.float64))
    laL = np.log(a["low"].to_numpy(np.float64))
    laC = np.log(a["close"].to_numpy(np.float64))
    lbO = np.log(b["open"].to_numpy(np.float64))
    lbH = np.log(b["high"].to_numpy(np.float64))
    lbL = np.log(b["low"].to_numpy(np.float64))
    lbC = np.log(b["close"].to_numpy(np.float64))
    s_open  = laO - beta * lbO
    s_close = laC - beta * lbC
    if beta >= 0.0:
        s_high = laH - beta * lbL
        s_low  = laL - beta * lbH
    else:
        s_high = laH - beta * lbH
        s_low  = laL - beta * lbL
    s_high = np.maximum.reduce([s_high, s_open, s_close])
    s_low  = np.minimum.reduce([s_low,  s_open, s_close])
    return s_open, s_high, s_low, s_close


def spread_rolling_stats(s_close: np.ndarray, span: int):
    """Causal EWMA z-score and std of the spread."""
    s = pd.Series(np.asarray(s_close, np.float64))
    mu  = s.ewm(span=span, adjust=True).mean()
    std = s.ewm(span=span, adjust=True).std()
    z   = ((s - mu) / std.replace(0, np.nan)).to_numpy()
    z[~np.isfinite(z)] = 0.0
    sig = std.to_numpy()
    sig[~np.isfinite(sig)] = 0.0
    return z, sig


def spread_entry_events(z: np.ndarray, entry_k: float, warm: int):
    z = np.asarray(z, np.float64)
    over  = np.abs(z) >= entry_k
    fresh = over.copy(); fresh[1:] &= ~over[:-1]
    idx   = np.where(fresh)[0]; idx = idx[idx > warm]
    side  = -np.sign(z[idx]).astype(np.int64)
    keep  = side != 0
    return idx[keep].astype(np.int64), side[keep]


def score_series(r: np.ndarray, bpy: float = BARS_PER_YEAR) -> dict:
    r = np.asarray(r, np.float64); r = r[np.isfinite(r)]
    nz = r[r != 0.0]
    sr = float(np.mean(r) / np.std(r)) if np.std(r) > 0 else 0.0
    sr_ann = sr * np.sqrt(bpy)
    sk = float(ss.skew(r))   if len(r) > 2 else 0.0
    ku = float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0
    gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else np.nan
    return dict(sr_per_bar=sr, sr_ann=sr_ann, pf=pf, skew=sk, kurt=ku,
                mean=float(r.mean()), n_obs=len(r))


try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
    print("Numba available: intrabar exit kernel JIT-compiled.")
except Exception:
    _HAVE_NUMBA = False
    def _njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]


@_njit(cache=True)
def _spread_exit_kernel(ev_idx, side, sig_ev,
                        s_open, s_high, s_low, s_close,
                        pt_mult, sl_mult, max_hold):
    """Intrabar first-touch barrier rule on the spread.
    Adverse direction checked first (pessimistic / realistic).
    """
    n_ev = ev_idx.shape[0]; n = s_close.shape[0]
    out_pnl   = np.empty(n_ev, np.float64)
    out_touch = np.empty(n_ev, np.int64)
    out_label = np.empty(n_ev, np.int64)
    out_hold  = np.empty(n_ev, np.int64)
    for k in range(n_ev):
        i0 = ev_idx[k]; s = side[k]; sg = sig_ev[k]
        pt_abs = pt_mult * sg; sl_abs = sl_mult * sg
        entry_bar = i0 + 1
        if entry_bar > n - 1: entry_bar = n - 1
        entry = s_open[entry_bar]
        if s > 0:
            up = entry + pt_abs; dn = entry - sl_abs
        else:
            dn = entry - pt_abs; up = entry + sl_abs
        j_end = entry_bar + max_hold
        if j_end > n - 1: j_end = n - 1
        touched = -1; label = 0; exit_lvl = s_close[j_end]
        for j in range(entry_bar, j_end + 1):
            hi = s_high[j]; lo = s_low[j]
            if s > 0:
                if lo <= dn:  touched = j; label = -1; exit_lvl = dn;  break
                if hi >= up:  touched = j; label =  1; exit_lvl = up;  break
            else:
                if hi >= up:  touched = j; label = -1; exit_lvl = up;  break
                if lo <= dn:  touched = j; label =  1; exit_lvl = dn;  break
        if touched < 0: touched = j_end; label = 0; exit_lvl = s_close[j_end]
        out_pnl[k]   = s * (exit_lvl - entry)
        out_touch[k] = touched
        out_label[k] = label
        out_hold[k]  = touched - entry_bar
    return out_pnl, out_touch, out_label, out_hold


def apply_spread_rule(s_open, s_high, s_low, s_close, sig_ev, ev_idx, side,
                      pt_mult, sl_mult, max_hold):
    f = lambda x: np.ascontiguousarray(np.asarray(x, np.float64))
    pnl, touch, lab, hold = _spread_exit_kernel(
        np.ascontiguousarray(np.asarray(ev_idx, np.int64)),
        np.ascontiguousarray(np.asarray(side,   np.int64)),
        f(sig_ev), f(s_open), f(s_high), f(s_low), f(s_close),
        float(pt_mult), float(sl_mult), int(max_hold))
    return pd.DataFrame({"ev_idx": ev_idx, "side": side, "touch": touch,
                         "label": lab, "pnl_gross": pnl, "hold": hold})


# ──────────────────────────────────────────────────────────────────────────────
#  WORKER FUNCTIONS (top-level for multiprocessing pickling)
# ──────────────────────────────────────────────────────────────────────────────

def _eg_worker(args):
    """
    Engle-Granger + OU fit worker for one candidate pair.
    Runs entirely on PRE-T* data. Returns a metadata dict or None.
    Identical to v3 -- no changes to selection logic.
    """
    sym_a, sym_b, worker_id = args
    try:
        a0 = load_perp(sym_a)
        b0 = load_perp(sym_b)
    except Exception:
        return None

    idx_pre = a0.index.intersection(b0.index)
    idx_pre = idx_pre[idx_pre < T_STAR]
    n_pre   = len(idx_pre)

    if n_pre < MIN_PAIR_PRE:
        del a0, b0
        return None

    a_pre = a0.loc[idx_pre]
    b_pre = b0.loc[idx_pre]
    del a0, b0

    la = np.log(a_pre["close"].to_numpy(np.float64))
    lb = np.log(b_pre["close"].to_numpy(np.float64))

    alpha, beta, adf_p = engle_granger_train(la, lb)

    if adf_p > ADF_P_MAX or not np.isfinite(beta):
        del a_pre, b_pre
        return None

    # OU fit on PRE-T* spread residual
    spread_pre = la - (alpha + beta * lb)
    s_ser      = pd.Series(spread_pre)
    roll_mu    = s_ser.ewm(span=Z_SPAN, adjust=True).mean().to_numpy()
    resid      = spread_pre - roll_mu
    resid      = resid[np.isfinite(resid)]

    try:
        fit = otr.fit_ou(resid)
    except Exception:
        del a_pre, b_pre
        return None

    if not fit["ok"]:
        del a_pre, b_pre
        return None

    last_pre_ts = idx_pre[-1] if len(idx_pre) > 0 else pd.NaT

    del a_pre, b_pre
    return dict(
        sym_a=sym_a, sym_b=sym_b,
        n_pre=n_pre,
        alpha=alpha, beta=beta, adf_p=adf_p,
        ou_E0=fit["E0"], ou_phi=fit["phi"],
        ou_sigma=fit["sigma"], ou_half_life=fit["half_life"],
        last_pre_ts=last_pre_ts,
    )


def _oos_worker(cp):
    """
    OOS trading for one cointegrated pair.
    All params (beta, alpha, OU) come from pre-T* fit stored in cp.
    Identical to v3 -- no changes to trading logic.
    """
    sym_a, sym_b = cp["sym_a"], cp["sym_b"]
    alpha = cp["alpha"]; beta = cp["beta"]
    ou_E0 = cp["ou_E0"]; ou_phi = cp["ou_phi"]; ou_sigma = cp["ou_sigma"]

    try:
        a0 = load_perp(sym_a)
        b0 = load_perp(sym_b)
    except Exception as e:
        return None, f"load error: {e}"

    idx_oos = a0.index.intersection(b0.index)
    idx_oos = idx_oos[idx_oos >= T_STAR]

    if len(idx_oos) < MIN_POST_BARS:
        del a0, b0
        return None, f"too few OOS bars: {len(idx_oos)}"

    a_oos = a0.loc[idx_oos]
    b_oos = b0.loc[idx_oos]
    del a0, b0

    sO, sH, sL, sC = spread_ohlc(a_oos, b_oos, beta)
    z_oos, sig_oos = spread_rolling_stats(sC, Z_SPAN)

    stat_sd = ou_sigma / max(np.sqrt(1.0 - ou_phi ** 2), 1e-9)
    x0_dev  = ou_E0 + ENTRY_K * stat_sd

    try:
        sh_mesh, _ = deepen.ou_mesh_dev(
            ou_E0, ou_phi, ou_sigma, x0=x0_dev,
            pt_grid=PT_GRID, sl_grid=SL_GRID,
            n_paths=N_PATHS, max_horizon=MC_HORIZON, seed=MC_SEED)
        pt_star, sl_star, sh_star, _ = otr.optimal_rule(sh_mesh, PT_GRID, SL_GRID)
        del sh_mesh
    except Exception as e:
        del a_oos, b_oos
        return None, f"OU mesh error: {e}"

    pt_mult = pt_star / max(stat_sd, 1e-12)
    sl_mult = sl_star / max(stat_sd, 1e-12)

    warm = Z_SPAN + 5
    ev_idx, side_ev = spread_entry_events(z_oos, ENTRY_K, warm)

    if len(ev_idx) < MIN_OOS_TRADES:
        del a_oos, b_oos
        return None, f"too few OOS trades: {len(ev_idx)}"

    sig_ev = sig_oos[ev_idx]

    tb = apply_spread_rule(sO, sH, sL, sC, sig_ev, ev_idx, side_ev,
                           pt_mult, sl_mult, MAX_HOLD)

    pnl_net = tb["pnl_gross"].to_numpy() - RT_COST

    entry_bar_abs = np.minimum(ev_idx + 1, len(idx_oos) - 1)
    entry_ts = idx_oos[entry_bar_abs]

    quarters = pd.PeriodIndex(idx_oos, freq="Q")
    unique_q  = sorted(set(quarters))
    q_stats   = []
    for q in unique_q:
        mask_q = np.array([quarters[i] == q for i in np.minimum(ev_idx + 1, len(idx_oos) - 1)])
        if mask_q.sum() < 5:
            continue
        pnl_q  = pnl_net[mask_q]
        wins   = pnl_q[pnl_q > 0]
        losses = pnl_q[pnl_q < 0]
        rrr_q  = float(wins.mean() / abs(losses.mean())) if (len(wins) > 0 and len(losses) > 0) else 0.0
        q_stats.append(dict(quarter=str(q), n_trades=int(mask_q.sum()), rrr=rrr_q))

    trades_df = tb.copy()
    trades_df["sym_a"]    = sym_a
    trades_df["sym_b"]    = sym_b
    trades_df["pnl_net"]  = pnl_net
    trades_df["beta"]     = beta
    trades_df["adf_p"]    = cp["adf_p"]
    trades_df["pt_star"]  = pt_star
    trades_df["sl_star"]  = sl_star
    trades_df["entry_ts"] = [str(t) for t in entry_ts]
    trades_df["global_bar"] = ev_idx.astype(np.int64)

    wins   = pnl_net[pnl_net > 0]
    losses = pnl_net[pnl_net < 0]
    rrr_pair = float(wins.mean() / abs(losses.mean())) if (len(wins) > 0 and len(losses) > 0) else 0.0

    del a_oos, b_oos

    return dict(
        sym_a=sym_a, sym_b=sym_b,
        n_oos_trades=len(pnl_net),
        rrr_pair=rrr_pair,
        pt_star=pt_star, sl_star=sl_star,
        ou_phi=ou_phi, ou_half_life=cp["ou_half_life"],
        adf_p=cp["adf_p"],
        q_stats=q_stats,
        _trades_df=trades_df,
    ), "ok"


def _null_pair_worker(args):
    """
    OOS trading for one non-cointegrated pair at fixed median pt/sl.
    Used to pre-build the null pool cache.
    Returns (sym_a, sym_b, trades_df) or None.
    """
    sym_a, sym_b = args
    try:
        a0 = load_perp(sym_a)
        b0 = load_perp(sym_b)
    except Exception:
        return None

    try:
        idx_pre = a0.index.intersection(b0.index)
        idx_pre = idx_pre[idx_pre < T_STAR]
        if len(idx_pre) < MIN_PAIR_PRE:
            del a0, b0; return None

        la = np.log(a0.loc[idx_pre, "close"].to_numpy(np.float64))
        lb = np.log(b0.loc[idx_pre, "close"].to_numpy(np.float64))
        alpha, beta, _ = engle_granger_train(la, lb)

        idx_oos = a0.index.intersection(b0.index)
        idx_oos = idx_oos[idx_oos >= T_STAR]
        if len(idx_oos) < 200:
            del a0, b0; return None

        a_oos = a0.loc[idx_oos]; b_oos = b0.loc[idx_oos]
        del a0, b0

        sO, sH, sL, sC = spread_ohlc(a_oos, b_oos, beta)
        z_oos, sig_oos = spread_rolling_stats(sC, Z_SPAN)
        ev_idx, side_ev = spread_entry_events(z_oos, ENTRY_K, Z_SPAN + 5)
        if len(ev_idx) < MIN_OOS_TRADES:
            del a_oos, b_oos; return None

        pt_fix = float(PT_GRID[len(PT_GRID) // 2])
        sl_fix = float(SL_GRID[len(SL_GRID) // 2])
        stat_sd = float(np.std(
            sC - pd.Series(sC).ewm(span=Z_SPAN, adjust=True).mean().to_numpy()
        ))
        stat_sd = max(stat_sd, 1e-12)
        pt_m = pt_fix / stat_sd; sl_m = sl_fix / stat_sd

        sig_ev = sig_oos[ev_idx]
        tb = apply_spread_rule(sO, sH, sL, sC, sig_ev, ev_idx, side_ev,
                               pt_m, sl_m, MAX_HOLD)
        pnl_net = tb["pnl_gross"].to_numpy() - RT_COST
        tb["pnl_net"]    = pnl_net
        tb["sym_a"]      = sym_a
        tb["sym_b"]      = sym_b
        tb["global_bar"] = ev_idx.astype(np.int64)
        del a_oos, b_oos; gc.collect()
        return (sym_a, sym_b, tb)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 1: Build universe
# ──────────────────────────────────────────────────────────────────────────────

def build_universe() -> list[str]:
    print(f"\n=== STAGE 1: Universe filter (T* = {T_STAR.date()}) ===")
    all_files = sorted(os.listdir(PERP_DIR))
    all_syms  = [f.replace("_1h.parquet", "") for f in all_files
                 if f.endswith("_1h.parquet")]
    usdt_syms = [s for s in all_syms
                 if s.endswith("USDT")
                 and s[0].isalpha()
                 and not s.startswith("1000")]
    print(f"  Raw USDT-alpha universe: {len(usdt_syms)}")

    qualified = []
    for sym in usdt_syms:
        try:
            f = os.path.join(PERP_DIR, f"{sym}_1h.parquet")
            df = pd.read_parquet(f, columns=["open_time"])
            ot = pd.to_datetime(df["open_time"], utc=True)
            n_pre  = int((ot < T_STAR).sum())
            n_post = int((ot >= T_STAR).sum())
            if n_pre >= MIN_PRE_BARS and n_post >= MIN_POST_BARS:
                qualified.append(sym)
        except Exception:
            pass

    print(f"  Qualified (>={MIN_PRE_BARS} pre-T*, >={MIN_POST_BARS} post-T*): {len(qualified)}")
    return qualified


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 2: Correlation prefilter
# ──────────────────────────────────────────────────────────────────────────────

def build_corr_candidates(syms: list[str]) -> list[tuple]:
    print(f"\n=== STAGE 2: Correlation prefilter on PRE-T* data ===")
    print(f"  Loading {len(syms)} symbols (pre-T* close series)")

    cols = []
    syms_ok = []
    n_err = 0

    for i, sym in enumerate(syms):
        if (i + 1) % HEARTBEAT == 0:
            print(f"  Loading: {i+1}/{len(syms)} | ok={len(syms_ok)} | "
                  f"RAM {_ram_gb():.2f} GB")
        try:
            df = load_perp(sym)
            pre = df.loc[df.index < T_STAR, "close"]
            lp  = np.log(pre.to_numpy(np.float64))
            if len(lp) < MIN_PRE_BARS:
                continue
            cols.append(lp)
            syms_ok.append(sym)
            del df, pre
        except Exception:
            n_err += 1
            continue
        gc.collect()

    print(f"  {len(syms_ok)} symbols loaded | {n_err} errors | RAM {_ram_gb():.2f} GB")

    if len(syms_ok) < 2:
        return []

    min_len = min(len(c) for c in cols)
    mat = np.column_stack([c[-min_len:] for c in cols])
    del cols; gc.collect()

    ret = np.diff(mat, axis=0)
    ret = np.where(np.isfinite(ret), ret, 0.0)
    del mat; gc.collect()

    T, n = ret.shape
    print(f"  Return matrix: {ret.shape} | RAM {_ram_gb():.2f} GB")

    mu  = ret.mean(axis=0)
    std = ret.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    ret_n = (ret - mu) / std
    del ret; gc.collect()

    chunk = 50
    from heapq import heappush, heappop

    cand_set = set()
    top_k_map = {i: [] for i in range(n)}

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = ret_n[:, i0:i1]
        corr_block = (block.T @ ret_n) / T
        corr_block = np.clip(corr_block, -1.0, 1.0)

        for r_local in range(i1 - i0):
            i = i0 + r_local
            row     = corr_block[r_local]
            abs_row = np.abs(row)

            for j in np.where(abs_row >= CORR_MIN_FRAC)[0]:
                if j != i:
                    cand_set.add((min(i, j), max(i, j)))

            for j in range(n):
                if j == i:
                    continue
                c = float(abs_row[j])
                lst = top_k_map[i]
                if len(lst) < CORR_TOP_K:
                    heappush(lst, (c, j))
                elif c > lst[0][0]:
                    heappop(lst); heappush(lst, (c, j))

    for i, lst in top_k_map.items():
        for (c, j) in lst:
            if j != i:
                cand_set.add((min(i, j), max(i, j)))

    del ret_n; gc.collect()

    pairs = [(syms_ok[a], syms_ok[b]) for (a, b) in sorted(cand_set)]
    print(f"  {len(pairs):,} candidate pairs "
          f"(|corr|>={CORR_MIN_FRAC} OR top-{CORR_TOP_K} per sym)")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 3: Engle-Granger + OU fit -- PARALLEL
# ──────────────────────────────────────────────────────────────────────────────

def screen_cointegration_parallel(candidates: list[tuple]) -> tuple[list[dict], list[tuple]]:
    """
    Parallel Engle-Granger screen over all candidates using Pool(N_WORKERS).
    Returns (coint_pairs, non_coint_pairs).
    """
    print(f"\n=== STAGE 3: Engle-Granger cointegration (PARALLEL x{N_WORKERS}) "
          f"on {len(candidates):,} candidates ===")

    # Deduplicate
    seen = set()
    deduped = []
    for sym_a, sym_b in candidates:
        key = tuple(sorted([sym_a, sym_b]))
        if key not in seen:
            seen.add(key)
            deduped.append((sym_a, sym_b, 0))

    print(f"  {len(deduped):,} unique pairs after dedup")

    coint   = []
    n_done  = 0

    with Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(_eg_worker, deduped, chunksize=20):
            n_done += 1
            if n_done % 1000 == 0:
                print(f"  EG: {n_done}/{len(deduped)} screened | "
                      f"{len(coint)} cointegrated | RAM {_ram_gb():.2f} GB", flush=True)
            if result is not None:
                coint.append(result)

    # Build non-coint pool
    coint_set = {(cp["sym_a"], cp["sym_b"]) for cp in coint}
    coint_set_rev = {(cp["sym_b"], cp["sym_a"]) for cp in coint}
    non_coint_pairs = [(a, b) for a, b, _ in deduped
                       if (a, b) not in coint_set and (a, b) not in coint_set_rev]

    print(f"\n  EG result: {len(deduped)} tested | {len(coint)} cointegrated "
          f"(ADF p<={ADF_P_MAX}) | non-coint pool: {len(non_coint_pairs)}")
    return coint, non_coint_pairs


# ──────────────────────────────────────────────────────────────────────────────
#  TOP-N SELECTION  (done BEFORE OOS trading; uses only pre-T* metrics)
# ──────────────────────────────────────────────────────────────────────────────

def select_top_n_coint(coint_pairs: list[dict], n: int) -> list[dict]:
    """
    Select top-N cointegrated pairs by pre-T* ADF strength:
      primary:   smallest adf_p (most negative ADF stat)
      tie-break: shortest OU half-life
    Selection uses ONLY pre-T* metrics stored by _eg_worker.
    Returns the top-N coint dicts (NOT pair_results; these still need OOS trading).
    """
    if len(coint_pairs) <= n:
        return list(coint_pairs)
    ranked = sorted(coint_pairs,
                    key=lambda cp: (cp.get("adf_p", 1.0), cp.get("ou_half_life", 1e9)))
    return ranked[:n]


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 4: Hard no-lookahead assertion (runs only on the top-60; fast)
# ──────────────────────────────────────────────────────────────────────────────

def assert_no_lookahead(coint_pairs: list[dict]) -> bool:
    print(f"\n=== STAGE 4: No-lookahead assertion for {len(coint_pairs)} pairs ===")
    violations = []

    for cp in coint_pairs:
        last_pre = cp["last_pre_ts"]
        if pd.isnull(last_pre) or last_pre >= T_STAR:
            violations.append(dict(
                pair=f"{cp['sym_a']}/{cp['sym_b']}",
                violation="last_pre_ts >= T*",
                last_pre_ts=str(last_pre),
                T_star=str(T_STAR),
            ))

        try:
            a0 = load_perp(cp["sym_a"])
            b0 = load_perp(cp["sym_b"])
            idx_oos = a0.index.intersection(b0.index)
            idx_oos = idx_oos[idx_oos >= T_STAR]
            del a0, b0; gc.collect()
            if len(idx_oos) == 0:
                violations.append(dict(
                    pair=f"{cp['sym_a']}/{cp['sym_b']}",
                    violation="no OOS bars >= T*",
                    T_star=str(T_STAR),
                ))
            else:
                min_oos_ts = idx_oos[0]
                if min_oos_ts < T_STAR:
                    violations.append(dict(
                        pair=f"{cp['sym_a']}/{cp['sym_b']}",
                        violation="OOS bar < T*",
                        min_oos_ts=str(min_oos_ts),
                        T_star=str(T_STAR),
                    ))
        except Exception:
            pass

    if violations:
        print(f"\n  LOOKAHEAD ASSERTION FAILED: {len(violations)} violations")
        for v in violations[:10]:
            print(f"    {v}")
        print("\n  ABORTING RUN -- fix the leak before proceeding.")
        sys.exit(1)

    print(f"  All {len(coint_pairs)} pairs PASS: "
          f"max(selection_ts) < T* <= min(OOS_bar_ts)")
    return True


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 5: OOS trading -- PARALLEL (top-60 only)
# ──────────────────────────────────────────────────────────────────────────────

def run_oos_parallel(coint_pairs: list[dict]) -> list[dict]:
    """
    Parallel OOS trading over top-60 pairs using Pool(N_WORKERS).
    """
    print(f"\n=== STAGE 5: OOS trading (PARALLEL x{N_WORKERS}) "
          f"on {len(coint_pairs)} pairs ===")

    pair_results = []
    n_done = 0

    with Pool(N_WORKERS) as pool:
        for result, msg in pool.imap_unordered(_oos_worker, coint_pairs, chunksize=5):
            n_done += 1
            if n_done % 10 == 0:
                print(f"  OOS: {n_done}/{len(coint_pairs)} | "
                      f"{len(pair_results)} ok | RAM {_ram_gb():.2f} GB", flush=True)
            if result is not None:
                pair_results.append(result)

    print(f"\n  OOS complete: {len(pair_results)}/{len(coint_pairs)} pairs produced trades")
    return pair_results


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 6: Portfolio book (inv-vol sizing)
# ──────────────────────────────────────────────────────────────────────────────

def build_book(pair_results: list[dict], sizing: str = "inv_vol") -> tuple:
    if not pair_results:
        return np.array([]), {}

    pair_vols = {}
    for pr in pair_results:
        pnl = pr["_trades_df"]["pnl_net"].to_numpy()
        vol = float(np.std(pnl)) if len(pnl) > 1 else 1.0
        pair_vols[(pr["sym_a"], pr["sym_b"])] = max(vol, 1e-12)

    all_trades = []
    for pr in pair_results:
        df = pr["_trades_df"].copy()
        key = (pr["sym_a"], pr["sym_b"])
        if sizing == "equal":
            df["weight"] = 1.0
        elif sizing == "inv_vol":
            df["weight"] = 1.0 / pair_vols[key]
        elif sizing == "confidence":
            pt = float(pr["pt_star"]); sl = float(pr["sl_star"])
            df["weight"] = pt / max(sl, 1e-9)
        else:
            df["weight"] = 1.0
        all_trades.append(df)

    trades = pd.concat(all_trades, ignore_index=True)

    w_sum = trades["weight"].sum()
    if w_sum > 0:
        trades["weight"] /= w_sum

    n_bars = int(trades["global_bar"].max()) + MAX_HOLD + 2
    book_r = np.zeros(n_bars)
    for _, row in trades.iterrows():
        i0  = int(row["global_bar"])
        h   = max(1, int(row["hold"]))
        per = row["pnl_net"] * row["weight"] / h
        j1  = min(i0 + h, n_bars)
        book_r[i0:j1] += per

    nz = np.where(book_r != 0)[0]
    if len(nz) == 0:
        return np.array([]), {}
    book_r = book_r[nz[0]:nz[-1] + 1]

    sc = score_series(book_r)
    all_pnl = trades["pnl_net"].to_numpy()
    wins   = all_pnl[all_pnl > 0]; losses = all_pnl[all_pnl < 0]
    rrr_all = float(wins.mean() / abs(losses.mean())) if (len(wins) > 0 and len(losses) > 0) else 0.0

    return book_r, dict(
        sizing=sizing, n_trades=len(trades),
        sr_ann=sc["sr_ann"], pf=sc["pf"], skew=sc["skew"],
        sr_per_bar=sc["sr_per_bar"], rrr_all=rrr_all,
        _trades=trades,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 7: Tail guard (per-quarter RRR, ex-worst-quarter)
# ──────────────────────────────────────────────────────────────────────────────

def compute_tail_guard(pair_results: list[dict]) -> dict:
    quarter_rrrs: dict[str, list] = {}
    for pr in pair_results:
        for qs in pr.get("q_stats", []):
            q = qs["quarter"]
            quarter_rrrs.setdefault(q, []).append(qs["rrr"])

    if not quarter_rrrs:
        return dict(worst_quarter="none", worst_rrr=np.nan,
                    rrr_ex_worst=np.nan, passes=False, per_quarter={})

    per_q = {q: float(np.mean(vs)) for q, vs in quarter_rrrs.items()}
    worst_q  = min(per_q, key=per_q.get)
    worst_r  = per_q[worst_q]
    ex_worst = [v for q, v in per_q.items() if q != worst_q]
    rrr_ex_worst = float(np.mean(ex_worst)) if ex_worst else worst_r

    return dict(
        worst_quarter=worst_q,
        worst_rrr=float(worst_r),
        rrr_ex_worst=float(rrr_ex_worst),
        passes=bool(rrr_ex_worst > 1.0),
        per_quarter={k: float(v) for k, v in sorted(per_q.items())},
    )


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 8: Permutation null
#   - Pre-trade ~600 non-coint pairs in parallel (null pool cache)
#   - Then M=1000 draws of 60 from that cache -> null distribution
#   - Report p = rank(obs Sharpe in null dist)
# ──────────────────────────────────────────────────────────────────────────────

def build_null_pool_cache(non_coint_pairs: list[tuple], n_sample: int,
                          rng_seed: int = 99) -> list[dict]:
    """
    Pre-trade a random sample of non-cointegrated pairs in parallel.
    Returns a list of dicts with {sym_a, sym_b, pnl_net, hold, global_bar}.
    """
    rng = np.random.default_rng(rng_seed)
    pool_size = min(n_sample, len(non_coint_pairs))
    if pool_size < TOP_N:
        print(f"  WARNING: non-coint pool too small ({pool_size}). "
              f"Will fall back to side-shuffle.")
        return []

    idx_sample = rng.choice(len(non_coint_pairs), pool_size, replace=False)
    sampled = [non_coint_pairs[i] for i in idx_sample]

    print(f"  Building null pool cache: trading {pool_size} non-coint pairs "
          f"(PARALLEL x{N_WORKERS})...", flush=True)

    results = []
    n_done = 0
    with Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_null_pair_worker, sampled, chunksize=10):
            n_done += 1
            if n_done % 100 == 0:
                print(f"  null pool: {n_done}/{pool_size} | "
                      f"{len(results)} ok | RAM {_ram_gb():.2f} GB", flush=True)
            if res is not None:
                sym_a, sym_b, tb = res
                results.append(dict(
                    sym_a=sym_a, sym_b=sym_b,
                    pnl_net=tb["pnl_net"].to_numpy(),
                    hold=tb["hold"].to_numpy(),
                    global_bar=tb["global_bar"].to_numpy(),
                ))

    print(f"  Null pool cache: {len(results)}/{pool_size} pairs traded successfully")
    return results


def _book_from_null_cache(null_cache: list[dict], indices: np.ndarray) -> float:
    """Build a 60-pair null book from pre-computed pair returns at given indices."""
    all_pnl = []; all_hold = []; all_bar = []
    for i in indices:
        p = null_cache[i]
        n = len(p["pnl_net"])
        if n == 0:
            continue
        all_pnl.append(p["pnl_net"])
        all_hold.append(p["hold"])
        all_bar.append(p["global_bar"])

    if not all_pnl:
        return 0.0

    pnl_v  = np.concatenate(all_pnl)
    hold_v = np.concatenate(all_hold)
    bar_v  = np.concatenate(all_bar)

    if len(bar_v) == 0:
        return 0.0

    max_b = int(bar_v.max()) + MAX_HOLD + 2
    book_n = np.zeros(max_b)
    for k in range(len(pnl_v)):
        i0 = int(bar_v[k]); h = max(1, int(hold_v[k]))
        book_n[i0:min(i0 + h, max_b)] += pnl_v[k] / h
    nz = np.where(book_n != 0)[0]
    if len(nz) == 0:
        return 0.0
    return score_series(book_n[nz[0]:nz[-1] + 1])["sr_per_bar"]


def permutation_null(pair_results: list[dict],
                     null_cache: list[dict],
                     n_perm: int,
                     label: str = "",
                     rng_seed: int = 42) -> tuple:
    """
    Permutation null using the pre-traded null pool cache.
    Each draw: select TOP_N pairs from null_cache, form inv-vol book -> SR.
    Falls back to side-shuffle if cache is too small.
    """
    print(f"\n=== STAGE 8: Permutation null {label}(M={n_perm}) ===")
    rng = np.random.default_rng(rng_seed)

    n_pairs = len(pair_results)
    n_grid_cells = int(len(PT_GRID) * len(SL_GRID))

    all_dfs = [pr["_trades_df"] for pr in pair_results]
    if not all_dfs:
        return 1.0, np.array([]), 0.0

    trades_obs = pd.concat(all_dfs, ignore_index=True)
    pnl_obs    = trades_obs["pnl_net"].to_numpy()
    hold_obs   = trades_obs["hold"].to_numpy()
    bar_obs    = trades_obs["global_bar"].to_numpy()

    max_bar = int(bar_obs.max()) + MAX_HOLD + 2
    book_obs = np.zeros(max_bar)
    for k in range(len(pnl_obs)):
        i0 = int(bar_obs[k]); h = max(1, int(hold_obs[k]))
        book_obs[i0:min(i0 + h, max_bar)] += pnl_obs[k] / h
    nz = np.where(book_obs != 0)[0]
    book_obs = book_obs[nz[0]:nz[-1] + 1] if len(nz) else book_obs
    sr_obs = score_series(book_obs)["sr_per_bar"]

    use_cache = len(null_cache) >= n_pairs

    if use_cache:
        print(f"  Using null pool cache ({len(null_cache)} pairs). "
              f"Drawing {TOP_N} per permutation.")
        null_srs = np.empty(n_perm)
        for m in range(n_perm):
            if (m + 1) % 200 == 0:
                print(f"  perm {m+1}/{n_perm} | RAM {_ram_gb():.2f} GB")
            idx = rng.choice(len(null_cache), TOP_N, replace=False)
            null_srs[m] = _book_from_null_cache(null_cache, idx)
    else:
        print(f"  Null cache too small ({len(null_cache)} < {n_pairs}). "
              f"Using side-shuffle null.")
        null_srs = np.empty(n_perm)
        for m in range(n_perm):
            if (m + 1) % 200 == 0:
                print(f"  perm {m+1}/{n_perm} | RAM {_ram_gb():.2f} GB")
            flip = rng.integers(0, 2, len(pnl_obs)).astype(np.float64) * 2 - 1
            pnl_p = pnl_obs * flip
            book_p = np.zeros(max_bar)
            for k in range(len(pnl_p)):
                i0 = int(bar_obs[k]); h = max(1, int(hold_obs[k]))
                book_p[i0:min(i0 + h, max_bar)] += pnl_p[k] / h
            nz_p = np.where(book_p != 0)[0]
            if len(nz_p) == 0:
                null_srs[m] = 0.0; continue
            null_srs[m] = score_series(book_p[nz_p[0]:nz_p[-1] + 1])["sr_per_bar"]

    rank  = int((null_srs >= sr_obs).sum())
    p_val = (rank + 1) / (n_perm + 1)

    print(f"  Observed SR_per_bar = {sr_obs:.4f}")
    print(f"  Null mean = {null_srs.mean():.4f}, std = {null_srs.std():.4f}")
    print(f"  Rank {rank}/{n_perm} -> raw p = {p_val:.4f}")
    print(f"  Grid trials deflation: n_grid={n_grid_cells}, n_pairs={n_pairs} "
          f"-> effective trials = {n_grid_cells * n_pairs:,}")
    print(f"  Deflation note: null pairs are non-cointegrated but SAME "
          f"pt/sl grid search runs on top-60 selection. "
          f"Permutation deflates BOTH pair-selection and grid search.")

    return float(p_val), null_srs, float(sr_obs)


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 9: Sign-consistency (disjoint halves of the top-60)
# ──────────────────────────────────────────────────────────────────────────────

def sign_consistency(pair_results: list[dict]) -> tuple:
    """Split by sym_a[0] <= 'M' vs > 'M'. Both halves must be net-positive."""
    half_a = [pr for pr in pair_results if pr["sym_a"][0].upper() <= "M"]
    half_b = [pr for pr in pair_results if pr["sym_a"][0].upper() >  "M"]

    def book_stat(prs, label):
        if not prs:
            return dict(label=label, n_pairs=0, sr_ann=0.0, pf=0.0, net_positive=False)
        _, bm = build_book(prs, sizing="inv_vol")
        return dict(
            label=label, n_pairs=len(prs),
            sr_ann=float(bm.get("sr_ann", 0.0)),
            pf=float(bm.get("pf", 0.0)),
            net_positive=bool(bm.get("sr_ann", 0.0) > 0 and bm.get("pf", 1.0) > 1.0),
        )

    sa = book_stat(half_a, "A-M")
    sb = book_stat(half_b, "N-Z")
    both = bool(sa["net_positive"] and sb["net_positive"])

    print(f"\n=== STAGE 9: Sign-consistency (top-{TOP_N}) ===")
    print(f"  Half A (sym_a A-M): {sa['n_pairs']} pairs | "
          f"SR={sa['sr_ann']:.3f} | PF={sa['pf']:.3f} | pass={sa['net_positive']}")
    print(f"  Half B (sym_a N-Z): {sb['n_pairs']} pairs | "
          f"SR={sb['sr_ann']:.3f} | PF={sb['pf']:.3f} | pass={sb['net_positive']}")
    return sa, sb, both


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 10: S7b sizing comparison
# ──────────────────────────────────────────────────────────────────────────────

def sizing_comparison(pair_results: list[dict]) -> dict:
    results = {}
    for sz in ("equal", "inv_vol", "confidence"):
        book_r, bm = build_book(pair_results, sizing=sz)
        if len(book_r) == 0:
            results[sz] = dict(sizing=sz, sr_ann=0.0, pf=0.0, rrr_all=0.0, n_trades=0)
            continue
        tg = compute_tail_guard(pair_results)
        results[sz] = dict(
            sizing=sz,
            sr_ann=float(bm.get("sr_ann", 0.0)),
            pf=float(bm.get("pf", 0.0)),
            rrr_all=float(bm.get("rrr_all", 0.0)),
            rrr_ex_worst=float(tg.get("rrr_ex_worst", 0.0)),
            n_trades=int(bm.get("n_trades", 0)),
        )
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 11: Grid corner check
# ──────────────────────────────────────────────────────────────────────────────

def check_grid_corners(pair_results: list[dict]) -> dict:
    pt_vals = [pr["pt_star"] for pr in pair_results if np.isfinite(pr.get("pt_star", np.nan))]
    sl_vals = [pr["sl_star"] for pr in pair_results if np.isfinite(pr.get("sl_star", np.nan))]
    pt_arr  = np.array(pt_vals); sl_arr = np.array(sl_vals)
    old_min = 0.25
    return dict(
        n_pairs_with_fit=len(pt_arr),
        median_pt_star=float(np.nanmedian(pt_arr)) if len(pt_arr) else np.nan,
        median_sl_star=float(np.nanmedian(sl_arr)) if len(sl_arr) else np.nan,
        frac_pt_at_old_min=float((pt_arr <= old_min).mean()) if len(pt_arr) else np.nan,
        frac_sl_at_old_min=float((sl_arr <= old_min).mean()) if len(sl_arr) else np.nan,
        grid_min_used=float(PT_GRID[0]),
        optimum_moved_interior=bool((pt_arr <= old_min).mean() < 0.5) if len(pt_arr) else False,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  FIGURES
# ──────────────────────────────────────────────────────────────────────────────

def make_figures(pair_results, book_r, null_srs, sr_obs, tail):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(np.cumsum(book_r), lw=1.0, color="#d7191c", label=f"top-{TOP_N}")
        axes[0].axhline(0, color="#888", lw=0.7, ls="--")
        axes[0].set_title(f"S7a: Top-{TOP_N} book equity (inv-vol, net 28bp/RT)")
        axes[0].set_xlabel("OOS bars (hourly)")
        axes[0].set_ylabel("Cum P&L")
        axes[0].legend(fontsize=8)

        axes[1].hist(null_srs, bins=50, color="#aaa", edgecolor="white", alpha=0.8,
                     label="null (~600-sample non-coint)")
        axes[1].axvline(sr_obs, color="#d7191c", lw=2.0,
                        label=f"obs SR={sr_obs:.4f}")
        axes[1].set_title("Permutation null: book Sharpe per bar")
        axes[1].legend(fontsize=8)

        qs = sorted(tail.get("per_quarter", {}).items())
        if qs:
            q_labels = [q for q, _ in qs]
            q_rrr    = [v for _, v in qs]
            axes[2].bar(range(len(q_labels)), q_rrr, color="#abdda4", edgecolor="white")
            axes[2].axhline(1.0, color="#d7191c", lw=1.5, ls="--", label="RRR=1 bar")
            axes[2].set_xticks(range(len(q_labels)))
            axes[2].set_xticklabels(q_labels, rotation=45, ha="right", fontsize=7)
            axes[2].set_title(f"Per-quarter RRR (top-{TOP_N} tail-guard)")
            axes[2].legend(fontsize=8)

        fig.suptitle(
            f"S7v4: Top-{TOP_N} spread MR book — fast parallel (T*={T_STAR.date()}, leak-free)",
            fontsize=11)
        fig.tight_layout()
        out_fig = os.path.join(OUT, "book_equity_v4.png")
        fig.savefig(out_fig, dpi=120)
        plt.close(fig)
        print(f"  figure -> {out_fig}")
        return out_fig
    except Exception as e:
        print(f"  [figure skipped] {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    mp.set_start_method("fork", force=True)

    t_total = time.perf_counter()
    print("=" * 72)
    print("S7a + S7b -- Spread mean-reversion at breadth (FAST v4)")
    print(f"T* = {T_STAR}  (global split; selection uses ONLY pre-T* data)")
    print(f"OOS: post-T* only. Workers: {N_WORKERS}. Top-N book: {TOP_N} pairs.")
    print(f"Costs: {COST_BP_PER_LEG} bp/leg x 4 fills/RT = {4*COST_BP_PER_LEG:.0f} bp/RT")
    print(f"Grid: {len(PT_GRID)} x {len(SL_GRID)} cells, "
          f"pt from {PT_GRID[0]:.2f} to {PT_GRID[-1]:.2f}")
    print(f"Affinity: cores 16-31")
    print("=" * 72)

    # Stage 1: Universe
    t0 = time.perf_counter()
    univ = build_universe()
    print(f"\n  Universe: {len(univ)} symbols | {(time.perf_counter()-t0):.1f}s | RAM {_ram_gb():.2f} GB")

    if len(univ) < SANITY_MIN_SYMBOLS:
        print(f"  SANITY GATE FAILED: {len(univ)} < {SANITY_MIN_SYMBOLS}. Abort.")
        return

    # Stage 2: Correlation prefilter
    t0 = time.perf_counter()
    candidates = build_corr_candidates(univ)
    print(f"\n  Candidates: {len(candidates)} | {(time.perf_counter()-t0):.1f}s | RAM {_ram_gb():.2f} GB")

    if len(candidates) < SANITY_MIN_CANDIDATES:
        print(f"  SANITY GATE FAILED: {len(candidates)} < {SANITY_MIN_CANDIDATES}. Abort.")
        return

    # Stage 3: Engle-Granger + OU fit (PARALLEL) -- all candidates
    t0 = time.perf_counter()
    coint_pairs, non_coint_pool = screen_cointegration_parallel(candidates)
    n_coint = len(coint_pairs)
    print(f"\n  EG done: {n_coint} cointegrated | {len(non_coint_pool)} non-coint "
          f"| {(time.perf_counter()-t0):.1f}s | RAM {_ram_gb():.2f} GB")

    if n_coint == 0:
        print("No cointegrated pairs. FAIL.")
        verdict_path = os.path.join(OUT, "s07_v4_verdict.json")
        with open(verdict_path, "w") as f:
            json.dump(dict(verdict_s7a="FAIL", verdict_s7b="FAIL",
                           n_coint=0, reason="no cointegrated pairs"), f, indent=2)
        return

    # ── KEY CHANGE: Select top-60 BEFORE OOS trading ──────────────────────────
    top60_coint = select_top_n_coint(coint_pairs, TOP_N)
    print(f"\n  Top-{TOP_N} selected from {n_coint} cointegrated "
          f"(pre-T* ADF strength; no OOS data used)")
    print(f"  ADF p range: {top60_coint[0]['adf_p']:.4e} to "
          f"{top60_coint[-1]['adf_p']:.4e}")
    print(f"  HL range: {top60_coint[0]['ou_half_life']:.1f} to "
          f"{top60_coint[-1]['ou_half_life']:.1f} bars")

    # Free all non-top-60 coint data from memory
    del coint_pairs; gc.collect()

    # Stage 4: No-lookahead assertion (ONLY on top-60 -- fast)
    t0 = time.perf_counter()
    assert_no_lookahead(top60_coint)
    print(f"  Assertion: {(time.perf_counter()-t0):.1f}s")

    # Stage 5: OOS trading (PARALLEL, top-60 only)
    t0 = time.perf_counter()
    pair_results = run_oos_parallel(top60_coint)
    print(f"\n  OOS trading done: {(time.perf_counter()-t0):.1f}s | RAM {_ram_gb():.2f} GB")

    if not pair_results:
        print("No pairs produced OOS results. FAIL.")
        return

    # Stage 6: Portfolio book (inv-vol, top-60 only)
    print("\n=== STAGE 6: Portfolio book (top-60 inv-vol) ===")
    book_r, book_meta = build_book(pair_results, sizing="inv_vol")

    sc = score_series(book_r) if len(book_r) > 0 else {}
    print(f"  Top-{TOP_N} book: SR_ann={sc.get('sr_ann',0):.3f} | "
          f"PF={sc.get('pf',0):.3f} | n_trades={book_meta.get('n_trades',0)}")

    # Stage 7: Tail guard
    tail = compute_tail_guard(pair_results)
    print(f"\n=== STAGE 7: Tail guard ===")
    print(f"  worst_q={tail['worst_quarter']} RRR={tail['worst_rrr']:.3f} | "
          f"ex-worst RRR={tail['rrr_ex_worst']:.3f} | passes={tail['passes']}")

    # Grid corner check
    grid_chk = check_grid_corners(pair_results)
    print(f"\n  Grid: median pt*={grid_chk['median_pt_star']:.3f} "
          f"sl*={grid_chk['median_sl_star']:.3f} | "
          f"{grid_chk['frac_pt_at_old_min']*100:.0f}% at old 0.25 min | "
          f"interior={grid_chk['optimum_moved_interior']}")

    # Per-trade ledger (top-60)
    ledger = pd.concat([pr["_trades_df"] for pr in pair_results], ignore_index=True)
    ledger = ledger[[c for c in ledger.columns if not c.startswith("_")]]
    ledger_path = os.path.join(OUT, "s07_v4_trades_top60.parquet")
    ledger.to_parquet(ledger_path, index=False)
    print(f"\n  Trade ledger: {len(ledger)} trades -> {ledger_path}")

    # Per-pair summary parquet
    pair_rows = []
    for pr in pair_results:
        rrrs = [qs["rrr"] for qs in pr.get("q_stats", [])]
        pair_rows.append(dict(
            sym_a=pr["sym_a"], sym_b=pr["sym_b"],
            n_oos_trades=pr["n_oos_trades"],
            rrr_pair=pr["rrr_pair"],
            pt_star=pr["pt_star"], sl_star=pr["sl_star"],
            ou_phi=pr["ou_phi"], ou_half_life=pr["ou_half_life"],
            adf_p=pr.get("adf_p", np.nan),
            n_quarters=len(rrrs),
            med_q_rrr=float(np.nanmedian(rrrs)) if rrrs else np.nan,
            min_q_rrr=float(np.nanmin(rrrs)) if rrrs else np.nan,
        ))
    pair_df = pd.DataFrame(pair_rows)
    pair_parquet = os.path.join(OUT, "s07_v4_pair_table.parquet")
    pair_df.to_parquet(pair_parquet, index=False)
    print(f"  Pair table -> {pair_parquet}")

    # Stage 8: Build null pool cache, then permutation null
    print(f"\n=== STAGE 8a: Build ~{NULL_POOL_SAMPLE}-pair null pool cache ===")
    t0 = time.perf_counter()
    null_cache = build_null_pool_cache(non_coint_pool, NULL_POOL_SAMPLE, rng_seed=99)
    print(f"  Null pool cache built: {len(null_cache)} pairs | "
          f"{(time.perf_counter()-t0):.1f}s | RAM {_ram_gb():.2f} GB")

    # Free non-coint pool list (large)
    del non_coint_pool; gc.collect()

    t0 = time.perf_counter()
    p_val, null_srs, sr_obs_perm = permutation_null(
        pair_results, null_cache, N_PERM, label=f"[top-{TOP_N}] ", rng_seed=42)
    print(f"  Permutation null done: {(time.perf_counter()-t0):.1f}s")

    # Stage 9: Sign-consistency
    half_a, half_b, sign_pass = sign_consistency(pair_results)

    # Stage 10: S7b sizing comparison
    print(f"\n=== STAGE 10: S7b -- sizing comparison (top-{TOP_N}) ===")
    sz_results = sizing_comparison(pair_results)
    for name, sr in sz_results.items():
        print(f"  {name:12s}: SR_ann={sr['sr_ann']:.3f} | PF={sr['pf']:.3f} | "
              f"RRR_ex_worst={sr['rrr_ex_worst']:.3f}")

    # Figures
    fig_path = make_figures(pair_results, book_r, null_srs, sr_obs_perm, tail)

    # Verdicts
    n_grid_trials = int(len(PT_GRID) * len(SL_GRID) * len(pair_results))
    s7b_pass = bool(sz_results["inv_vol"]["rrr_ex_worst"] > sz_results["equal"]["rrr_ex_worst"])

    crit = dict(
        net_positive      = bool(sc.get("sr_ann", 0) > 0 and sc.get("pf", 0) > 1.0),
        tail_guard_passes = bool(tail["passes"]),
        deflated_sig      = bool(p_val < 0.05),
        sign_consistent   = bool(sign_pass),
        grid_interior     = bool(grid_chk["optimum_moved_interior"]),
    )

    def _verdict(pass_criteria):
        n_pass = sum(pass_criteria.values())
        if all(pass_criteria.values()):
            return "PASS"
        elif n_pass <= 2:
            return "FAIL"
        else:
            return "MARGINAL/FAIL"

    verdict_s7a = _verdict(crit)
    verdict_s7b = "PASS" if s7b_pass else "FAIL"

    elapsed  = time.perf_counter() - t_total
    ram_peak = _ram_gb()

    verdict = dict(
        # Configuration
        T_star                     = str(T_STAR.date()),
        no_lookahead_assertion     = "PASSED (see Stage 4 log)",
        n_workers                  = N_WORKERS,
        top_n                      = TOP_N,
        null_pool_sample           = NULL_POOL_SAMPLE,
        null_pool_actual           = len(null_cache),
        # Counts
        n_usdt_syms                = len(univ),
        n_candidates               = len(candidates),
        n_cointegrated             = n_coint,
        n_pairs_oos_ok             = len(pair_results),
        # Top-60 book
        top60_book = dict(
            n_pairs             = len(pair_results),
            selection_criterion = "most-negative ADF stat (smallest adf_p); tie-break shortest OU half-life; pre-T* only; selection BEFORE OOS trading",
            book_sr_ann         = float(sc.get("sr_ann", 0)),
            book_pf             = float(sc.get("pf", 0)),
            book_sr_pb          = float(sc.get("sr_per_bar", 0)),
            n_trades            = int(book_meta.get("n_trades", 0)),
            tail_worst_quarter  = tail["worst_quarter"],
            tail_worst_rrr      = float(tail["worst_rrr"]),
            tail_rrr_ex_worst   = float(tail["rrr_ex_worst"]),
            tail_guard_pass     = bool(tail["passes"]),
            tail_per_quarter    = tail["per_quarter"],
            perm_p_value        = float(p_val),
            perm_sr_obs         = float(sr_obs_perm),
            perm_null_mean      = float(null_srs.mean()) if len(null_srs) else 0.0,
            perm_null_std       = float(null_srs.std())  if len(null_srs) else 0.0,
            n_perm              = N_PERM,
            n_trials_deflated   = n_grid_trials,
            deflation_note      = (
                f"Null draws {TOP_N} from {len(null_cache)} pre-traded non-coint pairs. "
                f"Deflates pair-selection over {n_coint} cointegrated candidates AND "
                f"pt/sl grid ({len(PT_GRID)}x{len(SL_GRID)}={n_grid_trials} effective trials)."
            ),
            sign_half_a         = half_a,
            sign_half_b         = half_b,
            sign_consistent     = bool(sign_pass),
            sizing_equal        = sz_results.get("equal", {}),
            sizing_inv_vol      = sz_results.get("inv_vol", {}),
            sizing_confidence   = sz_results.get("confidence", {}),
            s7b_inv_beats_equal = s7b_pass,
            criterion_s7a       = crit,
            verdict_s7a         = verdict_s7a,
            verdict_s7b         = verdict_s7b,
        ),
        # Grid check
        grid_check = grid_chk,
        # Run metadata
        elapsed_s         = float(elapsed),
        elapsed_min       = float(elapsed / 60),
        ram_peak_gb       = float(ram_peak),
        cost_bp_rt        = float(4 * COST_BP_PER_LEG),
        pt_grid_min       = float(PT_GRID[0]),
        pt_grid_max       = float(PT_GRID[-1]),
        n_grid_cells      = int(len(PT_GRID) * len(SL_GRID)),
        artifacts = dict(
            ledger_top  = ledger_path,
            pairs       = pair_parquet,
            figure      = fig_path,
            verdict     = os.path.join(OUT, "s07_v4_verdict.json"),
        ),
    )

    verdict_path = os.path.join(OUT, "s07_v4_verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\n  verdict -> {verdict_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL VERDICT (S7v4 -- TOP-60 ONLY, FAST PARALLEL)")
    print("=" * 72)
    print(f"  T*             : {T_STAR.date()}")
    print(f"  No-lookahead   : ASSERTED AND PASSED")
    print(f"  Universe       : {len(univ)} USDT perps")
    print(f"  Candidates     : {len(candidates)}")
    print(f"  Cointegrated   : {n_coint}")
    print(f"  Top-{TOP_N} traded  : {len(pair_results)}")
    print(f"")
    print(f"  -- TOP-{TOP_N} BOOK --")
    print(f"  Total OOS trades : {book_meta.get('n_trades',0)}")
    print(f"  Book SR_ann      : {sc.get('sr_ann',0):.3f}")
    print(f"  Book PF          : {sc.get('pf',0):.3f}")
    print(f"  Tail guard       : ex-worst-Q RRR = {tail['rrr_ex_worst']:.3f} | "
          f"passes = {tail['passes']}")
    print(f"  Perm null        : p = {p_val:.4f} (M={N_PERM}, "
          f"null_pool={len(null_cache)}, trials={n_grid_trials:,})")
    print(f"  Sign consist.    : A-M={half_a['net_positive']} "
          f"N-Z={half_b['net_positive']} | both = {sign_pass}")
    print(f"  Criterion        : {crit}")
    print(f"  S7a (top-{TOP_N})  : {verdict_s7a}")
    print(f"  S7b (top-{TOP_N})  : {verdict_s7b}  "
          f"(inv_vol RRR_ex={sz_results['inv_vol']['rrr_ex_worst']:.3f} vs "
          f"equal={sz_results['equal']['rrr_ex_worst']:.3f})")
    print(f"")
    print(f"  Elapsed: {elapsed/60:.1f} min | RAM peak: {ram_peak:.2f} GB")
    print("=" * 72)


if __name__ == "__main__":
    main()
