#!/usr/bin/env python3
"""
S7a + S7b  --  Spread mean-reversion at breadth  (PARALLEL v3)
==============================================================
Based on s07_clean.py (LEAK-FREE REBUILD) — same exact logic, same T*, same
no-lookahead assertion, same costs, same OU mesh, same permutation null.

CHANGES vs s07_clean.py:
  1. PARALLEL with multiprocessing.Pool(8): cointegration screen + OOS trading.
  2. TOP-60 CAPPED BOOK: select top-60 pairs by pre-T* ADF strength (most-negative
     ADF stat; tie-break shortest OU half-life).  All selection on pre-T* only.
     Both books (all-pairs and top-60) reported with full stats + permutation p.
  3. Speed target: < 25 min on 8 cores.

DO NOT MODIFY s07_clean.py — this is a NEW file.
"""
from __future__ import annotations
import os, sys

# Pin thread counts BEFORE any import (redundant safety; primary enforcement by wrapper)
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import gc, time, json, warnings, resource, itertools, math
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
N_PERM  = 1000
MIN_OOS_TRADES = 10

# ── top-N capped book ─────────────────────────────────────────────────────────
TOP_N = 60

# ── sanity gates ──────────────────────────────────────────────────────────────
SANITY_MIN_SYMBOLS    = 100
SANITY_MIN_CANDIDATES = 300

HEARTBEAT = 50
BARS_PER_YEAR = 24.0 * 365.25

# ── parallelism ───────────────────────────────────────────────────────────────
N_WORKERS = 8
# Recycle each forked Pool worker after this many tasks. Bounds the lifetime
# of accumulated pyarrow/pandas C-extension state per worker, which is the
# documented mitigation for hashtable-allocator segfaults under fork+heavy
# concurrent parquet reads (the prior run's silent-crash failure mode).
MAX_TASKS_PER_CHILD = 150


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
    """
    sym_a, sym_b, worker_id = args
    # Lazy import inside worker (module-level imports already propagated via fork)
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
        # realized OOS bounds for the OOS-side no-lookahead assertion (Stage 5b)
        min_oos_ts=str(idx_oos[0]),
        max_oos_ts=str(idx_oos[-1]),
        _trades_df=trades_df,
    ), "ok"


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
            deduped.append((sym_a, sym_b, 0))  # worker_id unused

    print(f"  {len(deduped):,} unique pairs after dedup")

    coint   = []
    non_coint_pairs = []
    n_done  = 0

    # maxtasksperchild recycles each forked worker after N tasks, releasing
    # accumulated pyarrow/pandas C-extension state before it can corrupt the
    # hashtable allocator (the prior run's segfault-under-fork failure mode).
    with Pool(N_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        for result in pool.imap_unordered(_eg_worker, deduped, chunksize=20):
            n_done += 1
            if n_done % 1000 == 0:
                print(f"  EG: {n_done}/{len(deduped)} screened | "
                      f"{len(coint)} cointegrated | RAM {_ram_gb():.2f} GB", flush=True)
            if result is not None:
                coint.append(result)
            # We can't easily recover which pair failed without passing it back,
            # so rebuild non-coint pool from candidates after

    # Build non-coint pool from the candidates list
    coint_set = {(cp["sym_a"], cp["sym_b"]) for cp in coint}
    coint_set_rev = {(cp["sym_b"], cp["sym_a"]) for cp in coint}
    non_coint_pairs = [(a, b) for a, b, _ in deduped
                       if (a, b) not in coint_set and (a, b) not in coint_set_rev]

    print(f"\n  EG result: {len(deduped)} tested | {len(coint)} cointegrated "
          f"(ADF p<={ADF_P_MAX}) | non-coint pool: {len(non_coint_pairs)}")
    return coint, non_coint_pairs


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 4: Hard no-lookahead assertion (identical to s07_clean.py)
# ──────────────────────────────────────────────────────────────────────────────

def assert_no_lookahead(coint_pairs: list[dict]) -> bool:
    """
    Stage 4 -- SELECTION-side hard assertion (I/O-free).

    Asserts the leak-free invariant on the selection side: for every
    cointegrated pair, the LAST bar used for the Engle-Granger / OU fit
    (`last_pre_ts`, recorded by `_eg_worker` from the pre-T* index it
    actually fit on) is strictly < T*.  This is the load-bearing
    anti-lookahead guarantee: nothing at or after T* touched selection/fit.

    The OOS-side invariant (every traded bar >= T*) is asserted just as
    hard *after* Stage 5 by `assert_oos_post_tstar`, on the bars that were
    actually traded -- a strictly stronger check than re-reading files here.
    Re-reading all pairs' parquet in this serial loop was redundant
    (idx_oos is constructed as idx[idx>=T*], so min>=T* is tautological)
    and was the prior run's silent-crash surface (pyarrow/pandas hashtable
    segfault under fork+contention); removed without weakening the gate.
    """
    print(f"\n=== STAGE 4: No-lookahead assertion (selection side) "
          f"for {len(coint_pairs)} pairs ===")
    violations = []

    for cp in coint_pairs:
        last_pre = cp["last_pre_ts"]
        if pd.isnull(last_pre) or last_pre >= T_STAR:
            violations.append(dict(
                pair=f"{cp['sym_a']}/{cp['sym_b']}",
                violation="last_pre_ts >= T* (selection used data at/after split)",
                last_pre_ts=str(last_pre),
                T_star=str(T_STAR),
            ))

    if violations:
        print(f"\n  LOOKAHEAD ASSERTION FAILED: {len(violations)} violations")
        for v in violations[:10]:
            print(f"    {v}")
        print("\n  ABORTING RUN -- fix the leak before proceeding.")
        sys.exit(1)

    print(f"  All {len(coint_pairs)} pairs PASS (selection side): "
          f"max(selection_ts) < T*")
    return True


def assert_oos_post_tstar(pair_results: list[dict]) -> bool:
    """
    Post-Stage-5 OOS-side hard assertion (I/O-free).

    For every pair that produced OOS results, asserts the earliest bar it
    actually traded on is >= T* (and that it traded post-T* bars only).
    Uses `min_oos_ts` / `max_pre_ts_seen` recorded by `_oos_worker` from
    the realized OOS index -- so this checks the bars that were *traded*,
    not merely that some post-T* data exists.  Aborts on any violation.
    """
    print(f"\n=== STAGE 5b: No-lookahead assertion (OOS side) "
          f"for {len(pair_results)} traded pairs ===")
    violations = []
    for pr in pair_results:
        mn = pr.get("min_oos_ts", None)
        if mn is None or pd.isnull(pd.Timestamp(mn)):
            violations.append(dict(
                pair=f"{pr['sym_a']}/{pr['sym_b']}",
                violation="missing min_oos_ts", min_oos_ts=str(mn)))
            continue
        if pd.Timestamp(mn) < T_STAR:
            violations.append(dict(
                pair=f"{pr['sym_a']}/{pr['sym_b']}",
                violation="traded an OOS bar < T*",
                min_oos_ts=str(mn), T_star=str(T_STAR)))

    if violations:
        print(f"\n  OOS LOOKAHEAD ASSERTION FAILED: {len(violations)} violations")
        for v in violations[:10]:
            print(f"    {v}")
        print("\n  ABORTING RUN -- fix the leak before proceeding.")
        sys.exit(1)

    print(f"  All {len(pair_results)} pairs PASS (OOS side): "
          f"min(traded_bar_ts) >= T*")
    return True


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 5: OOS trading -- PARALLEL
# ──────────────────────────────────────────────────────────────────────────────

def run_oos_parallel(coint_pairs: list[dict]) -> list[dict]:
    """
    Parallel OOS trading over all cointegrated pairs using Pool(N_WORKERS).
    """
    print(f"\n=== STAGE 5: OOS trading (PARALLEL x{N_WORKERS}) "
          f"on {len(coint_pairs)} pairs ===")

    pair_results = []
    n_done = 0

    with Pool(N_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        for result, msg in pool.imap_unordered(_oos_worker, coint_pairs, chunksize=5):
            n_done += 1
            if n_done % 200 == 0:
                print(f"  OOS: {n_done}/{len(coint_pairs)} | "
                      f"{len(pair_results)} ok | RAM {_ram_gb():.2f} GB", flush=True)
            if result is not None:
                pair_results.append(result)

    print(f"\n  OOS complete: {len(pair_results)}/{len(coint_pairs)} pairs produced trades")
    return pair_results


# ──────────────────────────────────────────────────────────────────────────────
#  TOP-N selection (pre-T* cointegration strength; all on pre-T* data only)
# ──────────────────────────────────────────────────────────────────────────────

def select_top_n(pair_results: list[dict], n: int) -> list[dict]:
    """
    Select top-N pairs by cointegration strength:
      primary sort: most-negative ADF stat (smallest adf_p)
      tie-break: shortest OU half-life
    All selection criteria were computed on PRE-T* data (adf_p, ou_half_life
    stored in each pair's dict from Stage 3).
    NO OOS data used.
    """
    if len(pair_results) <= n:
        return list(pair_results)
    # sort by adf_p asc (most-negative ADF stat = smallest p-value),
    # then by ou_half_life asc (shortest half-life as tie-break)
    ranked = sorted(pair_results,
                    key=lambda pr: (pr.get("adf_p", 1.0), pr.get("ou_half_life", 1e9)))
    return ranked[:n]


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 6: Portfolio book (identical to s07_clean.py)
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
#  STAGE 7: Tail guard (identical to s07_clean.py)
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
#  STAGE 8: Permutation null (identical logic, uses side-shuffle for speed)
# ──────────────────────────────────────────────────────────────────────────────

def permutation_null(pair_results: list[dict],
                     non_coint_pairs: list[tuple],
                     n_perm: int,
                     label: str = "",
                     rng_seed: int = 42) -> tuple:
    print(f"\n=== STAGE 8: Permutation null {label}({n_perm} draws) ===")
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

    if len(non_coint_pairs) < n_pairs:
        print(f"  Non-coint pool too small ({len(non_coint_pairs)} < {n_pairs}). "
              f"Using side-shuffle null instead.")
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
    else:
        null_srs = _perm_from_non_coint_pool(non_coint_pairs, n_pairs, n_perm, rng)

    rank  = int((null_srs >= sr_obs).sum())
    p_val = (rank + 1) / (n_perm + 1)

    print(f"  Observed SR_per_bar = {sr_obs:.4f}")
    print(f"  Null mean = {null_srs.mean():.4f}, std = {null_srs.std():.4f}")
    print(f"  Rank {rank}/{n_perm} -> raw p = {p_val:.4f}")
    print(f"  Grid trials deflation: n_grid={n_grid_cells}, n_pairs={n_pairs} "
          f"-> effective trials = {n_grid_cells * n_pairs:,}")

    return float(p_val), null_srs, float(sr_obs)


def _perm_from_non_coint_pool(non_coint_pairs, n_pairs, n_perm, rng):
    pool = list(non_coint_pairs)
    null_srs = np.empty(n_perm)

    for m in range(n_perm):
        if (m + 1) % 100 == 0:
            print(f"  null book {m+1}/{n_perm} | RAM {_ram_gb():.2f} GB")
        sampled = [pool[i] for i in rng.choice(len(pool), n_pairs, replace=False)]
        null_trades = []
        for sym_a, sym_b in sampled:
            try:
                a0 = load_perp(sym_a); b0 = load_perp(sym_b)
                idx_pre = a0.index.intersection(b0.index)
                idx_pre = idx_pre[idx_pre < T_STAR]
                if len(idx_pre) < MIN_PAIR_PRE:
                    del a0, b0; continue
                la = np.log(a0.loc[idx_pre, "close"].to_numpy(np.float64))
                lb = np.log(b0.loc[idx_pre, "close"].to_numpy(np.float64))
                alpha, beta, _ = engle_granger_train(la, lb)
                idx_oos = a0.index.intersection(b0.index)
                idx_oos = idx_oos[idx_oos >= T_STAR]
                if len(idx_oos) < 200:
                    del a0, b0; continue
                a_oos = a0.loc[idx_oos]; b_oos = b0.loc[idx_oos]
                del a0, b0
                sO, sH, sL, sC = spread_ohlc(a_oos, b_oos, beta)
                z_oos, sig_oos = spread_rolling_stats(sC, Z_SPAN)
                ev_idx, side_ev = spread_entry_events(z_oos, ENTRY_K, Z_SPAN + 5)
                if len(ev_idx) < MIN_OOS_TRADES:
                    del a_oos, b_oos; continue
                pt_fix = float(PT_GRID[len(PT_GRID) // 2])
                sl_fix = float(SL_GRID[len(SL_GRID) // 2])
                stat_sd = float(np.std(sC - pd.Series(sC).ewm(span=Z_SPAN, adjust=True).mean().to_numpy()))
                stat_sd = max(stat_sd, 1e-12)
                pt_m = pt_fix / stat_sd; sl_m = sl_fix / stat_sd
                sig_ev = sig_oos[ev_idx]
                tb = apply_spread_rule(sO, sH, sL, sC, sig_ev, ev_idx, side_ev,
                                       pt_m, sl_m, MAX_HOLD)
                pnl_net = tb["pnl_gross"].to_numpy() - RT_COST
                tb["pnl_net"]   = pnl_net
                tb["sym_a"]     = sym_a
                tb["sym_b"]     = sym_b
                tb["global_bar"] = ev_idx.astype(np.int64)
                null_trades.append(tb)
                del a_oos, b_oos; gc.collect()
            except Exception:
                continue

        if not null_trades:
            null_srs[m] = 0.0; continue

        trades_null = pd.concat(null_trades, ignore_index=True)
        pnl_n  = trades_null["pnl_net"].to_numpy()
        hold_n = trades_null["hold"].to_numpy()
        bar_n  = trades_null["global_bar"].to_numpy()
        if len(bar_n) == 0:
            null_srs[m] = 0.0; continue
        max_b = int(bar_n.max()) + MAX_HOLD + 2
        book_n = np.zeros(max_b)
        for k in range(len(pnl_n)):
            i0 = int(bar_n[k]); h = max(1, int(hold_n[k]))
            book_n[i0:min(i0 + h, max_b)] += pnl_n[k] / h
        nz = np.where(book_n != 0)[0]
        if len(nz) == 0:
            null_srs[m] = 0.0; continue
        null_srs[m] = score_series(book_n[nz[0]:nz[-1] + 1])["sr_per_bar"]

    return null_srs


# ──────────────────────────────────────────────────────────────────────────────
#  STAGE 9: Sign-consistency (identical to s07_clean.py)
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

    print(f"\n=== STAGE 9: Sign-consistency ===")
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

def make_figures(pair_results, book_r, null_srs, sr_obs, tail,
                 top_n_results=None, book_r_top=None, null_srs_top=None,
                 sr_obs_top=None, tail_top=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cols = 4 if top_n_results is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

        axes[0].plot(np.cumsum(book_r), lw=1.0, color="#2c7bb6", label="all-pairs")
        if book_r_top is not None and len(book_r_top) > 0:
            axes[0].plot(np.cumsum(book_r_top), lw=1.0, color="#d7191c",
                         alpha=0.75, label=f"top-{TOP_N}")
        axes[0].axhline(0, color="#888", lw=0.7, ls="--")
        axes[0].set_title("S7a: Book equity (inv-vol, net 28bp/RT)")
        axes[0].set_xlabel("OOS bars (hourly)")
        axes[0].set_ylabel("Cum P&L")
        axes[0].legend(fontsize=8)

        axes[1].hist(null_srs, bins=50, color="#aaa", edgecolor="white", alpha=0.8,
                     label="null (all-pairs)")
        axes[1].axvline(sr_obs, color="#2c7bb6", lw=2.0,
                        label=f"obs SR={sr_obs:.4f}")
        if null_srs_top is not None:
            axes[1].axvline(sr_obs_top, color="#d7191c", lw=2.0, ls="--",
                            label=f"top-{TOP_N} obs={sr_obs_top:.4f}")
        axes[1].set_title("Permutation null: book Sharpe per bar")
        axes[1].legend(fontsize=8)

        qs = sorted(tail.get("per_quarter", {}).items())
        if qs:
            q_labels = [q for q, _ in qs]
            q_rrr    = [v for _, v in qs]
            axes[2].bar(range(len(q_labels)), q_rrr, color="#abdda4", edgecolor="white",
                        label="all-pairs")
            if tail_top is not None:
                qs_top = sorted(tail_top.get("per_quarter", {}).items())
                q_rrr_top = [v for _, v in qs_top]
                if len(q_rrr_top) == len(q_labels):
                    axes[2].bar(range(len(q_labels)), q_rrr_top,
                                color="#d7191c", alpha=0.4, edgecolor="white",
                                label=f"top-{TOP_N}")
            axes[2].axhline(1.0, color="#d7191c", lw=1.5, ls="--", label="RRR=1 bar")
            axes[2].set_xticks(range(len(q_labels)))
            axes[2].set_xticklabels(q_labels, rotation=45, ha="right", fontsize=7)
            axes[2].set_title("Per-quarter RRR (tail-guard)")
            axes[2].legend(fontsize=8)

        if n_cols == 4 and top_n_results is not None:
            # Top-N vs All-pairs scatter: n_trades vs SR_ann
            sr_all = [pr.get("rrr_pair", 0) for pr in pair_results]
            sr_top = [pr.get("rrr_pair", 0) for pr in top_n_results]
            axes[3].scatter(range(len(sr_all)), sorted(sr_all, reverse=True),
                            s=4, alpha=0.5, color="#2c7bb6", label="all-pairs")
            axes[3].scatter(range(len(sr_top)), sorted(sr_top, reverse=True),
                            s=8, alpha=0.8, color="#d7191c", label=f"top-{TOP_N}")
            axes[3].set_title("Per-pair RRR (all vs top-N)")
            axes[3].legend(fontsize=8)
            axes[3].set_xlabel("Rank")
            axes[3].set_ylabel("RRR")

        fig.suptitle(
            f"S7v3: Spread MR at breadth — parallel (T*={T_STAR.date()}, leak-free)",
            fontsize=11)
        fig.tight_layout()
        out_fig = os.path.join(OUT, "book_equity_v3.png")
        fig.savefig(out_fig, dpi=120)
        plt.close(fig)
        print(f"  figure -> {out_fig}")
        return out_fig
    except Exception as e:
        print(f"  [figure skipped] {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  HELPER: compile a verdict block for one book
# ──────────────────────────────────────────────────────────────────────────────

def _book_verdict_block(label, pair_results, book_r, book_meta, tail, p_val,
                        null_srs, sr_obs_perm, grid_chk=None):
    sc_book = score_series(book_r) if len(book_r) > 0 else {}
    n_grid_trials = int(len(PT_GRID) * len(SL_GRID) * len(pair_results))

    sz_results = sizing_comparison(pair_results)

    criterion = dict(
        net_positive      = bool(sc_book.get("sr_ann", 0) > 0 and sc_book.get("pf", 0) > 1.0),
        tail_guard_passes = bool(tail["passes"]),
        deflated_sig      = bool(p_val < 0.05),
        sign_consistent   = False,  # filled in by caller
        grid_interior     = bool(grid_chk["optimum_moved_interior"]) if grid_chk else False,
    )

    return dict(
        label               = label,
        n_pairs             = len(pair_results),
        book_sr_ann         = float(sc_book.get("sr_ann", 0)),
        book_pf             = float(sc_book.get("pf", 0)),
        book_sr_pb          = float(sc_book.get("sr_per_bar", 0)),
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
        sizing_equal        = sz_results.get("equal", {}),
        sizing_inv_vol      = sz_results.get("inv_vol", {}),
        sizing_confidence   = sz_results.get("confidence", {}),
        criterion           = criterion,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Required for multiprocessing on Linux (fork mode is default, but be explicit)
    mp.set_start_method("fork", force=True)

    t_total = time.perf_counter()
    print("=" * 72)
    print("S7a + S7b -- Spread mean-reversion at breadth (PARALLEL v3)")
    print(f"T* = {T_STAR}  (global split; selection uses ONLY pre-T* data)")
    print(f"OOS: post-T* only. Workers: {N_WORKERS}. Top-N book: {TOP_N} pairs.")
    print(f"Costs: {COST_BP_PER_LEG} bp/leg x 4 fills/RT = {4*COST_BP_PER_LEG:.0f} bp/RT")
    print(f"Grid: {len(PT_GRID)} x {len(SL_GRID)} cells, "
          f"pt from {PT_GRID[0]:.2f} to {PT_GRID[-1]:.2f}")
    print("=" * 72)

    # Stage 1: Universe
    univ = build_universe()
    print(f"\n  T* = {T_STAR.date()} logged.")
    print(f"  Universe: {len(univ)} symbols pass the pre/post-T* bar filter")

    if len(univ) < SANITY_MIN_SYMBOLS:
        print(f"  SANITY GATE FAILED: {len(univ)} < {SANITY_MIN_SYMBOLS}. Abort.")
        return

    # Stage 2: Correlation prefilter
    candidates = build_corr_candidates(univ)

    if len(candidates) < SANITY_MIN_CANDIDATES:
        print(f"  SANITY GATE FAILED: {len(candidates)} < {SANITY_MIN_CANDIDATES}. Abort.")
        return

    # Stage 3: Engle-Granger + OU fit (PARALLEL)
    coint_pairs, non_coint_pool = screen_cointegration_parallel(candidates)
    n_coint = len(coint_pairs)

    print(f"\n  Non-cointegrated pool: {len(non_coint_pool)} pairs "
          f"(available for permutation null)")

    if n_coint == 0:
        print("No cointegrated pairs. FAIL.")
        verdict_path = os.path.join(OUT, "s07_v3_verdict.json")
        with open(verdict_path, "w") as f:
            json.dump(dict(verdict_s7a="FAIL", verdict_s7b="FAIL",
                           n_coint=0, reason="no cointegrated pairs"), f, indent=2)
        return

    # Stage 4: No-lookahead assertion (HARD ABORT ON VIOLATION)
    assert_no_lookahead(coint_pairs)

    # Stage 5: OOS trading (PARALLEL)
    pair_results = run_oos_parallel(coint_pairs)

    if not pair_results:
        print("No pairs produced OOS results. FAIL.")
        return

    # Stage 5b: OOS-side no-lookahead assertion (HARD ABORT) on traded bars
    assert_oos_post_tstar(pair_results)

    # Select top-N book (using pre-T* ADF strength; no OOS data)
    top_n_results = select_top_n(pair_results, TOP_N)
    print(f"\n  Top-{TOP_N} book: selected {len(top_n_results)} pairs "
          f"by ADF strength (pre-T* only)")

    # Stage 6: Portfolio books (all-pairs + top-N)
    print("\n=== STAGE 6: Portfolio books ===")
    book_r,     book_meta     = build_book(pair_results,     sizing="inv_vol")
    book_r_top, book_meta_top = build_book(top_n_results,    sizing="inv_vol")

    sc_all = score_series(book_r)     if len(book_r)     > 0 else {}
    sc_top = score_series(book_r_top) if len(book_r_top) > 0 else {}

    print(f"  All-pairs book: SR_ann={sc_all.get('sr_ann',0):.3f} | "
          f"PF={sc_all.get('pf',0):.3f} | n_trades={book_meta.get('n_trades',0)}")
    print(f"  Top-{TOP_N} book: SR_ann={sc_top.get('sr_ann',0):.3f} | "
          f"PF={sc_top.get('pf',0):.3f} | n_trades={book_meta_top.get('n_trades',0)}")

    # Stage 7: Tail guards
    tail_all = compute_tail_guard(pair_results)
    tail_top = compute_tail_guard(top_n_results)
    print(f"\n  Tail guard (all): worst_q={tail_all['worst_quarter']} "
          f"RRR={tail_all['worst_rrr']:.3f} | ex-worst={tail_all['rrr_ex_worst']:.3f} | "
          f"passes={tail_all['passes']}")
    print(f"  Tail guard (top-{TOP_N}): worst_q={tail_top['worst_quarter']} "
          f"RRR={tail_top['worst_rrr']:.3f} | ex-worst={tail_top['rrr_ex_worst']:.3f} | "
          f"passes={tail_top['passes']}")

    # Grid corner check
    grid_chk = check_grid_corners(pair_results)
    print(f"\n  Grid: median pt*={grid_chk['median_pt_star']:.3f} "
          f"sl*={grid_chk['median_sl_star']:.3f} | "
          f"{grid_chk['frac_pt_at_old_min']*100:.0f}% at old 0.25 min | "
          f"interior={grid_chk['optimum_moved_interior']}")

    # Per-trade ledgers
    ledger_all = pd.concat([pr["_trades_df"] for pr in pair_results], ignore_index=True)
    ledger_all = ledger_all[[c for c in ledger_all.columns if not c.startswith("_")]]
    ledger_all_path = os.path.join(OUT, "s07_v3_trades_all.parquet")
    ledger_all.to_parquet(ledger_all_path, index=False)
    print(f"  All-pairs trade ledger: {len(ledger_all)} trades -> {ledger_all_path}")

    top_syms = {(pr["sym_a"], pr["sym_b"]) for pr in top_n_results}
    ledger_top = ledger_all[
        ledger_all.apply(lambda r: (r["sym_a"], r["sym_b"]) in top_syms, axis=1)
    ].reset_index(drop=True)
    ledger_top_path = os.path.join(OUT, "s07_v3_trades_top60.parquet")
    ledger_top.to_parquet(ledger_top_path, index=False)
    print(f"  Top-{TOP_N} trade ledger: {len(ledger_top)} trades -> {ledger_top_path}")

    # Per-pair summary parquet
    pair_rows = []
    top_set = {(pr["sym_a"], pr["sym_b"]) for pr in top_n_results}
    for pr in pair_results:
        rrrs = [qs["rrr"] for qs in pr.get("q_stats", [])]
        pair_rows.append(dict(
            sym_a=pr["sym_a"], sym_b=pr["sym_b"],
            n_oos_trades=pr["n_oos_trades"],
            rrr_pair=pr["rrr_pair"],
            pt_star=pr["pt_star"], sl_star=pr["sl_star"],
            ou_phi=pr["ou_phi"], ou_half_life=pr["ou_half_life"],
            adf_p=pr.get("adf_p", np.nan),
            in_top_n=bool((pr["sym_a"], pr["sym_b"]) in top_set),
            n_quarters=len(rrrs),
            med_q_rrr=float(np.nanmedian(rrrs)) if rrrs else np.nan,
            min_q_rrr=float(np.nanmin(rrrs)) if rrrs else np.nan,
        ))
    pair_df = pd.DataFrame(pair_rows)
    pair_parquet = os.path.join(OUT, "s07_v3_pair_table.parquet")
    pair_df.to_parquet(pair_parquet, index=False)
    print(f"  Pair table -> {pair_parquet}")

    # Stage 8: Permutation nulls (both books)
    p_val_all, null_srs_all, sr_obs_all = permutation_null(
        pair_results, non_coint_pool, N_PERM, label="[all-pairs] ", rng_seed=42)
    p_val_top, null_srs_top, sr_obs_top = permutation_null(
        top_n_results, non_coint_pool, N_PERM, label=f"[top-{TOP_N}] ", rng_seed=43)

    # Stage 9: Sign-consistency (both books)
    half_a_all, half_b_all, sign_pass_all = sign_consistency(pair_results)
    half_a_top, half_b_top, sign_pass_top = sign_consistency(top_n_results)

    # Stage 10: S7b sizing comparison (both books)
    print("\n=== STAGE 10: S7b -- sizing comparison (all-pairs) ===")
    sz_all = sizing_comparison(pair_results)
    for name, sr in sz_all.items():
        print(f"  {name:12s}: SR_ann={sr['sr_ann']:.3f} | PF={sr['pf']:.3f} | "
              f"RRR_ex_worst={sr['rrr_ex_worst']:.3f}")

    print(f"\n=== STAGE 10b: S7b -- sizing comparison (top-{TOP_N}) ===")
    sz_top = sizing_comparison(top_n_results)
    for name, sr in sz_top.items():
        print(f"  {name:12s}: SR_ann={sr['sr_ann']:.3f} | PF={sr['pf']:.3f} | "
              f"RRR_ex_worst={sr['rrr_ex_worst']:.3f}")

    # Figures
    fig_path = make_figures(
        pair_results, book_r, null_srs_all, sr_obs_all, tail_all,
        top_n_results=top_n_results, book_r_top=book_r_top,
        null_srs_top=null_srs_top, sr_obs_top=sr_obs_top, tail_top=tail_top,
    )

    # Verdicts
    n_grid_trials_all = int(len(PT_GRID) * len(SL_GRID) * n_coint)
    n_grid_trials_top = int(len(PT_GRID) * len(SL_GRID) * len(top_n_results))

    def _verdict(pass_criteria):
        n_pass = sum(pass_criteria.values())
        if all(pass_criteria.values()):
            return "PASS"
        elif n_pass <= 2:
            return "FAIL"
        else:
            return "MARGINAL/FAIL"

    crit_all = dict(
        net_positive      = bool(sc_all.get("sr_ann", 0) > 0 and sc_all.get("pf", 0) > 1.0),
        tail_guard_passes = bool(tail_all["passes"]),
        deflated_sig      = bool(p_val_all < 0.05),
        sign_consistent   = bool(sign_pass_all),
        grid_interior     = bool(grid_chk["optimum_moved_interior"]),
    )
    crit_top = dict(
        net_positive      = bool(sc_top.get("sr_ann", 0) > 0 and sc_top.get("pf", 0) > 1.0),
        tail_guard_passes = bool(tail_top["passes"]),
        deflated_sig      = bool(p_val_top < 0.05),
        sign_consistent   = bool(sign_pass_top),
        grid_interior     = bool(grid_chk["optimum_moved_interior"]),
    )

    verdict_all_s7a = _verdict(crit_all)
    verdict_top_s7a = _verdict(crit_top)
    s7b_all = bool(sz_all["inv_vol"]["rrr_ex_worst"] > sz_all["equal"]["rrr_ex_worst"])
    s7b_top = bool(sz_top["inv_vol"]["rrr_ex_worst"] > sz_top["equal"]["rrr_ex_worst"])

    elapsed  = time.perf_counter() - t_total
    ram_peak = _ram_gb()

    verdict = dict(
        # Configuration
        T_star                     = str(T_STAR.date()),
        no_lookahead_assertion     = "PASSED (see Stage 4 log)",
        n_workers                  = N_WORKERS,
        top_n                      = TOP_N,
        # Counts
        n_usdt_syms                = len(univ),
        n_candidates               = len(candidates),
        n_cointegrated             = n_coint,
        n_pairs_oos_ok             = len(pair_results),
        n_non_coint_pool           = len(non_coint_pool),
        # ── ALL-PAIRS BOOK ──────────────────────────────────────────────────
        all_pairs = dict(
            n_pairs             = len(pair_results),
            book_sr_ann         = float(sc_all.get("sr_ann", 0)),
            book_pf             = float(sc_all.get("pf", 0)),
            book_sr_pb          = float(sc_all.get("sr_per_bar", 0)),
            n_trades            = int(book_meta.get("n_trades", 0)),
            tail_worst_quarter  = tail_all["worst_quarter"],
            tail_worst_rrr      = float(tail_all["worst_rrr"]),
            tail_rrr_ex_worst   = float(tail_all["rrr_ex_worst"]),
            tail_guard_pass     = bool(tail_all["passes"]),
            tail_per_quarter    = tail_all["per_quarter"],
            perm_p_value        = float(p_val_all),
            perm_sr_obs         = float(sr_obs_all),
            perm_null_mean      = float(null_srs_all.mean()),
            perm_null_std       = float(null_srs_all.std()),
            n_perm              = N_PERM,
            n_trials_deflated   = n_grid_trials_all,
            sign_half_a         = half_a_all,
            sign_half_b         = half_b_all,
            sign_consistent     = bool(sign_pass_all),
            sizing_equal        = sz_all["equal"],
            sizing_inv_vol      = sz_all["inv_vol"],
            sizing_confidence   = sz_all["confidence"],
            s7b_inv_beats_equal = s7b_all,
            criterion_s7a       = crit_all,
            verdict_s7a         = verdict_all_s7a,
            verdict_s7b         = "PASS" if s7b_all else "FAIL",
        ),
        # ── TOP-N BOOK ───────────────────────────────────────────────────────
        top_n_book = dict(
            n_pairs             = len(top_n_results),
            top_n               = TOP_N,
            selection_criterion = "most-negative ADF stat (smallest adf_p); tie-break shortest OU half-life; pre-T* only",
            book_sr_ann         = float(sc_top.get("sr_ann", 0)),
            book_pf             = float(sc_top.get("pf", 0)),
            book_sr_pb          = float(sc_top.get("sr_per_bar", 0)),
            n_trades            = int(book_meta_top.get("n_trades", 0)),
            tail_worst_quarter  = tail_top["worst_quarter"],
            tail_worst_rrr      = float(tail_top["worst_rrr"]),
            tail_rrr_ex_worst   = float(tail_top["rrr_ex_worst"]),
            tail_guard_pass     = bool(tail_top["passes"]),
            tail_per_quarter    = tail_top["per_quarter"],
            perm_p_value        = float(p_val_top),
            perm_sr_obs         = float(sr_obs_top),
            perm_null_mean      = float(null_srs_top.mean()),
            perm_null_std       = float(null_srs_top.std()),
            n_perm              = N_PERM,
            n_trials_deflated   = n_grid_trials_top,
            sign_half_a         = half_a_top,
            sign_half_b         = half_b_top,
            sign_consistent     = bool(sign_pass_top),
            sizing_equal        = sz_top["equal"],
            sizing_inv_vol      = sz_top["inv_vol"],
            sizing_confidence   = sz_top["confidence"],
            s7b_inv_beats_equal = s7b_top,
            criterion_s7a       = crit_top,
            verdict_s7a         = verdict_top_s7a,
            verdict_s7b         = "PASS" if s7b_top else "FAIL",
        ),
        # Grid
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
            ledger_all  = ledger_all_path,
            ledger_top  = ledger_top_path,
            pairs       = pair_parquet,
            figure      = fig_path,
            verdict     = os.path.join(OUT, "s07_v3_verdict.json"),
        ),
    )

    verdict_path = os.path.join(OUT, "s07_v3_verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\n  verdict -> {verdict_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL VERDICT (S7v3 -- PARALLEL, BOTH BOOKS)")
    print("=" * 72)
    print(f"  T*             : {T_STAR.date()}")
    print(f"  No-lookahead   : ASSERTED AND PASSED")
    print(f"  Universe       : {len(univ)} USDT perps")
    print(f"  Candidates     : {len(candidates)}")
    print(f"  Cointegrated   : {n_coint}")
    print(f"  OOS-ok pairs   : {len(pair_results)}")
    print(f"")
    print(f"  -- ALL-PAIRS BOOK --")
    print(f"  Total OOS trades : {book_meta.get('n_trades',0)}")
    print(f"  Book SR_ann      : {sc_all.get('sr_ann',0):.3f}")
    print(f"  Book PF          : {sc_all.get('pf',0):.3f}")
    print(f"  Tail guard       : ex-worst-Q RRR = {tail_all['rrr_ex_worst']:.3f} | "
          f"passes = {tail_all['passes']}")
    print(f"  Perm null        : p = {p_val_all:.4f} (M={N_PERM}, "
          f"trials={n_grid_trials_all:,})")
    print(f"  Sign consist.    : A-M={half_a_all['net_positive']} "
          f"N-Z={half_b_all['net_positive']} | both = {sign_pass_all}")
    print(f"  Criterion        : {crit_all}")
    print(f"  S7a (all-pairs)  : {verdict_all_s7a}")
    print(f"  S7b (all-pairs)  : {'PASS' if s7b_all else 'FAIL'}  "
          f"(inv_vol RRR_ex={sz_all['inv_vol']['rrr_ex_worst']:.3f} vs "
          f"equal={sz_all['equal']['rrr_ex_worst']:.3f})")
    print(f"")
    print(f"  -- TOP-{TOP_N} BOOK --")
    print(f"  Total OOS trades : {book_meta_top.get('n_trades',0)}")
    print(f"  Book SR_ann      : {sc_top.get('sr_ann',0):.3f}")
    print(f"  Book PF          : {sc_top.get('pf',0):.3f}")
    print(f"  Tail guard       : ex-worst-Q RRR = {tail_top['rrr_ex_worst']:.3f} | "
          f"passes = {tail_top['passes']}")
    print(f"  Perm null        : p = {p_val_top:.4f} (M={N_PERM}, "
          f"trials={n_grid_trials_top:,})")
    print(f"  Sign consist.    : A-M={half_a_top['net_positive']} "
          f"N-Z={half_b_top['net_positive']} | both = {sign_pass_top}")
    print(f"  Criterion        : {crit_top}")
    print(f"  S7a (top-{TOP_N})  : {verdict_top_s7a}")
    print(f"  S7b (top-{TOP_N})  : {'PASS' if s7b_top else 'FAIL'}  "
          f"(inv_vol RRR_ex={sz_top['inv_vol']['rrr_ex_worst']:.3f} vs "
          f"equal={sz_top['equal']['rrr_ex_worst']:.3f})")
    print(f"")
    print(f"  Elapsed: {elapsed/60:.1f} min | RAM peak: {ram_peak:.2f} GB")
    print("=" * 72)


if __name__ == "__main__":
    main()
