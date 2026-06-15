#!/usr/bin/env python3
"""
S7a + S7b  –  Spread mean-reversion at breadth + vol-targeted sizing
=====================================================================
RE-RUN (fixed universe/prefilter bug).

THE BUG IN THE PRIOR RUN:
  The prefilter anchored to BTC's full 2019-2023 history, requiring every
  symbol to have data from 2019.  That collapsed the universe to 124 old coins
  and a |corr|>=0.7 cutoff left only 8 candidate pairs -> 0 cointegrated.

THE FIX (3 changes to stages 1-2):
  1. RECENT prefilter window (2024-01-01 -> data end, i.e. ~21 k hourly bars).
     Include any symbol with >=50% bar-coverage IN THAT WINDOW.  No per-symbol
     total-history requirement at the prefilter stage (that check lives at the
     WFO stage only).  Log how many symbols enter the matrix (expect 250-350).
  2. Top-K candidate ranking, NOT a hard 0.7 cutoff.  Keep each symbol's top-5
     partners by |corr| AND all pairs with |corr|>=0.50.  Target 1000-3000
     candidate pairs.  Log the count.
  3. Per-pair own overlap.  For EG test + OU fit + rolling WFO, align each pair
     on ITS OWN full shared history (not the global prefilter grid).

EARLY SANITY GATE:
  After stages 1-2 assert universe >= 200 symbols and candidates >= 500.
  If not, stop and report counts (don't waste compute on a collapsed universe).

Everything else per the locked S7a A-bar:
  - Threads pinned; 7 bp per leg both legs entry+exit; RAM-safe one-pair-at-a-time.
  - NO-LOOKAHEAD: beta/OU/pair-selection TRAIN-only; intrabar exits OHLC-bounded.
  - ROLLING WFO (>=5 folds) per cointegrated pair -> enables the tail-guard.
  - Grid extended below 0.25 (0.05..3.0) to resolve the corner degeneracy.
  - PORTFOLIO BOOK across cointegrated pairs (inverse-vol/equal-risk).
  - DEFLATION M>=1000: permutation null (shuffle side), p-value.
  - SIGN-CONSISTENCY across two disjoint halves of the cointegrated universe.
  - S7b: inv-vol vs equal vs confidence sizing.
  - Full per-trade LEDGER -> parquet; artifacts + verdict.json under OUT.

Peak RAM estimate: prefilter matrix ~350 symbols x ~21k hourly bars x 8 bytes
  ~ 59 MB for returns matrix; held briefly then freed.
  Pair-by-pair: 2 frames < 100 MB each.
  MC buffer 20k x 500 x 8 = 80 MB.
  Total peak < 1 GB.

THREAD PINNING -- set before importing numpy/scipy/numba.
"""
from __future__ import annotations
import os
# Pin all threading backends BEFORE importing numpy
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import sys, gc, time, json, warnings, resource, itertools
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
from lib import overfit as O

# ── output directory ──────────────────────────────────────────────────────────
OUT = HERE
os.makedirs(OUT, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
PERP_DIR = CRYPTO_PERP_1H

# ── global parameters ─────────────────────────────────────────────────────────
IS_FRAC         = 0.6
ADF_P_MAX       = 0.05
# NOTE: CORR_THRESHOLD is now only used as a FALLBACK lower bound.
# Primary selection is top-K per symbol (see CORR_TOP_K and CORR_MIN_PAIRS_PER_SYM).
CORR_THRESHOLD  = 0.50          # |corr| >= this to enter (stage 2 lower bound)
CORR_TOP_K      = 5             # keep each symbol's top-K partners by |corr|
MIN_SHARED_BARS = 8000          # per-pair EG test minimum (own shared history)
MIN_OOS_TRADES  = 25
COST_BP_PER_LEG = 7.0           # bp per leg per fill; 4 fills/RT = 28 bp total
ENTRY_K         = 2.0
Z_SPAN          = 500
MAX_HOLD        = 200
BARS_PER_YEAR   = 24.0 * 365.25

# Extended grid -- resolves the corner-pinning degeneracy noted in shallow run
PT_GRID = np.round(np.concatenate([
    np.arange(0.05, 0.26, 0.05),    # 0.05, 0.10, 0.15, 0.20, 0.25
    np.arange(0.50, 3.01, 0.25),    # 0.50 ... 3.00
]), 4)
SL_GRID = PT_GRID.copy()

N_PATHS    = 20000
MC_HORIZON = 500
MC_SEED    = 12

# WFO: rolling windows -- >= 5 folds
N_WFO_FOLDS   = 6       # produces up to 6 IS->OOS windows
WFO_IS_BARS   = 20000   # ~2.3 yr IS
WFO_OOS_BARS  = 5000    # ~7 months OOS
WFO_MIN_BARS  = WFO_IS_BARS + WFO_OOS_BARS  # 25000 -- required for any WFO

# Permutation null
N_PERM = 1000

HEARTBEAT = 50          # print every N pairs during prefilter

# ── STAGE 1+2 parameters: RECENT prefilter window ─────────────────────────────
# Use the most recent available data window for correlation prefilter.
# This maximises the number of coexisting symbols (late-2024 listings have data).
# CORR_REF_END = None -> inferred from BTC at runtime (latest available bar).
# NO-LOOKAHEAD: the correlation prefilter window lies entirely within the IS
# period of subsequent WFO runs (since WFO IS starts from the pair's own history
# beginning, and OOS is the last WFO_OOS_BARS of that pair's history).
# The prefilter is train-only by design -- it never touches OOS bars.
CORR_REF_START = pd.Timestamp("2024-01-01", tz="UTC")   # recent 2-yr window
# Coverage requirement for the prefilter: any symbol with >=50% bar coverage
# in [CORR_REF_START, CORR_REF_END] is included (no total-history filter here).
CORR_MIN_COVERAGE_FRAC = 0.50

# Early sanity gate thresholds
SANITY_MIN_SYMBOLS    = 200    # abort if fewer symbols enter the matrix
SANITY_MIN_CANDIDATES = 500    # abort if fewer candidate pairs produced


# ── helpers ───────────────────────────────────────────────────────────────────

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
    laO, laH, laL, laC = (np.log(a[c].to_numpy(np.float64)) for c in ("open","high","low","close"))
    lbO, lbH, lbL, lbC = (np.log(b[c].to_numpy(np.float64)) for c in ("open","high","low","close"))
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


def spread_rolling_stats(s_close, span):
    s = pd.Series(np.asarray(s_close, np.float64))
    mu = s.ewm(span=span, adjust=True).mean()
    sd = s.ewm(span=span, adjust=True).std()
    z = ((s - mu) / sd.replace(0, np.nan)).to_numpy()
    z[~np.isfinite(z)] = 0.0
    sig = sd.to_numpy()
    sig[~np.isfinite(sig)] = 0.0
    return z, sig


def spread_entry_events(z, entry_k, warm):
    z = np.asarray(z, np.float64)
    over  = np.abs(z) >= entry_k
    fresh = over.copy(); fresh[1:] &= ~over[:-1]
    idx   = np.where(fresh)[0]; idx = idx[idx > warm]
    side  = -np.sign(z[idx]).astype(np.int64)
    keep  = side != 0
    return idx[keep].astype(np.int64), side[keep]


try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def _njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]


@_njit(cache=True)
def _spread_exit_kernel(ev_idx, side, sig_ev, s_open, s_high, s_low, s_close,
                        pt_mult, sl_mult, max_hold):
    """Intrabar first-touch barrier rule. Adverse direction checked first."""
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
        np.ascontiguousarray(np.asarray(side, np.int64)),
        f(sig_ev), f(s_open), f(s_high), f(s_low), f(s_close),
        float(pt_mult), float(sl_mult), int(max_hold))
    return pd.DataFrame({"ev_idx": ev_idx, "side": side, "touch": touch,
                         "label": lab, "pnl_gross": pnl, "hold": hold})


def score_series(r, bpy=BARS_PER_YEAR):
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


# ── STAGE 1: Correlation prefilter (RECENT window, no WFO-bar requirement) ────

def build_corr_prefilter(all_syms: list[str]):
    """
    FIXED prefilter (replaces the bug that collapsed the universe to 124 symbols).

    Key changes vs prior run:
      1. Recent window [CORR_REF_START, data_end] -- maximises coexisting symbols.
      2. Requires only CORR_MIN_COVERAGE_FRAC in-window bars (not WFO_MIN_BARS total).
      3. fillna(0) returns so late-listers still participate -- matrix NOT
         intersection-collapsed to the oldest coins.

    NO-LOOKAHEAD: the prefilter window is IS-data only -- it lies before the
    OOS portion of any subsequent per-pair WFO (which uses the pair's own full
    shared history; its last WFO_OOS_BARS form the OOS, the earlier data is IS).
    The prefilter itself uses raw close returns, not any statistic that peeks
    at future data.

    RAM: ~350 symbols x ~21k bars x 8 bytes ~ 59 MB for the returns matrix.
    """
    print("\n=== STAGE 1: building correlation prefilter (RECENT WINDOW) ===")

    # Infer data end from BTC (assumed to always be current)
    btc = load_perp("BTCUSDT")
    data_end = btc.index.max()
    del btc; gc.collect()
    corr_end = data_end  # use full available recent data

    expected_bars = int((corr_end - CORR_REF_START).total_seconds() / 3600)
    min_in_window = int(expected_bars * CORR_MIN_COVERAGE_FRAC)

    print(f"  Recent window: {CORR_REF_START.date()} -> {corr_end.date()} "
          f"({expected_bars} expected hourly bars)")
    print(f"  Coverage requirement: >= {CORR_MIN_COVERAGE_FRAC*100:.0f}% "
          f"= {min_in_window} bars (no total-history requirement)")
    print(f"  Loading {len(all_syms)} symbols one-at-a-time")

    # Build the master hourly index for the reference window from BTC
    btc2 = load_perp("BTCUSDT")
    ref_idx = btc2.loc[CORR_REF_START:corr_end].index
    del btc2; gc.collect()
    n_common = len(ref_idx)
    print(f"  Master index: {n_common} hourly bars")

    lp_cols  = []
    syms_ok  = []
    n_loaded = 0
    n_skipped_coverage = 0
    n_skipped_error    = 0

    for i, sym in enumerate(all_syms):
        if (i + 1) % HEARTBEAT == 0:
            print(f"  Prefilter: {i+1}/{len(all_syms)} | "
                  f"{n_loaded} in matrix | {n_skipped_coverage} skip-coverage | "
                  f"RAM {_ram_gb():.2f} GB")
        try:
            df = load_perp(sym)
        except Exception:
            n_skipped_error += 1
            continue

        # Restrict to reference window
        df_ref = df.loc[CORR_REF_START:corr_end]
        n_in_window = len(df_ref)
        del df

        # Coverage check: require CORR_MIN_COVERAGE_FRAC in the window
        # (no WFO_MIN_BARS requirement -- that check is at the WFO stage)
        if n_in_window < min_in_window:
            n_skipped_coverage += 1
            del df_ref; continue

        # Align to master hourly index (fill gaps with NaN -> 0-return later)
        shared = df_ref.index.intersection(ref_idx)
        if len(shared) < 500:    # need enough bars to compute any correlation
            n_skipped_coverage += 1
            del df_ref; continue

        lp_full = np.full(n_common, np.nan)
        pos = ref_idx.searchsorted(shared)
        # Guard against out-of-bounds (should not happen but be safe)
        valid = pos < n_common
        pos = pos[valid]; shared_valid = shared[valid]
        lp_full[pos] = np.log(df_ref.loc[shared_valid, "close"].to_numpy(np.float64))
        lp_cols.append(lp_full)
        syms_ok.append(sym)
        del df_ref; gc.collect()
        n_loaded += 1

    print(f"\n  PREFILTER RESULT: {n_loaded} symbols entered matrix "
          f"(skipped: {n_skipped_coverage} coverage, {n_skipped_error} load-errors)")

    if len(lp_cols) < 2:
        return [], None, ref_idx

    # Stack into matrix; convert log-prices to log-returns; NaN -> 0
    # (NaN = symbol not trading at that bar; 0-return is conservative/correct)
    lp_mat = np.column_stack(lp_cols)            # (n_common, n_syms_ok)
    lr_mat = np.diff(lp_mat, axis=0)             # (n_common-1, n_syms_ok)
    lr_mat = np.where(np.isfinite(lr_mat), lr_mat, 0.0)  # fillna(0)
    print(f"  Return matrix: {lr_mat.shape} | RAM {_ram_gb():.2f} GB")
    return syms_ok, lr_mat, ref_idx[:-1]


# ── STAGE 2: Candidate pairs (top-K + |corr|>=threshold) ─────────────────────

def compute_candidate_pairs(syms: list[str], lr_mat: np.ndarray):
    """
    FIXED candidate generation (replaces the hard 0.7 cutoff that left only 8 pairs).

    Two complementary rules (union):
      A. All pairs with |corr| >= CORR_THRESHOLD (0.50)
      B. Each symbol's top CORR_TOP_K (5) partners by |corr|

    This is designed to produce 1000-3000 candidate pairs from ~200-300 symbols.

    NO-LOOKAHEAD: correlation is computed on the prefilter window, which is
    IS-data only (lies within the IS portion of all subsequent WFO runs).
    """
    print(f"\n=== STAGE 2: candidate pairs (|corr|>={CORR_THRESHOLD} OR top-{CORR_TOP_K} per sym) ===")
    n = lr_mat.shape[1]
    n_pairs_total = n * (n - 1) // 2
    print(f"  {n} symbols -> {n_pairs_total:,} total pairs to scan")

    # Normalize: zero mean, unit std (using the non-zero values; 0-fill means
    # zero-return bars don't bias the mean/var but do dilute the correlation,
    # which is the conservative direction)
    col_mean = lr_mat.mean(axis=0)
    col_std  = lr_mat.std(axis=0)
    col_std  = np.where(col_std < 1e-12, 1.0, col_std)
    lp_c = (lr_mat - col_mean) / col_std

    # Compute pairwise correlation in chunks (memory-efficient)
    chunk = 50
    T = lr_mat.shape[0]

    # corr_mat[i, j] = Pearson corr between symbols i and j (approximate, via
    # normalized dot product; exact for the zero-mean unit-var normalization)
    candidates_set = set()
    # top-K bookkeeping: for each symbol, keep a heap of (|corr|, j) of size K
    from heapq import heappush, heappop, nlargest

    top_k_map = {i: [] for i in range(n)}   # i -> list of (|corr|, j)

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block_a = lp_c[:, i0:i1]             # (T, chunk)
        # corr_block[r, c] = corr(sym i0+r, sym c)
        corr_block = (block_a.T @ lp_c) / T  # (chunk, n)
        corr_block = np.clip(corr_block, -1.0, 1.0)

        for r_local in range(i1 - i0):
            i = i0 + r_local
            row = corr_block[r_local]     # corr of sym i with all n syms
            abs_row = np.abs(row)

            # Rule A: |corr| >= threshold
            jj = np.where(abs_row >= CORR_THRESHOLD)[0]
            for j in jj:
                if j == i: continue
                key = (min(i, j), max(i, j))
                candidates_set.add(key)

            # Rule B: track top-K partners for each symbol
            for j in range(n):
                if j == i: continue
                c = float(abs_row[j])
                lst = top_k_map[i]
                if len(lst) < CORR_TOP_K:
                    heappush(lst, (c, j))
                elif c > lst[0][0]:
                    heappop(lst)
                    heappush(lst, (c, j))

    # Add top-K pairs to candidate set
    for i, lst in top_k_map.items():
        for (c, j) in lst:
            if j == i: continue
            key = (min(i, j), max(i, j))
            candidates_set.add(key)

    # Convert to list of (sym_a, sym_b) with deduplication
    out = [(syms[a], syms[b]) for (a, b) in sorted(candidates_set)]

    print(f"  {len(out):,} candidate pairs "
          f"(|corr|>={CORR_THRESHOLD} OR top-{CORR_TOP_K} per sym)")
    return out


# ── STAGE 3: Engle-Granger on candidate pairs ─────────────────────────────────

def screen_cointegration(candidates: list[tuple], all_syms_set: set):
    """
    For each candidate pair:
      - load both symbols (one at a time)
      - align on THEIR OWN full shared history (not the prefilter window)
      - split at IS_FRAC
      - run Engle-Granger on TRAIN log-prices
      - keep pairs with ADF p <= ADF_P_MAX and >= MIN_SHARED_BARS shared history

    NO-LOOKAHEAD CHECK: hedge ratio beta and ADF test are computed on TRAIN
    portion only (la[:n_is], lb[:n_is]). The OOS portion is never seen here.

    Returns list of dicts with pair metadata.
    """
    print(f"\n=== STAGE 3: Engle-Granger cointegration on {len(candidates):,} candidates ===")
    coint_pairs = []
    seen = set()
    n_tested = 0
    n_too_short = 0

    for sym_a, sym_b in candidates:
        key = tuple(sorted([sym_a, sym_b]))
        if key in seen:
            continue
        seen.add(key)

        if sym_a not in all_syms_set or sym_b not in all_syms_set:
            continue

        try:
            a0 = load_perp(sym_a); b0 = load_perp(sym_b)
        except Exception:
            continue

        # Per-pair own overlap -- NOT constrained to any global grid
        idx = a0.index.intersection(b0.index)
        a = a0.loc[idx]; b = b0.loc[idx]
        del a0, b0; gc.collect()

        n = len(idx)
        if n < MIN_SHARED_BARS:
            del a, b; n_too_short += 1; continue

        la = np.log(a["close"].to_numpy(np.float64))
        lb = np.log(b["close"].to_numpy(np.float64))
        n_is = int(n * IS_FRAC)

        alpha, beta, adf_p = engle_granger_train(la[:n_is], lb[:n_is])
        cointegrated = (adf_p <= ADF_P_MAX) and np.isfinite(beta)

        n_tested += 1
        if n_tested % 200 == 0:
            print(f"  EG test: {n_tested}/{len(candidates)} tested | "
                  f"{len(coint_pairs)} cointegrated | {n_too_short} too-short | "
                  f"RAM {_ram_gb():.2f} GB")

        if cointegrated:
            coint_pairs.append(dict(
                sym_a=sym_a, sym_b=sym_b, n=n, n_is=n_is,
                alpha=alpha, beta=beta, adf_p=adf_p,
                _idx=idx,
            ))

        del a, b; gc.collect()

    print(f"  EG: tested {n_tested} / {len(candidates)} candidates | "
          f"{len(coint_pairs)} cointegrated (ADF p <= {ADF_P_MAX}) | "
          f"{n_too_short} skipped (< {MIN_SHARED_BARS} shared bars)")
    return coint_pairs


# ── STAGE 4: Rolling WFO per pair ─────────────────────────────────────────────

def _wfo_windows(n: int, n_is: int, n_oos: int, n_folds: int):
    """
    Generate (is_start, is_end, oos_end) index tuples for rolling WFO.
    Sliding IS window (not expanding). Non-overlapping OOS windows.

    NO-LOOKAHEAD: is_end is the boundary; oos spans [is_end, oos_end).
    """
    total_needed = n_is + n_oos
    if n < total_needed:
        return []

    windows = []
    oos_end = n
    for _ in range(n_folds):
        oos_start = oos_end - n_oos
        is_start  = oos_start - n_is
        if is_start < 0:
            break
        windows.append((is_start, oos_start, oos_end))
        oos_end = oos_start

    windows.reverse()
    return windows


def run_pair_wfo(sym_a: str, sym_b: str, meta: dict):
    """
    Rolling WFO for one cointegrated pair.

    NO-LOOKAHEAD:
      - beta fit on IS la, lb only
      - OU params fit on IS spread residual only
      - vol scaler (sig_oos) is causal EWMA -- no OOS data used in fitting
      - pair selection done in Stage 3 on the IS portion (frozen)
      - intrabar exits use spread_ohlc OHLC bounds (leg-OHLC-bounded)
    """
    try:
        a0 = load_perp(sym_a)
        b0 = load_perp(sym_b)
    except Exception as e:
        return None, f"load error: {e}"

    idx = a0.index.intersection(b0.index)
    a = a0.loc[idx]; b = b0.loc[idx]
    del a0, b0; gc.collect()

    n = len(idx)
    if n < WFO_MIN_BARS:
        del a, b; return None, f"short history: {n} < {WFO_MIN_BARS}"

    windows = _wfo_windows(n, WFO_IS_BARS, WFO_OOS_BARS, N_WFO_FOLDS)
    if len(windows) < 3:
        del a, b; return None, f"too few WFO windows: {len(windows)}"

    la = np.log(a["close"].to_numpy(np.float64))
    lb = np.log(b["close"].to_numpy(np.float64))

    cpl     = COST_BP_PER_LEG / 1e4
    rt_cost = 4.0 * cpl   # 4 fills per spread RT

    window_rows  = []
    window_stats = []

    for w_idx, (is_start, oos_start, oos_end) in enumerate(windows):
        # ── IS: refit beta + OU ──────────────────────────────────────────────
        la_is = la[is_start:oos_start]
        lb_is = lb[is_start:oos_start]

        alpha_w, beta_w, adf_p_w = engle_granger_train(la_is, lb_is)
        if not np.isfinite(beta_w):
            continue

        # ── Spread OHLC for this window's [is_start, oos_end) slice ─────────
        a_w = a.iloc[is_start:oos_end]
        b_w = b.iloc[is_start:oos_end]
        sO, sH, sL, sC = spread_ohlc(a_w, b_w, beta_w)

        # ── Causal rolling stats over the IS+OOS window ──────────────────────
        z_w, sig_w = spread_rolling_stats(sC, Z_SPAN)

        # ── OU fit on IS spread residual only ────────────────────────────────
        s_ser = pd.Series(sC)
        roll_mu_w = s_ser.ewm(span=Z_SPAN, adjust=True).mean().to_numpy()
        resid_is  = (sC - roll_mu_w)[:oos_start - is_start]
        resid_is  = resid_is[np.isfinite(resid_is)]
        fit = otr.fit_ou(resid_is)
        if not fit["ok"]:
            continue

        stat_sd = fit["sigma"] / max(np.sqrt(1.0 - fit["phi"]**2), 1e-9)

        # ── OU mesh: derive (pt*, sl*) on fitted IS OU process ───────────────
        x0_dev = fit["E0"] + ENTRY_K * stat_sd
        sh_mesh, _ = deepen.ou_mesh_dev(
            fit["E0"], fit["phi"], fit["sigma"], x0=x0_dev,
            pt_grid=PT_GRID, sl_grid=SL_GRID,
            n_paths=N_PATHS, max_horizon=MC_HORIZON, seed=MC_SEED)
        pt_star, sl_star, sh_star, _ = otr.optimal_rule(sh_mesh, PT_GRID, SL_GRID)
        pt_mult = pt_star / stat_sd
        sl_mult = sl_star / stat_sd
        del sh_mesh; gc.collect()

        # ── OOS trading ──────────────────────────────────────────────────────
        oos_len    = oos_end - oos_start
        warm_w     = Z_SPAN + 5
        ev_all, side_all = spread_entry_events(z_w, ENTRY_K, warm_w)
        oos_offset = oos_start - is_start
        oos_mask   = ev_all >= oos_offset
        ev_oos     = ev_all[oos_mask]
        side_oos   = side_all[oos_mask]
        if len(ev_oos) < MIN_OOS_TRADES:
            continue
        sig_oos = sig_w[ev_oos]

        tb = apply_spread_rule(sO, sH, sL, sC, sig_oos, ev_oos, side_oos,
                               pt_mult, sl_mult, MAX_HOLD)
        pnl_net = tb["pnl_gross"].to_numpy() - rt_cost

        tb_out = tb.copy()
        tb_out["window"]        = w_idx
        tb_out["sym_a"]         = sym_a
        tb_out["sym_b"]         = sym_b
        tb_out["pnl_net"]       = pnl_net
        tb_out["beta_w"]        = beta_w
        tb_out["adf_p_w"]       = adf_p_w
        tb_out["pt_star"]       = pt_star
        tb_out["sl_star"]       = sl_star
        tb_out["is_start_abs"]  = int(is_start)
        tb_out["oos_start_abs"] = int(is_start + oos_offset)
        tb_out["oos_end_abs"]   = int(oos_end)
        tb_out["global_bar"]    = (ev_oos + is_start).astype(np.int64)
        tb_out["entry_ts"]      = idx[np.minimum(ev_oos + is_start + 1, n - 1)].strftime("%Y-%m-%dT%H:%M")
        window_rows.append(tb_out)

        # ── window stats for RRR tail-guard ─────────────────────────────────
        wins   = pnl_net[pnl_net > 0]
        losses = pnl_net[pnl_net < 0]
        rrr_w  = (wins.mean() / abs(losses.mean())) if (len(wins) > 0 and len(losses) > 0) else 0.0

        r_oos = np.zeros(oos_len)
        for k in range(len(ev_oos)):
            i0 = int(ev_oos[k]) - oos_offset; h = max(1, int(tb["hold"].iloc[k]))
            per = pnl_net[k] / h
            j1  = min(i0 + h, oos_len)
            r_oos[i0:j1] += per
        sc = score_series(r_oos)
        window_stats.append(dict(
            window=w_idx, is_start=is_start, oos_start=oos_start, oos_end=oos_end,
            n_trades=len(ev_oos), sr_ann=sc["sr_ann"], pf=sc["pf"],
            rrr=float(rrr_w), adf_p=adf_p_w, pt_star=pt_star, sl_star=sl_star,
            beta=beta_w,
        ))

    del a, b; gc.collect()

    if not window_rows or not window_stats:
        return None, "no usable WFO windows"

    return dict(
        sym_a=sym_a, sym_b=sym_b,
        window_stats=window_stats,
        _trades_df=pd.concat(window_rows, ignore_index=True),
    ), "ok"


# ── STAGE 5: Portfolio book assembly ──────────────────────────────────────────

def build_book(pair_results: list[dict], sizing: str = "inv_vol"):
    """
    Combine per-pair OOS trade returns into a portfolio book.

    sizing:
      "equal"      -- equal notional per trade
      "inv_vol"    -- inverse proportional to pair's trailing OOS pnl std
      "confidence" -- proportional to IS mesh Sharpe proxy (pt*/sl*)

    NO-LOOKAHEAD for inv_vol: vol scaler for each trade is EWMA std of prior
    pnl from the SAME pair's earlier windows only (purely causal).
    """
    all_trades = []
    for pr in pair_results:
        df = pr["_trades_df"].copy()
        all_trades.append(df)
    if not all_trades:
        return np.array([]), {}
    trades = pd.concat(all_trades, ignore_index=True)

    pair_window_vol = {}
    for pr in pair_results:
        sym_a, sym_b = pr["sym_a"], pr["sym_b"]
        ws = sorted(pr["window_stats"], key=lambda w: w["window"])
        running_pnl = []
        for w in ws:
            wdf = pr["_trades_df"][pr["_trades_df"]["window"] == w["window"]]
            pnl_w = wdf["pnl_net"].to_numpy()
            running_pnl.extend(pnl_w.tolist())
            if len(running_pnl) > 5:
                pair_window_vol[(sym_a, sym_b, w["window"])] = float(np.std(running_pnl))
            else:
                pair_window_vol[(sym_a, sym_b, w["window"])] = 1.0

    def get_weight(row):
        key = (row["sym_a"], row["sym_b"], row["window"])
        if sizing == "equal":
            return 1.0
        elif sizing == "inv_vol":
            vol = pair_window_vol.get(key, 1.0)
            return 1.0 / max(vol, 1e-9)
        elif sizing == "confidence":
            return float(row.get("pt_star", 1.0)) / float(max(row.get("sl_star", 1.0), 1e-9))
        return 1.0

    trades["weight"] = trades.apply(get_weight, axis=1)
    for window in trades["window"].unique():
        wm = trades["window"] == window
        w_sum = trades.loc[wm, "weight"].sum()
        if w_sum > 0:
            trades.loc[wm, "weight"] /= w_sum

    max_bar = int(trades["global_bar"].max()) + MAX_HOLD + 2
    book_r = np.zeros(max_bar)
    for _, row in trades.iterrows():
        i0  = int(row["global_bar"])
        h   = max(1, int(row["hold"]))
        per = row["pnl_net"] * row["weight"] / h
        j1  = min(i0 + h, max_bar)
        book_r[i0:j1] += per

    nz = np.where(book_r != 0)[0]
    if len(nz) == 0:
        return np.array([]), {}
    book_r = book_r[nz[0]:nz[-1]+1]

    sc = score_series(book_r)
    all_pnl = trades["pnl_net"].to_numpy()
    wins   = all_pnl[all_pnl > 0]
    losses = all_pnl[all_pnl < 0]
    rrr_all = (wins.mean() / abs(losses.mean())) if (len(wins) > 0 and len(losses) > 0) else 0.0

    return book_r, dict(
        sizing=sizing,
        n_trades=len(trades),
        sr_ann=sc["sr_ann"], pf=sc["pf"], skew=sc["skew"],
        sr_per_bar=sc["sr_per_bar"],
        rrr_all=float(rrr_all),
        _trades=trades,
    )


def compute_tail_guard(pair_results: list[dict]):
    """
    Ex-worst-window RRR: per-window RRR across all pairs, report minimum
    (worst), mean ex-worst, and flag if ex-worst RRR > 1 (A-bar).
    """
    window_rrr = {}
    for pr in pair_results:
        for ws in pr["window_stats"]:
            w = ws["window"]
            window_rrr.setdefault(w, []).append(ws["rrr"])

    per_window = {w: float(np.mean(rrs)) for w, rrs in window_rrr.items()}
    if not per_window:
        return dict(worst_window=np.nan, rrr_ex_worst=np.nan, passes=False)

    worst_w   = min(per_window, key=per_window.get)
    worst_rrr = per_window[worst_w]
    ex_worst  = [v for w, v in per_window.items() if w != worst_w]
    rrr_ex_worst = float(np.mean(ex_worst)) if ex_worst else worst_rrr

    return dict(
        worst_window=int(worst_w),
        worst_window_rrr=float(worst_rrr),
        rrr_ex_worst=float(rrr_ex_worst),
        passes=bool(rrr_ex_worst > 1.0),
        per_window={int(k): float(v) for k, v in per_window.items()},
    )


# ── STAGE 6: Permutation null (deflation) ────────────────────────────────────

def permutation_null(pair_results: list[dict], n_perm: int,
                     n_grid_trials: int, rng_seed: int = 42):
    """
    Book-level permutation null (side-shuffle primary null).

    Deflation: effective trial count = n_grid_trials * n_pairs_screened.
    p-value = rank of observed Sharpe in null + 1 / (N_PERM + 1).
    """
    print(f"\n=== STAGE 6: Permutation null ({n_perm} permutations) ===")
    rng = np.random.default_rng(rng_seed)

    all_dfs = [pr["_trades_df"] for pr in pair_results]
    if not all_dfs:
        return 1.0, np.array([]), 0.0
    trades = pd.concat(all_dfs, ignore_index=True)
    pnl_arr  = trades["pnl_net"].to_numpy()
    hold_arr = trades["hold"].to_numpy()
    bar_arr  = trades["global_bar"].to_numpy()
    wt_arr   = trades["weight"].to_numpy() if "weight" in trades.columns else np.ones(len(trades))

    max_bar = int(bar_arr.max()) + MAX_HOLD + 2
    book_obs = np.zeros(max_bar)
    for k in range(len(pnl_arr)):
        i0 = int(bar_arr[k]); h = max(1, int(hold_arr[k]))
        per = pnl_arr[k] / h
        j1  = min(i0 + h, max_bar)
        book_obs[i0:j1] += per
    nz = np.where(book_obs != 0)[0]
    book_obs = book_obs[nz[0]:nz[-1]+1] if len(nz) > 0 else book_obs
    sc_obs = score_series(book_obs)
    sr_obs = sc_obs["sr_per_bar"]

    null_srs = np.empty(n_perm)
    for m in range(n_perm):
        if (m + 1) % 200 == 0:
            print(f"  perm {m+1}/{n_perm} | RAM {_ram_gb():.2f} GB")
        flip = rng.integers(0, 2, len(pnl_arr)).astype(np.float64) * 2 - 1
        pnl_perm = pnl_arr * flip
        book_p = np.zeros(max_bar)
        for k in range(len(pnl_perm)):
            i0 = int(bar_arr[k]); h = max(1, int(hold_arr[k]))
            per = pnl_perm[k] / h
            j1  = min(i0 + h, max_bar)
            book_p[i0:j1] += per
        nz_p = np.where(book_p != 0)[0]
        if len(nz_p) == 0:
            null_srs[m] = 0.0; continue
        book_p = book_p[nz_p[0]:nz_p[-1]+1]
        sc_p = score_series(book_p)
        null_srs[m] = sc_p["sr_per_bar"]

    rank  = int((null_srs >= sr_obs).sum())
    p_val = (rank + 1) / (n_perm + 1)

    print(f"  Observed SR_per_bar = {sr_obs:.4f}")
    print(f"  Null SR mean = {null_srs.mean():.4f}, std = {null_srs.std():.4f}")
    print(f"  Rank {rank}/{n_perm} -> p = {p_val:.4f} "
          f"(n_grid_trials={n_grid_trials:,})")
    return float(p_val), null_srs, float(sr_obs)


# ── STAGE 7: Sign-consistency splits ─────────────────────────────────────────

def sign_consistency(pair_results: list[dict]):
    """
    Split into two disjoint halves by sym_a first letter (A-M vs N-Z).
    Both halves must be net-positive (SR > 0, PF > 1).
    """
    half_a = [pr for pr in pair_results if pr["sym_a"][0].upper() <= "M"]
    half_b = [pr for pr in pair_results if pr["sym_a"][0].upper() >  "M"]

    def book_stats(prs, label):
        if not prs:
            return dict(label=label, n_pairs=0, sr_ann=0.0, pf=0.0, net_positive=False)
        _, bm = build_book(prs, sizing="inv_vol")
        return dict(
            label=label, n_pairs=len(prs),
            sr_ann=float(bm.get("sr_ann", 0.0)),
            pf=float(bm.get("pf", 0.0)),
            net_positive=bool(bm.get("sr_ann", 0.0) > 0 and bm.get("pf", 1.0) > 1.0),
        )

    stats_a = book_stats(half_a, "A-M")
    stats_b = book_stats(half_b, "N-Z")
    passes  = bool(stats_a["net_positive"] and stats_b["net_positive"])

    print(f"\n=== STAGE 7: Sign-consistency splits ===")
    print(f"  Half A (sym_a in A-M): {stats_a['n_pairs']} pairs | "
          f"SR={stats_a['sr_ann']:.3f} | PF={stats_a['pf']:.3f} | "
          f"pass={stats_a['net_positive']}")
    print(f"  Half B (sym_a in N-Z): {stats_b['n_pairs']} pairs | "
          f"SR={stats_b['sr_ann']:.3f} | PF={stats_b['pf']:.3f} | "
          f"pass={stats_b['net_positive']}")
    return stats_a, stats_b, passes


# ── STAGE 8: S7b -- sizing comparison ─────────────────────────────────────────

def sizing_comparison(pair_results: list[dict]):
    """Compare equal / inv_vol / confidence on the same pair_results."""
    results = {}
    for sz in ("equal", "inv_vol", "confidence"):
        book_r, bm = build_book(pair_results, sizing=sz)
        if len(book_r) == 0:
            results[sz] = dict(sizing=sz, sr_ann=0.0, pf=0.0, rrr=0.0,
                               rrr_ex_worst=0.0, n_trades=0)
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


# ── STAGE 9: Grid corner check ────────────────────────────────────────────────

def check_grid_corners(pair_results: list[dict]):
    """Report whether optimal (pt*, sl*) moved interior with the extended grid."""
    pt_vals = []
    sl_vals = []
    for pr in pair_results:
        for ws in pr.get("window_stats", []):
            pt_vals.append(ws.get("pt_star", np.nan))
            sl_vals.append(ws.get("sl_star", np.nan))
    pt_arr = np.array([v for v in pt_vals if np.isfinite(v)])
    sl_arr = np.array([v for v in sl_vals if np.isfinite(v)])

    old_min  = 0.25
    grid_min = float(PT_GRID[0])

    pt_at_old_min = float((pt_arr <= old_min).mean()) if len(pt_arr) > 0 else np.nan
    sl_at_old_min = float((sl_arr <= old_min).mean()) if len(sl_arr) > 0 else np.nan
    interior_pt   = float(np.nanmedian(pt_arr)) if len(pt_arr) > 0 else np.nan
    interior_sl   = float(np.nanmedian(sl_arr)) if len(sl_arr) > 0 else np.nan

    optimum_moved_interior = bool(pt_at_old_min < 0.5)

    return dict(
        n_window_fits=len(pt_arr),
        median_pt_star=float(interior_pt),
        median_sl_star=float(interior_sl),
        frac_pt_at_old_min_0_25=float(pt_at_old_min),
        frac_sl_at_old_min_0_25=float(sl_at_old_min),
        optimum_moved_interior=bool(optimum_moved_interior),
        grid_min_used=float(grid_min),
    )


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figures(pair_results, book_r, null_srs, sr_obs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        cum_r = np.cumsum(book_r)
        axes[0].plot(cum_r, lw=1.0, color="#2c7bb6")
        axes[0].axhline(0, color="#888", lw=0.7, ls="--")
        axes[0].set_title("S7a: Portfolio book equity (inv-vol sizing, net of 28bp/RT)")
        axes[0].set_xlabel("Bars (hourly OOS)")
        axes[0].set_ylabel("Cumulative spread P&L (log units)")

        axes[1].hist(null_srs, bins=50, color="#aaa", edgecolor="white", alpha=0.8,
                     label="permutation null")
        axes[1].axvline(sr_obs, color="#d7191c", lw=2.0,
                        label=f"observed SR={sr_obs:.4f}")
        axes[1].set_title("Permutation null: book Sharpe (1000 shuffles)")
        axes[1].set_xlabel("SR per bar")
        axes[1].legend(fontsize=9)

        rrr_all_windows = []
        for pr in pair_results:
            for ws in pr["window_stats"]:
                rrr_all_windows.append(ws["rrr"])
        axes[2].hist(rrr_all_windows, bins=30, color="#abdda4", edgecolor="white")
        axes[2].axvline(1.0, color="#d7191c", lw=1.5, ls="--", label="RRR=1 bar")
        axes[2].set_title("Per-window RRR distribution (A-bar tail guard)")
        axes[2].set_xlabel("Reward/Risk Ratio")
        axes[2].legend(fontsize=9)

        fig.suptitle(
            "S7a/S7b: Spread mean-reversion at breadth (deep run, net of costs)",
            fontsize=12)
        fig.tight_layout()
        out_fig = os.path.join(OUT, "book_equity.png")
        fig.savefig(out_fig, dpi=120)
        plt.close(fig)
        print(f"  figure -> {out_fig}")
        return out_fig
    except Exception as e:
        print(f"  [figure skipped] {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.perf_counter()
    print("=" * 72)
    print("S7a + S7b -- Spread mean-reversion at breadth (RE-RUN, bug fixed)")
    print("FIX: recent prefilter window + top-K candidates + per-pair overlap")
    print("COSTS: 7 bp per-leg per fill, 4 fills/RT = 28 bp total")
    print("=" * 72)

    # ── Discover universe ─────────────────────────────────────────────────────
    all_files = sorted(os.listdir(PERP_DIR))
    all_syms  = [f.replace("_1h.parquet", "") for f in all_files
                 if f.endswith("_1h.parquet")]
    print(f"\nUniverse: {len(all_syms)} symbols in {PERP_DIR}")

    usdt_syms = [s for s in all_syms
                 if s.endswith("USDT") and not s.startswith("1000")]
    print(f"  USDT-margin (excl. 1000x): {len(usdt_syms)} symbols")

    # ── Stage 1: Correlation prefilter ───────────────────────────────────────
    syms_ok, lr_mat, common_idx = build_corr_prefilter(usdt_syms)

    if lr_mat is None or len(syms_ok) == 0:
        print("ERROR: could not build return matrix"); return

    del common_idx; gc.collect()

    n_syms_in_matrix = len(syms_ok)
    print(f"\n  SANITY CHECK 1: {n_syms_in_matrix} symbols entered matrix "
          f"(gate: >= {SANITY_MIN_SYMBOLS})")
    if n_syms_in_matrix < SANITY_MIN_SYMBOLS:
        print(f"  SANITY GATE FAILED: only {n_syms_in_matrix} symbols "
              f"(expected >= {SANITY_MIN_SYMBOLS}). Aborting.")
        return

    # ── Stage 2: Candidate pairs ──────────────────────────────────────────────
    candidates = compute_candidate_pairs(syms_ok, lr_mat)
    del lr_mat; gc.collect()
    print(f"  RAM after corr matrix freed: {_ram_gb():.2f} GB")

    n_candidates = len(candidates)
    print(f"\n  SANITY CHECK 2: {n_candidates} candidate pairs "
          f"(gate: >= {SANITY_MIN_CANDIDATES})")
    if n_candidates < SANITY_MIN_CANDIDATES:
        print(f"  SANITY GATE FAILED: only {n_candidates} candidates "
              f"(expected >= {SANITY_MIN_CANDIDATES}). Aborting.")
        return

    # ── Stage 3: Cointegration screening ─────────────────────────────────────
    all_syms_set = set(syms_ok)
    coint_pairs  = screen_cointegration(candidates, all_syms_set)
    n_cointegrated = len(coint_pairs)
    print(f"\n  Summary: {n_candidates} candidates -> {n_cointegrated} cointegrated")

    if n_cointegrated == 0:
        print("No cointegrated pairs found -- FAIL (universe is real; spread reversion absent)")
        verdict_fail = dict(
            n_syms_in_matrix=n_syms_in_matrix,
            n_candidates=n_candidates,
            n_cointegrated=0,
            verdict_s7a="FAIL",
            verdict_s7b="FAIL",
            reason="no cointegrated pairs at ADF p<=0.05 after correct universe construction",
        )
        verdict_path = os.path.join(OUT, "verdict.json")
        with open(verdict_path, "w") as f:
            json.dump(verdict_fail, f, indent=2, default=str)
        print(f"  verdict -> {verdict_path}")
        return

    # ── Stage 4: Rolling WFO per pair ────────────────────────────────────────
    print(f"\n=== STAGE 4: Rolling WFO on {n_cointegrated} cointegrated pairs ===")
    print(f"  WFO: {N_WFO_FOLDS} folds, IS={WFO_IS_BARS} bars (~2.3yr), "
          f"OOS={WFO_OOS_BARS} bars (~7mo)")
    print(f"  Extended PT/SL grid: {len(PT_GRID)} points from "
          f"{PT_GRID[0]:.2f} to {PT_GRID[-1]:.2f}")

    pair_results = []
    n_wfo_ok = 0
    for i, cp in enumerate(coint_pairs):
        if (i + 1) % HEARTBEAT == 0:
            print(f"  WFO: {i+1}/{n_cointegrated} pairs | "
                  f"{n_wfo_ok} usable | RAM {_ram_gb():.2f} GB")
        res, msg = run_pair_wfo(cp["sym_a"], cp["sym_b"], cp)
        if res is None:
            continue
        pair_results.append(res)
        n_wfo_ok += 1

    print(f"\n  WFO complete: {n_wfo_ok}/{n_cointegrated} pairs produced usable windows")
    if not pair_results:
        print("No pairs produced WFO results"); return

    # ── Stage 5: Book assembly ────────────────────────────────────────────────
    print("\n=== STAGE 5: Portfolio book assembly (inv-vol sizing) ===")
    book_r, book_meta = build_book(pair_results, sizing="inv_vol")

    if len(book_r) == 0:
        print("Empty book"); return

    sc_book = score_series(book_r)
    print(f"  Book: SR_ann={sc_book['sr_ann']:.3f} | PF={sc_book['pf']:.3f} | "
          f"n_trades={book_meta['n_trades']}")

    # ── Tail guard ─────────────────────────────────────────────────────────────
    tail = compute_tail_guard(pair_results)
    print(f"\n  Tail guard: worst-window RRR={tail['worst_window_rrr']:.3f} "
          f"(window {tail['worst_window']}) | "
          f"ex-worst RRR={tail['rrr_ex_worst']:.3f} | passes={tail['passes']}")

    # ── Grid corner check ─────────────────────────────────────────────────────
    grid_check = check_grid_corners(pair_results)
    print(f"\n  Grid corners: median pt*={grid_check['median_pt_star']:.3f} "
          f"sl*={grid_check['median_sl_star']:.3f} | "
          f"{grid_check['frac_pt_at_old_min_0_25']*100:.0f}% at old 0.25 min | "
          f"optimum_moved_interior={grid_check['optimum_moved_interior']}")

    # ── Per-trade ledger ──────────────────────────────────────────────────────
    all_trades_list = [pr["_trades_df"] for pr in pair_results]
    ledger = pd.concat(all_trades_list, ignore_index=True)
    ledger_cols = [c for c in ledger.columns if not c.startswith("_")]
    ledger = ledger[ledger_cols]
    ledger_path = os.path.join(OUT, "trades_ledger.parquet")
    ledger.to_parquet(ledger_path, index=False)
    print(f"  Trade ledger: {len(ledger)} trades -> {ledger_path}")

    # ── Per-pair table ────────────────────────────────────────────────────────
    pair_rows = []
    for pr in pair_results:
        ws_arr = pr["window_stats"]
        sr_vals  = [w["sr_ann"] for w in ws_arr]
        pf_vals  = [w["pf"]     for w in ws_arr]
        rrr_vals = [w["rrr"]    for w in ws_arr]
        pair_rows.append(dict(
            sym_a=pr["sym_a"], sym_b=pr["sym_b"],
            n_windows=len(ws_arr),
            n_trades=sum(w["n_trades"] for w in ws_arr),
            med_sr_ann=float(np.nanmedian(sr_vals)),
            med_pf=float(np.nanmedian(pf_vals)),
            med_rrr=float(np.nanmedian(rrr_vals)),
            min_rrr=float(np.nanmin(rrr_vals)) if rrr_vals else np.nan,
            med_adf_p=float(np.nanmedian([w["adf_p"] for w in ws_arr])),
            med_pt_star=float(np.nanmedian([w["pt_star"] for w in ws_arr])),
            med_sl_star=float(np.nanmedian([w["sl_star"] for w in ws_arr])),
        ))
    pair_df = pd.DataFrame(pair_rows)
    pair_table_parquet = os.path.join(OUT, "pair_table.parquet")
    pair_table_csv     = os.path.join(OUT, "pair_table.csv")
    pair_df.to_parquet(pair_table_parquet, index=False)
    pair_df.to_csv(pair_table_csv, index=False)
    print(f"  Pair table: {pair_table_parquet}")

    # ── Stage 6: Permutation null ─────────────────────────────────────────────
    n_grid_trials = len(PT_GRID) * len(SL_GRID) * n_cointegrated
    # Attach weights from inv_vol book to trades
    for pr in pair_results:
        _, bm_tmp = build_book([pr], sizing="inv_vol")
        bm_trades = bm_tmp.get("_trades", pd.DataFrame())
        if "weight" in bm_trades.columns:
            pr["_trades_df"]["weight"] = bm_trades["weight"].values

    p_val, null_srs, sr_obs_perm = permutation_null(
        pair_results, N_PERM, n_grid_trials)

    # ── Stage 7: Sign-consistency ─────────────────────────────────────────────
    half_a, half_b, sign_pass = sign_consistency(pair_results)

    # ── Stage 8: S7b sizing comparison ───────────────────────────────────────
    print("\n=== STAGE 8: S7b -- sizing comparison ===")
    sz_results = sizing_comparison(pair_results)
    for sz, sr in sz_results.items():
        print(f"  {sz:12s}: SR_ann={sr['sr_ann']:.3f} | PF={sr['pf']:.3f} | "
              f"RRR_ex_worst={sr['rrr_ex_worst']:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_path = make_figures(pair_results, book_r, null_srs, sr_obs_perm)

    # ── Verdict ───────────────────────────────────────────────────────────────
    criterion = dict(
        net_positive      = bool(sc_book["sr_ann"] > 0 and sc_book["pf"] > 1.0),
        tail_guard_passes = bool(tail["passes"]),
        deflated_sig      = bool(p_val < 0.05),
        sign_consistent   = bool(sign_pass),
        grid_interior     = bool(grid_check["optimum_moved_interior"]),
    )
    n_pass = sum(criterion.values())
    verdict_s7a = "PASS" if all(criterion.values()) else (
        "FAIL" if n_pass <= 2 else "MARGINAL/FAIL")

    s7b_inv_beats_equal = bool(
        sz_results["inv_vol"]["rrr_ex_worst"] > sz_results["equal"]["rrr_ex_worst"])
    verdict_s7b = "PASS" if s7b_inv_beats_equal else "FAIL"

    elapsed   = time.perf_counter() - t_total
    ram_peak  = _ram_gb()

    verdict = dict(
        # Counts
        n_usdt_syms       = len(usdt_syms),
        n_syms_in_matrix  = n_syms_in_matrix,
        n_candidates      = n_candidates,
        n_cointegrated    = n_cointegrated,
        n_pairs_wfo       = n_wfo_ok,
        n_trades_total    = int(book_meta["n_trades"]),
        # Book stats (inv-vol sizing)
        book_sr_ann       = float(sc_book["sr_ann"]),
        book_pf           = float(sc_book["pf"]),
        book_sr_pb        = float(sc_book["sr_per_bar"]),
        # Tail guard
        rrr_worst_window  = float(tail["worst_window_rrr"]),
        rrr_ex_worst      = float(tail["rrr_ex_worst"]),
        tail_guard_pass   = bool(tail["passes"]),
        # Permutation null
        perm_p_value      = float(p_val),
        perm_sr_obs       = float(sr_obs_perm),
        perm_null_mean    = float(null_srs.mean()),
        perm_null_std     = float(null_srs.std()),
        n_perm            = N_PERM,
        n_trials_deflated = int(n_grid_trials),
        # Sign consistency
        sign_half_a       = half_a,
        sign_half_b       = half_b,
        sign_consistent   = bool(sign_pass),
        # Grid
        grid_check        = grid_check,
        # S7b sizing
        sizing_equal      = sz_results["equal"],
        sizing_inv_vol    = sz_results["inv_vol"],
        sizing_conf       = sz_results["confidence"],
        s7b_inv_beats_equal = bool(s7b_inv_beats_equal),
        # Criterion breakdown
        criterion_s7a     = criterion,
        # Verdicts
        verdict_s7a       = verdict_s7a,
        verdict_s7b       = verdict_s7b,
        # Run metadata
        elapsed_s         = float(elapsed),
        ram_peak_gb       = float(ram_peak),
        corr_threshold    = CORR_THRESHOLD,
        corr_top_k        = CORR_TOP_K,
        corr_min_coverage = CORR_MIN_COVERAGE_FRAC,
        adf_p_max         = ADF_P_MAX,
        cost_bp_rt        = float(4 * COST_BP_PER_LEG),
        n_wfo_folds       = N_WFO_FOLDS,
        pt_grid_min       = float(PT_GRID[0]),
        pt_grid_max       = float(PT_GRID[-1]),
        n_grid_cells      = int(len(PT_GRID) * len(SL_GRID)),
        # Artifacts
        artifacts         = dict(
            ledger    = ledger_path,
            pairs     = pair_table_parquet,
            pairs_csv = pair_table_csv,
            figure    = fig_path,
        ),
    )

    verdict_path = os.path.join(OUT, "verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\n  verdict -> {verdict_path}")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    print(f"  Universe  : {len(usdt_syms)} USDT perps -> {n_syms_in_matrix} in matrix "
          f"-> {n_candidates} candidates -> {n_cointegrated} cointegrated "
          f"-> {n_wfo_ok} WFO-ok")
    print(f"  Book      : SR_ann={sc_book['sr_ann']:.3f} | PF={sc_book['pf']:.3f} "
          f"| n_trades={book_meta['n_trades']}")
    print(f"  Tail guard: ex-worst RRR={tail['rrr_ex_worst']:.3f} | "
          f"passes={tail['passes']}")
    print(f"  Perm null : p={p_val:.4f} | observed SR={sr_obs_perm:.4f} | "
          f"n_perm={N_PERM} | n_trials={n_grid_trials:,}")
    print(f"  Sign cons : A-M={half_a['net_positive']} N-Z={half_b['net_positive']} | "
          f"both={sign_pass}")
    print(f"  Grid      : median pt*={grid_check['median_pt_star']:.3f} | "
          f"interior={grid_check['optimum_moved_interior']}")
    print(f"  Criterion : {criterion}")
    print(f"\n  S7a: {verdict_s7a}")
    print(f"  S7b: {verdict_s7b} "
          f"(inv_vol RRR_ex_worst={sz_results['inv_vol']['rrr_ex_worst']:.3f} vs "
          f"equal={sz_results['equal']['rrr_ex_worst']:.3f})")
    print(f"\n  Elapsed: {elapsed/60:.1f} min | RAM peak: {ram_peak:.2f} GB")
    print("=" * 72)

    return verdict


if __name__ == "__main__":
    main()
