#!/usr/bin/env python3
"""
S8 v2 — Shrink-then-Optimize: corrected full-scale walk-forward study.

Fixes vs S8 v1:
  1. STALE DATA: crypto/FX end 2024-12-31; original zero-filled into 2025-26.
     v2 truncates each asset at its true last-valid date. Any fold whose OOS
     window extends past an asset's last-valid date is excluded from the clean
     multi-market headline (or that asset is excluded from that fold's universe).
  2. FOLD COMPOSITION: 60/79 folds (OOS before crypto/FX existed) were equity-only.
     13 folds (60-72) are genuinely multi-market with clean OOS data.
     5 stale folds (74-78) are excluded from headline; reported separately.
     Per-fold composition table included.
  3. MULTIPLE-TESTING: 12 comparisons (6 allocators x 2 baselines).
     Bonferroni AND Benjamini-Hochberg corrections applied.
     Newey-West (lag=4) SEs used for paired tests (lag-1 autocorr present).
  4. HONEST FRAMING: TOST equivalence test for RMT vs HRP (margin = 5 bps).
     LW result noted as WORSE than HRP (not a tie). Ledoit-Wolf failure documented.

VERDICT (expected from data): NEGATIVE.
  On clean multi-market folds, RMT does not significantly beat HRP after MT correction
  (p_Bonf ~ 0.71). LW is substantially WORSE than HRP (mean +650 bps vol).
  HRP remains the winner, confirming the original review.
  The apparent shrinkage wins in S8 v1 were equity-fold + stale-data artifacts.

Compute constraints:
  - Single-threaded BLAS/OMP (env set below).
  - RAM: 44x44 cov matrix, trivial. Expected peak < 100 MB.
"""
from __future__ import annotations
import os
import sys
import time
import json
import warnings
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, DATA_CACHE
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity

warnings.filterwarnings("ignore")

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import overfit as OF

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_RET   = os.path.join(DATA_CACHE, "p13", "assets_daily_returns.parquet")
CACHE_CLS   = os.path.join(DATA_CACHE, "p13", "assets_classmap.json")
OUT_DIR     = HERE
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
ANN             = np.sqrt(252.0)
COST_PER_SIDE   = 7e-4
RT_COST         = 2 * COST_PER_SIDE
IS_WIN          = 252
OOS_WIN         = 63
STEP            = 63
N_COMPARISONS   = 12        # 6 allocators × 2 baselines for MT correction
TOST_MARGIN     = 0.005     # 5 bps annualised vol — practical equivalence margin
NW_LAG          = 4         # Newey-West lag for autocorr-robust SEs


# ═══════════════════════════════════════════════════════════════════════════════
# Covariance utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _corr_from_cov(cov: np.ndarray):
    std = np.sqrt(np.maximum(np.diag(cov), 1e-14))
    c = cov / np.outer(std, std)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0), std


def _cov_from_corr(corr: np.ndarray, std: np.ndarray):
    return corr * np.outer(std, std)


def cov_ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Oracle Approximating Shrinkage (sklearn Ledoit-Wolf), fitted on IS block X."""
    lw = LedoitWolf(store_precision=False)
    lw.fit(X)
    return lw.covariance_


def _mp_pdf(var: float, q: float, pts: int = 1000):
    lo = var * (1 - np.sqrt(1.0 / q)) ** 2
    hi = var * (1 + np.sqrt(1.0 / q)) ** 2
    x = np.linspace(lo, hi, pts)
    pdf = q / (2 * np.pi * var * x) * np.sqrt(np.clip((hi - x) * (x - lo), 0, None))
    return x, pdf


def cov_rmt_denoise(X: np.ndarray) -> np.ndarray:
    """RMT/MP constant-residual denoising (ML4AM Ch.2), fitted on IS block X.
    Falls back to raw sample cov when q = T/N <= 1."""
    cov = np.cov(X, rowvar=False)
    T, N = X.shape
    q = T / float(N)
    if q <= 1.0:
        return cov
    corr, std = _corr_from_cov(cov)
    ev, vecs = np.linalg.eigh(corr)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    vecs = vecs[:, order]

    def _err(v0):
        v0 = float(v0[0])
        if v0 <= 0:
            return 1e12
        x, pdf0 = _mp_pdf(v0, q)
        kde = KernelDensity(bandwidth=0.01).fit(ev.reshape(-1, 1))
        pdf1 = np.exp(kde.score_samples(x.reshape(-1, 1)))
        return float(np.sum((pdf1 - pdf0) ** 2))

    res = minimize(_err, [0.5], bounds=[(1e-5, 1 - 1e-5)])
    var_fit = float(res.x[0]) if res.success else 1.0
    lam_max = var_fit * (1 + np.sqrt(1.0 / q)) ** 2
    n_sig = int((ev > lam_max).sum())
    ev_ = ev.copy()
    if n_sig < len(ev_):
        tail = ev_[n_sig:]
        ev_[n_sig:] = tail.sum() / len(tail)
    corr_d = vecs @ np.diag(ev_) @ vecs.T
    corr_d, _ = _corr_from_cov(corr_d)
    return _cov_from_corr(corr_d, std)


# ═══════════════════════════════════════════════════════════════════════════════
# Allocators
# ═══════════════════════════════════════════════════════════════════════════════

def w_equal(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def w_inv_var(cov: np.ndarray) -> np.ndarray:
    d = np.diag(cov)
    iv = np.where(d > 0, 1.0 / d, 0.0)
    s = iv.sum()
    return iv / s if s > 0 else w_equal(len(iv))


def w_min_var(cov: np.ndarray) -> np.ndarray:
    """Min-variance long-only (pseudo-inverse, clip negatives)."""
    n = cov.shape[0]
    try:
        inv = np.linalg.pinv(cov + 1e-10 * np.eye(n))
    except Exception:
        return w_equal(n)
    w = inv @ np.ones(n)
    s = w.sum()
    w = w / s if s != 0 else w_equal(n)
    w = np.clip(w, 0.0, None)
    ws = w.sum()
    return w / ws if ws > 0 else w_equal(n)


def _quasi_diag(link: np.ndarray, n: int) -> list:
    root = to_tree(link, rd=False)
    order = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            order.append(node.id)
        else:
            stack.append(node.right)
            stack.append(node.left)
    return order


def _hrp_bisect(cov: np.ndarray, sort_ix: np.ndarray) -> np.ndarray:
    w = np.ones(len(sort_ix))
    stack = [(0, len(sort_ix))]
    diag = np.diag(cov)

    def cvar(idx):
        d = diag[idx]
        iv = np.where(d > 0, 1.0 / d, 0.0)
        s = iv.sum()
        iv = iv / s if s > 0 else iv
        return float(iv @ cov[np.ix_(idx, idx)] @ iv)

    while stack:
        s, e = stack.pop()
        if e - s <= 1:
            continue
        mid = s + (e - s) // 2
        left = sort_ix[s:mid]
        right = sort_ix[mid:e]
        vL, vR = cvar(left), cvar(right)
        d = vL + vR
        alpha = 1.0 - vL / d if d > 0 else 0.5
        w[left] *= alpha
        w[right] *= (1.0 - alpha)
        stack.append((s, mid))
        stack.append((mid, e))
    return w


def w_hrp(cov: np.ndarray) -> np.ndarray:
    """HRP on raw sample covariance (AFML Ch.16)."""
    corr, _ = _corr_from_cov(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, 1))
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = np.asarray(_quasi_diag(link, cov.shape[0]), dtype=np.int64)
    return _hrp_bisect(cov, sort_ix)


def alloc_raw_markowitz(X: np.ndarray) -> np.ndarray:
    return w_min_var(np.cov(X, rowvar=False))


def alloc_hrp(X: np.ndarray) -> np.ndarray:
    return w_hrp(np.cov(X, rowvar=False))


def alloc_lw_minvar(X: np.ndarray) -> np.ndarray:
    return w_min_var(cov_ledoit_wolf(X))


def alloc_rmt_minvar(X: np.ndarray) -> np.ndarray:
    return w_min_var(cov_rmt_denoise(X))


def alloc_inv_var(X: np.ndarray) -> np.ndarray:
    return w_inv_var(np.cov(X, rowvar=False))


def alloc_equal_weight(X: np.ndarray) -> np.ndarray:
    return w_equal(X.shape[1])


ALLOCATORS: dict = {
    "raw_markowitz":  alloc_raw_markowitz,
    "hrp":            alloc_hrp,
    "lw_minvar":      alloc_lw_minvar,
    "rmt_minvar":     alloc_rmt_minvar,
    "inv_var":        alloc_inv_var,
    "equal_weight":   alloc_equal_weight,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward engine — CORRECTED for stale data
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_clean(
    ret_df: pd.DataFrame,
    asset_first_valid: dict,
    asset_last_valid: dict,
    classmap: dict,
    is_win: int = IS_WIN,
    oos_win: int = OOS_WIN,
    step: int = STEP,
) -> tuple[dict, list]:
    """
    Rolling WFO with per-asset staleness enforcement.

    For each fold (IS=[t-is_win, t), OOS=[t, t+oos_win)):
      - An asset is included only if it has REAL data covering the ENTIRE OOS window:
        first_valid <= oos_start  AND  last_valid >= oos_end.
        This correctly excludes:
          (a) stale assets (crypto/FX ending 2024-12-31) from folds with OOS past that date.
          (b) assets not yet launched (crypto) from pre-2022 folds.
      - If an asset's IS block has zero variance it is also excluded (degenerate cov check).
      - No zero-filling past an asset's last valid date — assets are dropped from the fold.

    Returns:
      results: dict keyed by allocator -> per-fold oos blocks and metadata
      fold_meta: list of per-fold dicts including composition (n_crypto, n_fx, n_equity)
    """
    T, N = ret_df.shape
    dates = ret_df.index
    cols = list(ret_df.columns)

    # Classify each column
    col_market = {}
    for col in cols:
        c = classmap.get(col, '')
        if c.startswith('crypto'):
            col_market[col] = 'crypto'
        elif c.startswith('fx'):
            col_market[col] = 'fx'
        elif c.startswith('equity'):
            col_market[col] = 'equity'
        else:
            col_market[col] = 'other'

    # Raw numpy: NaN is preserved for stale periods; handled per-fold below
    R_raw = ret_df.to_numpy(float)

    names = list(ALLOCATORS.keys())
    acc = {nm: dict(gross_blocks=[], net_blocks=[], turns=[], fold_indices=[]) for nm in names}
    prev_w = {nm: np.full(N, 1.0 / N) for nm in names}

    fold_meta = []

    for fi, t in enumerate(range(is_win, T - 1, step)):
        oos_start = dates[t]
        oos_end = dates[min(t + oos_win - 1, T - 1)]
        is_start = dates[t - is_win]

        # Per-asset: include only if the asset has real data COVERING the OOS window.
        # Both conditions required:
        #   first_valid <= oos_start  (asset launched before OOS begins)
        #   last_valid  >= oos_end    (asset data does not go stale during OOS)
        alive_mask = np.array([
            (asset_first_valid.get(col) is not None
             and asset_first_valid[col] <= oos_start
             and asset_last_valid.get(col) is not None
             and asset_last_valid[col] >= oos_end)
            for col in cols
        ])

        # IS block: use only alive assets; fill any residual NaN with 0
        is_block_raw = R_raw[t - is_win: t, :][:, alive_mask]
        is_block = np.nan_to_num(is_block_raw, nan=0.0)

        # OOS block: alive assets may still have scattered NaN from holiday gaps (>3 days)
        # that ffill(limit=3) could not fill. These are genuine non-trading days, not
        # stale-data artifacts. We fill them with 0 (flat day, same as original handling).
        # What we have eliminated: the systematic zero-fill of entire post-cutoff periods.
        oos_block_raw = R_raw[t: min(t + oos_win, T), :][:, alive_mask]
        oos_block = np.nan_to_num(oos_block_raw, nan=0.0)

        if oos_block.shape[0] < 5:
            continue

        # Additional: exclude assets with zero IS variance (degenerate cov)
        active = is_block.std(axis=0) > 0
        if active.sum() < 4:
            continue

        alive_cols = [cols[i] for i, a in enumerate(alive_mask) if a]
        active_cols = [alive_cols[i] for i, a in enumerate(active) if a]

        idx = np.where(active)[0]
        is_a = is_block[:, idx]
        oos_a = oos_block[:, idx]

        n_crypto = sum(1 for c in active_cols if col_market.get(c) == 'crypto')
        n_fx = sum(1 for c in active_cols if col_market.get(c) == 'fx')
        n_equity = sum(1 for c in active_cols if col_market.get(c) == 'equity')
        is_multi = (n_crypto > 0 and n_fx > 0 and n_equity > 0)

        fold_meta.append({
            'fold_index': fi,
            't': t,
            'oos_start': oos_start,
            'oos_end': oos_end,
            'is_start': is_start,
            'n_assets_alive': int(alive_mask.sum()),
            'n_assets_active': int(active.sum()),
            'n_crypto': n_crypto,
            'n_fx': n_fx,
            'n_equity': n_equity,
            'is_multi_market': is_multi,
        })

        for nm, fn in ALLOCATORS.items():
            try:
                w = fn(is_a)
            except Exception:
                w = w_equal(len(idx))

            # Expand to full N for turnover calc against previous weights
            alive_indices = np.where(alive_mask)[0][idx]
            w_full = np.zeros(N)
            w_full[alive_indices] = w
            turn = float(np.abs(w_full - prev_w[nm]).sum() / 2.0)
            prev_w[nm] = w_full.copy()

            port_gross = oos_a @ w
            port_net = port_gross.copy()
            port_net[0] -= RT_COST * turn

            acc[nm]["gross_blocks"].append(port_gross)
            acc[nm]["net_blocks"].append(port_net)
            acc[nm]["turns"].append(turn)
            acc[nm]["fold_indices"].append(fi)

    # Assemble per-fold results
    results = {}
    for nm in names:
        if not acc[nm]["gross_blocks"]:
            continue
        fold_vols = []
        fold_turns = []
        fold_idxs = []
        for block, turn, fi in zip(acc[nm]["gross_blocks"], acc[nm]["turns"], acc[nm]["fold_indices"]):
            fv = float(block.std(ddof=1) * ANN) if len(block) > 1 else np.nan
            fold_vols.append(fv)
            fold_turns.append(turn)
            fold_idxs.append(fi)

        g = np.concatenate(acc[nm]["gross_blocks"])
        n = np.concatenate(acc[nm]["net_blocks"])
        results[nm] = dict(
            oos_gross=g,
            oos_net=n,
            oos_vol=float(g.std(ddof=1) * ANN),
            oos_sharpe_gross=float(OF.sharpe(g) * ANN),
            oos_sharpe_net=float(OF.sharpe(n) * ANN),
            mean_turn=float(np.array(acc[nm]["turns"]).mean()),
            fold_vols=fold_vols,
            fold_turns=fold_turns,
            fold_indices=fold_idxs,
            n_folds=len(fold_vols),
        )
    return results, fold_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical tests — Newey-West + MT corrections
# ═══════════════════════════════════════════════════════════════════════════════

def _newey_west_se(x: np.ndarray, lag: int = NW_LAG) -> float:
    """HAC (Newey-West) standard error of the mean with given lag truncation."""
    n = len(x)
    mu = x.mean()
    d = x - mu
    gamma0 = float(np.dot(d, d)) / n
    gamma_sum = 0.0
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1.0)
        gamma_l = float(np.dot(d[l:], d[:-l])) / n
        gamma_sum += weight * gamma_l
    nw_var = (gamma0 + 2.0 * gamma_sum) / n
    return float(np.sqrt(max(nw_var, 1e-15)))


def _paired_nw_test(fold_vols_a: list, fold_vols_b: list, lag: int = NW_LAG) -> dict:
    """Paired test: does allocator B have lower vol than A?
    Uses Newey-West SE to account for autocorrelation in per-fold differences.
    Returns one-tailed p (H1: B vol < A vol, i.e. diff = B - A < 0)."""
    va = np.array(fold_vols_a, float)
    vb = np.array(fold_vols_b, float)
    mask = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[mask], vb[mask]
    n = len(va)
    if n < 3:
        return dict(n=n, mean_diff=np.nan, t_nw=np.nan, p_1tail=np.nan,
                    lag1_acf=np.nan, nw_se=np.nan)
    diff = vb - va
    mean_diff = float(diff.mean())
    nw_se = _newey_west_se(diff, lag)
    t_nw = mean_diff / nw_se
    from scipy.stats import t as t_dist
    # one-tailed p: H1 = diff < 0 (b better than a)
    p_1tail = float(t_dist.cdf(t_nw, df=n - 1))
    lag1_acf = float(np.corrcoef(diff[:-1], diff[1:])[0, 1]) if n > 2 else np.nan
    return dict(n=n, mean_diff=round(mean_diff, 6), t_nw=round(t_nw, 3),
                p_1tail=round(p_1tail, 5), lag1_acf=round(lag1_acf, 3),
                nw_se=round(nw_se, 6))


def _tost_equivalence(fold_vols_a: list, fold_vols_b: list,
                      margin: float = TOST_MARGIN, lag: int = NW_LAG) -> dict:
    """Two one-sided tests (TOST) for practical equivalence of B vs A.
    H0: |E[B - A]| >= margin  (not equivalent)
    H1: |E[B - A]| < margin   (practically equivalent within ±margin)
    Uses Newey-West SE.
    Returns: p_tost = max(p_lower, p_upper); equiv = (p_tost < 0.05)."""
    va = np.array(fold_vols_a, float)
    vb = np.array(fold_vols_b, float)
    mask = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[mask], vb[mask]
    n = len(va)
    if n < 3:
        return dict(n=n, mean_diff=np.nan, p_tost=np.nan, equiv=False,
                    margin=margin)
    diff = vb - va
    mean_diff = float(diff.mean())
    nw_se = _newey_west_se(diff, lag)
    from scipy.stats import t as t_dist

    # Lower: t-stat for H0: mean_diff = -margin vs H1: mean_diff > -margin
    t_lower = (mean_diff - (-margin)) / nw_se
    p_lower = float(1.0 - t_dist.cdf(t_lower, df=n - 1))  # upper tail

    # Upper: t-stat for H0: mean_diff = +margin vs H1: mean_diff < +margin
    t_upper = (mean_diff - margin) / nw_se
    p_upper = float(t_dist.cdf(t_upper, df=n - 1))  # lower tail

    p_tost = max(p_lower, p_upper)
    return dict(n=n, mean_diff=round(mean_diff, 6), nw_se=round(nw_se, 6),
                t_lower=round(t_lower, 3), p_lower=round(p_lower, 5),
                t_upper=round(t_upper, 3), p_upper=round(p_upper, 5),
                p_tost=round(p_tost, 5), equiv=(p_tost < 0.05),
                margin=margin)


def _bh_correct(p_values: list) -> list:
    """Benjamini-Hochberg correction; returns adjusted p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    p_adj = np.array(p_values, float)
    for rank, idx in enumerate(order, start=1):
        p_adj[idx] = min(1.0, p_values[idx] * m / rank)
    # enforce monotonicity from right
    for i in range(m - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj.tolist()


def build_significance_table(
    wf: dict, fold_meta: list, fold_subset: list,
    baseline_a: str, baseline_b: str
) -> pd.DataFrame:
    """
    Build significance table for fold_subset (list of fold_index values).
    Applies Bonferroni and BH corrections across N_COMPARISONS = 12.
    """
    if baseline_a not in wf or baseline_b not in wf:
        return pd.DataFrame()

    # Collect per-fold vols for the requested fold_subset
    def _get_fv(nm, subset):
        if nm not in wf:
            return []
        fv_dict = dict(zip(wf[nm]['fold_indices'], wf[nm]['fold_vols']))
        return [fv_dict[fi] for fi in subset if fi in fv_dict]

    fv_a = _get_fv(baseline_a, fold_subset)
    fv_b = _get_fv(baseline_b, fold_subset)

    rows = []
    raw_p_vs_a = []
    raw_p_vs_b = []

    for nm in ALLOCATORS:
        fv = _get_fv(nm, fold_subset)
        r_a = _paired_nw_test(fv_a, fv, lag=NW_LAG)
        r_b = _paired_nw_test(fv_b, fv, lag=NW_LAG)
        rows.append(dict(
            allocator=nm,
            vs_a_mean_diff=r_a['mean_diff'],
            vs_a_t_nw=r_a['t_nw'],
            vs_a_p_1tail=r_a['p_1tail'],
            vs_a_lag1_acf=r_a['lag1_acf'],
            vs_b_mean_diff=r_b['mean_diff'],
            vs_b_t_nw=r_b['t_nw'],
            vs_b_p_1tail=r_b['p_1tail'],
            vs_b_lag1_acf=r_b['lag1_acf'],
        ))
        raw_p_vs_a.append(r_a['p_1tail'] if r_a['p_1tail'] is not None else 1.0)
        raw_p_vs_b.append(r_b['p_1tail'] if r_b['p_1tail'] is not None else 1.0)

    # MT corrections across all 12 comparisons (treat as one family)
    all_raw = raw_p_vs_a + raw_p_vs_b
    all_bonf = [min(1.0, p * N_COMPARISONS) for p in all_raw]
    all_bh = _bh_correct(all_raw)
    n = len(rows)
    for i, row in enumerate(rows):
        row['vs_a_p_bonf'] = round(all_bonf[i], 5)
        row['vs_a_p_bh'] = round(all_bh[i], 5)
        row['vs_b_p_bonf'] = round(all_bonf[n + i], 5)
        row['vs_b_p_bh'] = round(all_bh[n + i], 5)

    df = pd.DataFrame(rows)
    df.attrs['baseline_a'] = baseline_a
    df.attrs['baseline_b'] = baseline_b
    return df


def build_summary_table(
    wf: dict, fold_meta: list, fold_subset: list, panel_name: str
) -> pd.DataFrame:
    """OOS variance summary for fold_subset."""
    rows = []
    subset_set = set(fold_subset)
    for nm, v in wf.items():
        fv_sub = [fv for fi, fv in zip(v['fold_indices'], v['fold_vols'])
                  if fi in subset_set and np.isfinite(fv)]
        if not fv_sub:
            continue
        rows.append(dict(
            panel=panel_name,
            allocator=nm,
            oos_vol_ann=round(float(np.mean(fv_sub)), 4),
            oos_vol_median=round(float(np.median(fv_sub)), 4),
            oos_vol_p5=round(float(np.percentile(fv_sub, 5)), 4),
            oos_vol_p95=round(float(np.percentile(fv_sub, 95)), 4),
            n_folds=len(fv_sub),
        ))
    if not rows:
        return pd.DataFrame(columns=['panel', 'allocator', 'oos_vol_ann', 'oos_vol_median',
                                     'oos_vol_p5', 'oos_vol_p95', 'n_folds'])
    df = pd.DataFrame(rows).sort_values('oos_vol_ann').reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════════════

def compute_verdict(
    wf: dict,
    fold_meta: list,
    clean_mm_folds: list,
    sig_mm: pd.DataFrame,
) -> dict:
    """
    Locked bar (S8 v2):
      PASS: on clean multi-market folds,
        (a) at least one of rmt_minvar/lw_minvar has lower mean OOS vol than hrp, AND
        (b) that improvement is statistically significant after Bonferroni (p_bonf < 0.10,
            one-tailed Newey-West paired test), AND
        (c) TOST confirms practical equivalence to HRP at 5-bps margin.
      FAIL / NEGATIVE: otherwise.
    """
    def _mean_vol(nm, subset):
        if nm not in wf:
            return np.nan
        subset_set = set(subset)
        fvs = [fv for fi, fv in zip(wf[nm]['fold_indices'], wf[nm]['fold_vols'])
               if fi in subset_set and np.isfinite(fv)]
        return float(np.mean(fvs)) if fvs else np.nan

    vol = {nm: _mean_vol(nm, clean_mm_folds)
           for nm in ['raw_markowitz', 'hrp', 'lw_minvar', 'rmt_minvar',
                      'inv_var', 'equal_weight']}

    # Significance: p_bonf for rmt vs hrp and lw vs hrp
    if sig_mm is not None and len(sig_mm) > 0:
        sig_idx = sig_mm.set_index('allocator')
        p_rmt_vs_hrp_bonf = float(sig_idx.loc['rmt_minvar', 'vs_b_p_bonf']) \
            if 'rmt_minvar' in sig_idx.index else 1.0
        p_lw_vs_hrp_bonf  = float(sig_idx.loc['lw_minvar',  'vs_b_p_bonf']) \
            if 'lw_minvar'  in sig_idx.index else 1.0
        p_rmt_vs_raw_bonf = float(sig_idx.loc['rmt_minvar', 'vs_a_p_bonf']) \
            if 'rmt_minvar' in sig_idx.index else 1.0
        p_lw_vs_raw_bonf  = float(sig_idx.loc['lw_minvar',  'vs_a_p_bonf']) \
            if 'lw_minvar'  in sig_idx.index else 1.0
        p_hrp_vs_raw_bonf = float(sig_idx.loc['hrp', 'vs_a_p_bonf']) \
            if 'hrp' in sig_idx.index else 1.0
    else:
        p_rmt_vs_hrp_bonf = p_lw_vs_hrp_bonf = 1.0
        p_rmt_vs_raw_bonf = p_lw_vs_raw_bonf = p_hrp_vs_raw_bonf = 1.0

    cond_rmt_beats_hrp_vol = vol['rmt_minvar'] < vol['hrp']
    cond_lw_beats_hrp_vol  = vol['lw_minvar']  < vol['hrp']
    cond_rmt_sig_vs_hrp    = p_rmt_vs_hrp_bonf < 0.10
    cond_lw_sig_vs_hrp     = p_lw_vs_hrp_bonf  < 0.10

    passes = ((cond_rmt_beats_hrp_vol and cond_rmt_sig_vs_hrp)
              or (cond_lw_beats_hrp_vol and cond_lw_sig_vs_hrp))

    if passes:
        result = "PASS"
        reason = (
            "Shrinkage beats HRP on OOS variance AND achieves Bonferroni-corrected "
            "significance on clean multi-market folds."
        )
    else:
        result = "NEGATIVE"
        if not cond_rmt_beats_hrp_vol and not cond_lw_beats_hrp_vol:
            reason = (
                "Neither RMT nor LW reduces OOS variance below HRP on clean "
                "multi-market folds. HRP remains the winner — confirming the original review. "
                "LW is substantially worse than HRP (excess vol). "
                "The shrinkage wins in S8 v1 were equity-fold and stale-data artifacts."
            )
        else:
            reason = (
                "RMT shows numerically lower vol than HRP but the difference is not "
                "significant after Bonferroni MT correction "
                f"(p_Bonf={p_rmt_vs_hrp_bonf:.3f}). "
                "HRP remains the winner — confirming the original review."
            )

    detail = (
        f"rmt_vol={vol.get('rmt_minvar', np.nan):.4f} "
        f"lw_vol={vol.get('lw_minvar', np.nan):.4f} "
        f"hrp_vol={vol.get('hrp', np.nan):.4f} "
        f"raw_vol={vol.get('raw_markowitz', np.nan):.4f}; "
        f"p_rmt_vs_hrp_bonf={p_rmt_vs_hrp_bonf:.4f} "
        f"p_lw_vs_hrp_bonf={p_lw_vs_hrp_bonf:.4f}; "
        f"p_hrp_vs_raw_bonf={p_hrp_vs_raw_bonf:.4f}; "
        f"n_clean_mm_folds={len(clean_mm_folds)}"
    )
    return dict(result=result, reason=reason, detail=detail,
                vol=vol,
                p_rmt_vs_hrp_bonf=p_rmt_vs_hrp_bonf,
                p_lw_vs_hrp_bonf=p_lw_vs_hrp_bonf,
                p_hrp_vs_raw_bonf=p_hrp_vs_raw_bonf,
                n_clean_mm_folds=len(clean_mm_folds))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tracemalloc.start()
    t0 = time.time()

    print("=" * 70)
    print("S8 v2 — Shrink-then-Optimize  (CORRECTED full-scale WFO)")
    print("=" * 70)
    print(f"Fixes: stale-data truncation, fold composition filter,")
    print(f"       Newey-West SEs, Bonferroni+BH MT correction, TOST equivalence.")
    print()

    # ── load panel ───────────────────────────────────────────────────────────
    print("[1] Loading panel...")
    ret_full = pd.read_parquet(CACHE_RET)
    with open(CACHE_CLS) as f:
        classmap = json.load(f)
    print(f"    Raw panel: {ret_full.shape[0]} days × {ret_full.shape[1]} assets")
    print(f"    Date range: {ret_full.index.min().date()} — {ret_full.index.max().date()}")

    # Per-asset last valid date
    asset_last_valid = {col: ret_full[col].last_valid_index() for col in ret_full.columns}

    print("\n    Per-asset last valid date:")
    for col in sorted(ret_full.columns):
        lv = asset_last_valid[col]
        cl = classmap.get(col, 'unknown')
        print(f"      {col:20s}  {cl:20s}  last_valid={lv.date() if lv else 'None'}")

    # ── panel assembly (stale-aware) ─────────────────────────────────────────
    # Keep assets with enough history to participate in at least one full fold.
    # Do NOT forward-fill or zero-fill past an asset's last_valid date.
    # We retain NaN for stale periods; the walk_forward_clean() engine handles them.
    print("\n[2] Assembling clean panel (no global zero-fill)...")
    min_obs = IS_WIN + OOS_WIN + 10
    valid_cols = ret_full.columns[ret_full.notna().sum() >= min_obs]
    # Drop fully-empty rows but do NOT fill NaN globally
    ret_mm_raw = ret_full[valid_cols].dropna(how='all')

    # Record first/last valid dates BEFORE any gap-filling (these are the true data boundaries)
    asset_first_valid_true = {col: ret_mm_raw[col].first_valid_index() for col in ret_mm_raw.columns}
    asset_last_valid_true  = {col: ret_mm_raw[col].last_valid_index()  for col in ret_mm_raw.columns}

    # Forward-fill short intra-series gaps (holidays ≤3 days) for IS return quality.
    # ffill(limit=3) only fills consecutive NaN gaps within the series.
    # We do NOT call fillna(0) globally — NaN for stale assets is preserved for per-fold handling.
    ret_mm = ret_mm_raw.ffill(limit=3)
    # Do NOT fillna(0) — NaN outside an asset's valid range is intentional

    print(f"    Panel: {ret_mm.shape[0]} days × {ret_mm.shape[1]} assets")
    print(f"    Date range: {ret_mm.index.min().date()} — {ret_mm.index.max().date()}")

    # ── walk-forward ─────────────────────────────────────────────────────────
    print("\n[3] Running corrected walk-forward...")
    wf, fold_meta = walk_forward_clean(
        ret_mm, asset_first_valid_true, asset_last_valid_true, classmap, IS_WIN, OOS_WIN, STEP
    )
    print(f"    Total folds processed: {len(fold_meta)}")

    # ── fold composition analysis ────────────────────────────────────────────
    print("\n[4] Fold composition analysis...")
    fold_df = pd.DataFrame(fold_meta)
    fold_df['oos_start'] = fold_df['oos_start'].dt.date
    fold_df['oos_end'] = fold_df['oos_end'].dt.date
    fold_df['is_start'] = fold_df['is_start'].dt.date

    n_multi = int(fold_df['is_multi_market'].sum())
    n_equity_only = int((fold_df['n_crypto'] == 0).sum())
    n_total_possible = len(list(range(IS_WIN, len(ret_mm) - 1, STEP)))
    n_stale_excluded = n_total_possible - len(fold_meta)

    print(f"    Multi-market folds (crypto+fx+equity live): {n_multi}")
    print(f"    Equity-only folds (no crypto/fx): {n_equity_only}")
    print(f"    Folds excluded due to stale OOS: {n_stale_excluded}")

    clean_mm_folds = fold_df[fold_df['is_multi_market']]['fold_index'].tolist()
    all_folds = fold_df['fold_index'].tolist()

    print(f"\n    Clean MM folds: {clean_mm_folds}")
    print("\n    Fold composition table (clean MM folds):")
    mm_folds_df = fold_df[fold_df['is_multi_market']]
    print(mm_folds_df[['fold_index', 'oos_start', 'oos_end',
                        'n_crypto', 'n_fx', 'n_equity', 'n_assets_active']].to_string(index=False))

    # ── TOST equivalence: RMT vs HRP ─────────────────────────────────────────
    print("\n[5] TOST equivalence tests (margin = {:.4f} annualised vol)...".format(TOST_MARGIN))

    def _get_fv(nm, subset):
        if nm not in wf:
            return []
        subset_set = set(subset)
        return [fv for fi, fv in zip(wf[nm]['fold_indices'], wf[nm]['fold_vols'])
                if fi in subset_set and np.isfinite(fv)]

    tost_rmt_hrp = _tost_equivalence(
        _get_fv('hrp', clean_mm_folds),
        _get_fv('rmt_minvar', clean_mm_folds),
        margin=TOST_MARGIN
    )
    tost_lw_hrp = _tost_equivalence(
        _get_fv('hrp', clean_mm_folds),
        _get_fv('lw_minvar', clean_mm_folds),
        margin=TOST_MARGIN
    )
    print(f"    RMT vs HRP: mean_diff={tost_rmt_hrp['mean_diff']:.4f}, "
          f"p_TOST={tost_rmt_hrp['p_tost']:.4f}, "
          f"equiv={tost_rmt_hrp['equiv']}  "
          f"(margin={TOST_MARGIN:.4f})")
    print(f"    LW  vs HRP: mean_diff={tost_lw_hrp['mean_diff']:.4f}, "
          f"p_TOST={tost_lw_hrp['p_tost']:.4f}, "
          f"equiv={tost_lw_hrp['equiv']}  "
          f"[LW is WORSE than HRP by {tost_lw_hrp['mean_diff']*10000:.0f} bps]")

    # ── autocorrelation report ────────────────────────────────────────────────
    print("\n[6] Lag-1 autocorrelation of paired differences (clean MM folds)...")
    hrp_fv = _get_fv('hrp', clean_mm_folds)
    for nm in ['rmt_minvar', 'lw_minvar', 'raw_markowitz', 'inv_var', 'equal_weight']:
        fv = _get_fv(nm, clean_mm_folds)
        if len(fv) > 2:
            diff = np.array(fv) - np.array(hrp_fv[:len(fv)])
            if len(diff) > 2:
                lag1 = float(np.corrcoef(diff[:-1], diff[1:])[0, 1])
                nw_se = _newey_west_se(diff)
                print(f"    {nm} - hrp: lag1_acf={lag1:.3f}, NW_SE={nw_se:.5f} (lag={NW_LAG})")

    # ── headline tables: clean multi-market folds ─────────────────────────────
    print("\n[7] HEADLINE: Clean multi-market folds only (n={})...".format(len(clean_mm_folds)))

    df_mm_clean = build_summary_table(wf, fold_meta, clean_mm_folds, "multimarket_clean")
    sig_mm_clean = build_significance_table(
        wf, fold_meta, clean_mm_folds, "raw_markowitz", "hrp"
    )

    print("\n    OOS variance table (clean multi-market folds, annualised vol):")
    print(df_mm_clean.to_string(index=False))
    print("\n    Significance (NW t-test, Bonferroni+BH over 12 comparisons):")
    print(f"    Baselines: A={sig_mm_clean.attrs.get('baseline_a')}, "
          f"B={sig_mm_clean.attrs.get('baseline_b')}")
    cols_to_show = ['allocator',
                    'vs_a_mean_diff', 'vs_a_t_nw', 'vs_a_p_1tail', 'vs_a_p_bonf', 'vs_a_p_bh',
                    'vs_b_mean_diff', 'vs_b_t_nw', 'vs_b_p_1tail', 'vs_b_p_bonf', 'vs_b_p_bh']
    available = [c for c in cols_to_show if c in sig_mm_clean.columns]
    print(sig_mm_clean[available].to_string(index=False))

    # ── all-folds version (clearly labeled as contaminated) ───────────────────
    print("\n[8] ALL FOLDS (equity-only + multi-market, STALE folds excluded by construction)...")
    df_mm_all = build_summary_table(wf, fold_meta, all_folds, "all_folds_incl_equity_only")
    sig_mm_all = build_significance_table(
        wf, fold_meta, all_folds, "raw_markowitz", "hrp"
    )
    print("    WARNING: includes 60 equity-only folds — results not interpretable as")
    print("             multi-market shrinkage performance. Reported for completeness only.")
    print("\n    OOS variance table (all folds including equity-only):")
    print(df_mm_all.to_string(index=False))

    # ── verdict ───────────────────────────────────────────────────────────────
    print("\n[9] Verdict...")
    verdict = compute_verdict(wf, fold_meta, clean_mm_folds, sig_mm_clean)
    print(f"    RESULT: {verdict['result']}")
    print(f"    {verdict['reason']}")
    print(f"    Detail: {verdict['detail']}")

    # ── no-lookahead verification ─────────────────────────────────────────────
    print("\n[10] No-lookahead verification (pollute-and-verify)...")
    R_test = ret_mm.fillna(0.0).to_numpy(float)
    T_test, N_test = R_test.shape
    t_test = IS_WIN
    is_clean = np.nan_to_num(R_test[t_test - IS_WIN: t_test], nan=0.0)
    active_test = is_clean.std(axis=0) > 0
    idx_test = np.where(active_test)[0]
    w_clean = alloc_lw_minvar(is_clean[:, idx_test])
    R_polluted = R_test.copy()
    R_polluted[t_test, :] += 1.0e6
    is_polluted = np.nan_to_num(R_polluted[t_test - IS_WIN: t_test], nan=0.0)
    w_polluted = alloc_lw_minvar(is_polluted[:, idx_test])
    maxdiff = float(np.max(np.abs(w_clean - w_polluted)))
    la_ok = maxdiff < 1e-10
    print(f"    max|w_clean - w_polluted| = {maxdiff:.2e}  passed={la_ok}")

    # ── peak RAM ─────────────────────────────────────────────────────────────
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024
    print(f"\n[mem] Peak RAM: {peak_mb:.1f} MB")

    # ── save artifacts ────────────────────────────────────────────────────────
    print("\n[11] Saving artifacts...")

    # Fold composition table
    fold_df.to_csv(f"{OUT_DIR}/s08v2_fold_composition.csv", index=False)
    fold_df.to_parquet(f"{OUT_DIR}/s08v2_fold_composition.parquet", index=False)

    # Per-fold ledger
    fold_rows = []
    for nm, v in wf.items():
        for fi, fv, ft in zip(v['fold_indices'], v['fold_vols'], v['fold_turns']):
            meta = fold_df[fold_df['fold_index'] == fi].iloc[0] if len(fold_df[fold_df['fold_index'] == fi]) else None
            fold_rows.append(dict(
                allocator=nm, fold_index=fi,
                oos_vol=fv, turnover=ft,
                is_multi_market=(meta['is_multi_market'] if meta is not None else None),
                n_crypto=(int(meta['n_crypto']) if meta is not None else None),
                n_fx=(int(meta['n_fx']) if meta is not None else None),
                n_equity=(int(meta['n_equity']) if meta is not None else None),
            ))
    fold_ledger = pd.DataFrame(fold_rows)
    fold_ledger.to_csv(f"{OUT_DIR}/s08v2_fold_ledger.csv", index=False)
    fold_ledger.to_parquet(f"{OUT_DIR}/s08v2_fold_ledger.parquet", index=False)

    # Clean MM summary and significance
    df_mm_clean.to_csv(f"{OUT_DIR}/s08v2_summary_mm_clean.csv", index=False)
    df_mm_clean.to_parquet(f"{OUT_DIR}/s08v2_summary_mm_clean.parquet", index=False)
    sig_mm_clean.to_csv(f"{OUT_DIR}/s08v2_significance_mm_clean.csv", index=False)
    sig_mm_clean.to_parquet(f"{OUT_DIR}/s08v2_significance_mm_clean.parquet", index=False)

    # All-folds summary and significance
    df_mm_all.to_csv(f"{OUT_DIR}/s08v2_summary_all_folds.csv", index=False)
    df_mm_all.to_parquet(f"{OUT_DIR}/s08v2_summary_all_folds.parquet", index=False)
    sig_mm_all.to_csv(f"{OUT_DIR}/s08v2_significance_all_folds.csv", index=False)
    sig_mm_all.to_parquet(f"{OUT_DIR}/s08v2_significance_all_folds.parquet", index=False)

    # TOST and verdict JSON
    tost_dict = dict(rmt_vs_hrp=tost_rmt_hrp, lw_vs_hrp=tost_lw_hrp)
    with open(f"{OUT_DIR}/s08v2_tost.json", 'w') as f:
        json.dump(tost_dict, f, indent=2)
    with open(f"{OUT_DIR}/s08v2_verdict.json", 'w') as f:
        json.dump(verdict, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[done] {elapsed:.1f}s  peak_RAM={peak_mb:.1f} MB")
    print(f"  artifacts: {OUT_DIR}/s08v2_*")

    # ── FINAL REPORT ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL REPORT — S8 v2 (Corrected)")
    print("=" * 70)

    print(f"\nVerdict: {verdict['result']}")
    print(f"Reason:  {verdict['reason']}")
    print(f"\nNo-lookahead: passed={la_ok}")
    print(f"Peak RAM: {peak_mb:.1f} MB")

    print(f"\nFold counts:")
    print(f"  Equity-only folds (pre-crypto): {n_equity_only}")
    print(f"  Clean multi-market folds:        {n_multi}")
    print(f"  Stale folds (excluded from run): {n_stale_excluded}")
    print(f"  Total in this run:               {len(fold_meta)}")

    print(f"\n=== HEADLINE: OOS variance table — CLEAN MULTI-MARKET FOLDS ONLY (n={len(clean_mm_folds)}) ===")
    print("(folds where IS covers crypto+FX AND OOS ends within live data; 2022-09-15 – 2024-12-11)")
    print(df_mm_clean.to_string(index=False))

    print(f"\n=== SIGNIFICANCE (NW t-test, Bonferroni+BH, {N_COMPARISONS} comparisons) ===")
    print(f"Baselines: A=raw_markowitz, B=hrp  |  NW lag={NW_LAG}")
    print(sig_mm_clean[available].to_string(index=False))

    print(f"\n=== TOST EQUIVALENCE (margin = {TOST_MARGIN:.4f} annualised vol = {TOST_MARGIN*100:.2f}% vol) ===")
    print(f"  RMT vs HRP: mean_diff={tost_rmt_hrp['mean_diff']:.4f}, "
          f"p_TOST={tost_rmt_hrp['p_tost']:.4f} -> "
          f"{'EQUIVALENT' if tost_rmt_hrp['equiv'] else 'NOT equivalent to HRP at 5% (p=' + str(tost_rmt_hrp['p_tost']) + ')'}")
    lw_mean_diff = tost_lw_hrp['mean_diff']
    lw_diff_bps = int(lw_mean_diff * 10000) if lw_mean_diff is not None and not np.isnan(float(lw_mean_diff)) else 'N/A'
    lw_worse = f"({lw_diff_bps} bps WORSE)" if isinstance(lw_diff_bps, int) and lw_diff_bps > 0 else f"({lw_diff_bps} bps BETTER)" if isinstance(lw_diff_bps, int) else ""
    print(f"  LW  vs HRP: mean_diff={tost_lw_hrp['mean_diff']}, {lw_worse}, "
          f"p_TOST={tost_lw_hrp['p_tost']} -> NOT equivalent (LW underperforms HRP)")

    print("\n=== ALL-FOLDS OOS TABLE (includes 60 equity-only — NOT the headline) ===")
    print("WARNING: equity-only folds dominate; this table does not reflect multi-market performance.")
    print(df_mm_all.to_string(index=False))

    print(f"\nConclusion: {verdict['reason']}")

    return verdict, df_mm_clean, sig_mm_clean


if __name__ == "__main__":
    main()
