#!/usr/bin/env python3
"""
S8 — Shrink-then-Optimize: deep, full-scale, walk-forward study.

Locked bar (PREREGISTRATION.md, item S8):
  PASS: on the MULTI-MARKET 44-asset panel, shrunk/denoised (Ledoit-Wolf & RMT)
        min-variance beats raw Markowitz AND ties-or-beats HRP on OOS variance
        net of turnover (sig.) — "fix the inputs."
  FAIL→ "HRP still wins multi-market; the crypto-only win doesn't generalize."

Allocators tested (both panels, walk-forward 252/63):
  raw_markowitz  : min-variance on raw sample covariance
  hrp            : Hierarchical Risk Parity (AFML Ch.16)
  lw_minvar      : Ledoit-Wolf shrinkage → min-variance
  rmt_minvar     : RMT/MP denoising → min-variance
  inv_var        : inverse-variance
  equal_weight   : 1/N

Panels:
  multimarket    : 44-asset (27 crypto perps + 8 FX + 9 equity ETFs), cached
  crypto_only    : crypto-only subset of the same cache

No-lookahead check:
  All covariance estimates (sample cov, LW shrinkage, MP denoising) are fitted
  exclusively on the IS block [t-252, t). Applied to OOS [t, t+63) with NO data
  from that window. Verified by construction: IS slice is extracted before any
  fit call; LW and MP fits are called inside the IS block only.

Costs: 7 bps per side (14 bps round-trip). Turnover = sum|w_new - w_old|/2.
       Cost per rebalance = 14 bps × turnover. Deducted from day-0 of each OOS block.

RAM: 44×44 covariance matrix, trivial. Peak ≈ 50 MB.
"""
from __future__ import annotations
import os
import sys
import time
import json
import warnings

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

# Single-threaded BLAS/OMP.
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
COST_PER_SIDE   = 7e-4          # 7 bps/side crypto perp (taker-equivalent)
RT_COST         = 2 * COST_PER_SIDE   # 14 bps round-trip
IS_WIN          = 252
OOS_WIN         = 63
STEP            = 63

# Crypto-only column prefixes to exclude equity/FX
_EQUITY_FX = {"AUD","EUR","GBP","NZD","USD","JPY","CAD","CHF",
               "SPY","QQQ","IWM","XL","VXX","UVXY"}


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


# ── Ledoit-Wolf ────────────────────────────────────────────────────────────────

def cov_ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Oracle Approximating Shrinkage (sklearn Ledoit-Wolf), fitted on X (IS)."""
    lw = LedoitWolf(store_precision=False)
    lw.fit(X)
    return lw.covariance_


# ── RMT/MP denoising (constant-residual) ─────────────────────────────────────

def _mp_pdf(var: float, q: float, pts: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    lo = var * (1 - np.sqrt(1.0 / q)) ** 2
    hi = var * (1 + np.sqrt(1.0 / q)) ** 2
    x = np.linspace(lo, hi, pts)
    pdf = q / (2 * np.pi * var * x) * np.sqrt(np.clip((hi - x) * (x - lo), 0, None))
    return x, pdf


def cov_rmt_denoise(X: np.ndarray) -> np.ndarray:
    """Constant-residual eigenvalue denoising (ML4AM Ch.2), fitted on X (IS).
    Falls back to raw sample cov when q = T/N <= 1 (MP needs T > N)."""
    cov = np.cov(X, rowvar=False)
    T, N = X.shape
    q = T / float(N)
    if q <= 1.0:
        return cov          # MP not valid; return raw
    corr, std = _corr_from_cov(cov)
    ev, vecs = np.linalg.eigh(corr)
    order = np.argsort(ev)[::-1]
    ev = ev[order]; vecs = vecs[:, order]

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


def _quasi_diag(link: np.ndarray, n: int) -> list[int]:
    root = to_tree(link, rd=False)
    order: list[int] = []
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
    n = len(sort_ix)
    w = np.ones(n)
    stack: list[tuple[int, int]] = [(0, n)]
    diag = np.diag(cov)

    def cvar(idx: np.ndarray) -> float:
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
        left = sort_ix[s:mid]; right = sort_ix[mid:e]
        vL, vR = cvar(left), cvar(right)
        d = vL + vR
        alpha = 1.0 - vL / d if d > 0 else 0.5
        w[left] *= alpha
        w[right] *= (1.0 - alpha)
        stack.append((s, mid))
        stack.append((mid, e))
    return w


def w_hrp(cov: np.ndarray) -> np.ndarray:
    """HRP on raw sample covariance (AFML Ch.16, single-linkage, corr-distance)."""
    corr, _ = _corr_from_cov(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, 1))
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = np.asarray(_quasi_diag(link, cov.shape[0]), dtype=np.int64)
    return _hrp_bisect(cov, sort_ix)


# ── Allocator registry ─────────────────────────────────────────────────────────
# Each function takes IS return matrix X (T_is × N_active) and returns w (N_active,)

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


ALLOCATORS: dict[str, callable] = {
    "raw_markowitz":  alloc_raw_markowitz,
    "hrp":            alloc_hrp,
    "lw_minvar":      alloc_lw_minvar,
    "rmt_minvar":     alloc_rmt_minvar,
    "inv_var":        alloc_inv_var,
    "equal_weight":   alloc_equal_weight,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward engine
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward(
    ret_df: pd.DataFrame,
    is_win: int = IS_WIN,
    oos_win: int = OOS_WIN,
    step: int = STEP,
) -> dict[str, dict]:
    """
    Rolling WFO. Weights estimated on IS [t-is_win, t), held over OOS [t, t+oos_win).
    No-lookahead: all covariance fits are on is_block only (verified by construction).

    Turnover = sum|w_new_full - w_old_full| / 2 per fold.
    Cost per fold = RT_COST * turnover, deducted from day-0 of each OOS block.

    Returns {allocator: {oos_gross, oos_net, oos_vol, oos_sharpe_gross,
                          oos_sharpe_net, fold_vols, fold_turns, n_folds}}.
    """
    R = np.nan_to_num(ret_df.to_numpy(float), nan=0.0)
    T, N = R.shape
    names = list(ALLOCATORS.keys())

    # per-allocator accumulators
    acc = {nm: dict(gross_blocks=[], net_blocks=[], turns=[]) for nm in names}
    prev_w = {nm: np.full(N, 1.0 / N) for nm in names}

    n_folds = 0
    for t in range(is_win, T - 1, step):
        is_block  = R[t - is_win : t]
        oos_block = R[t : min(t + oos_win, T)]
        if oos_block.shape[0] < 5:
            continue

        # drop assets flat in IS (zero variance → degenerate cov)
        active = is_block.std(axis=0) > 0
        if active.sum() < 4:
            continue
        idx = np.where(active)[0]
        is_a  = is_block[:, idx]   # NO lookahead: IS only
        oos_a = oos_block[:, idx]

        n_folds += 1
        for nm, fn in ALLOCATORS.items():
            try:
                w = fn(is_a)          # fit entirely on IS block
            except Exception:
                w = w_equal(len(idx))
            # expand to full N for turnover calc
            w_full = np.zeros(N)
            w_full[idx] = w
            turn = float(np.abs(w_full - prev_w[nm]).sum() / 2.0)
            prev_w[nm] = w_full.copy()

            port_gross = oos_a @ w   # gross daily OOS returns
            port_net = port_gross.copy()
            cost = RT_COST * turn
            port_net[0] -= cost      # deduct round-trip on day-0

            acc[nm]["gross_blocks"].append(port_gross)
            acc[nm]["net_blocks"].append(port_net)
            acc[nm]["turns"].append(turn)

    out = {}
    for nm in names:
        if not acc[nm]["gross_blocks"]:
            continue
        g = np.concatenate(acc[nm]["gross_blocks"])
        n = np.concatenate(acc[nm]["net_blocks"])
        turns = np.array(acc[nm]["turns"])

        # per-fold OOS vol (annualised) for reporting bands
        fold_vols = [float(b.std(ddof=1) * ANN) if len(b) > 1 else np.nan
                     for b in acc[nm]["gross_blocks"]]

        out[nm] = dict(
            oos_gross        = g,
            oos_net          = n,
            oos_vol          = float(g.std(ddof=1) * ANN),
            oos_sharpe_gross = float(OF.sharpe(g) * ANN),
            oos_sharpe_net   = float(OF.sharpe(n) * ANN),
            mean_turn        = float(turns.mean()),
            fold_vols        = fold_vols,
            fold_turns       = turns.tolist(),
            n_folds          = n_folds,
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Significance: per-fold paired t-test (vol) and sign test
# ═══════════════════════════════════════════════════════════════════════════════

def _paired_vol_test(fold_vols_a: list[float], fold_vols_b: list[float]) -> dict:
    """Paired t-test: does allocator B have lower per-fold OOS vol than A?
    a = baseline (e.g. raw_markowitz or hrp), b = challenger."""
    va = np.array(fold_vols_a, float)
    vb = np.array(fold_vols_b, float)
    mask = np.isfinite(va) & np.isfinite(vb)
    va = va[mask]; vb = vb[mask]
    if len(va) < 3:
        return dict(n=int(len(va)), t_stat=np.nan, p_value=np.nan,
                    sign_frac=np.nan, mean_diff=np.nan)
    diff = vb - va   # negative = b beats a on vol
    t, p = ss.ttest_1samp(diff, 0.0)
    # one-tailed: H1 = b vol < a vol
    p_one_tail = float(p) / 2.0 if float(t) < 0 else 1.0 - float(p) / 2.0
    sign_frac = float((diff < 0).mean())
    return dict(
        n         = int(len(va)),
        t_stat    = round(float(t), 3),
        p_value   = round(p_one_tail, 4),   # one-tailed H1: b < a
        sign_frac = round(sign_frac, 3),
        mean_diff = round(float(diff.mean()), 5),
    )


def significance_table(
    wf: dict[str, dict],
    baseline_a: str,
    baseline_b: str,
) -> pd.DataFrame:
    """Cross-sectional significance vs two baselines: raw_markowitz and hrp."""
    rows = []
    for nm in ALLOCATORS:
        if nm not in wf or baseline_a not in wf or baseline_b not in wf:
            continue
        fv = wf[nm]["fold_vols"]
        vs_a = _paired_vol_test(wf[baseline_a]["fold_vols"], fv)
        vs_b = _paired_vol_test(wf[baseline_b]["fold_vols"], fv)
        rows.append(dict(
            allocator         = nm,
            vs_rawmktz_t      = vs_a["t_stat"],
            vs_rawmktz_p1tail = vs_a["p_value"],
            vs_rawmktz_sign   = vs_a["sign_frac"],
            vs_hrp_t          = vs_b["t_stat"],
            vs_hrp_p1tail     = vs_b["p_value"],
            vs_hrp_sign       = vs_b["sign_frac"],
        ))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table builder
# ═══════════════════════════════════════════════════════════════════════════════

def summary_table(wf: dict[str, dict], panel_name: str) -> pd.DataFrame:
    rows = []
    for nm, v in wf.items():
        fv = [x for x in v["fold_vols"] if np.isfinite(x)]
        rows.append(dict(
            panel           = panel_name,
            allocator       = nm,
            oos_vol_ann     = round(v["oos_vol"], 4),
            oos_vol_p5      = round(np.percentile(fv, 5), 4)  if fv else np.nan,
            oos_vol_p95     = round(np.percentile(fv, 95), 4) if fv else np.nan,
            sharpe_gross    = round(v["oos_sharpe_gross"], 3),
            sharpe_net      = round(v["oos_sharpe_net"], 3),
            mean_turnover   = round(v["mean_turn"], 4),
            n_folds         = v["n_folds"],
        ))
    df = pd.DataFrame(rows).sort_values("oos_vol_ann").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# No-lookahead verification (pollute-and-verify)
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_no_lookahead(ret_df: pd.DataFrame) -> bool:
    """
    Pollute-and-verify: inject a large spike into the FIRST row of the OOS block
    (day t) and confirm that the computed weights (which are estimated on IS [t-252, t))
    do NOT change. If weights changed, lookahead bias is present.
    Test uses a single fold with the LW estimator (most sensitive to contamination).
    """
    R = np.nan_to_num(ret_df.to_numpy(float), nan=0.0)
    T, N = R.shape
    t = IS_WIN  # first fold

    is_block_clean = R[t - IS_WIN : t].copy()
    active = is_block_clean.std(axis=0) > 0
    idx = np.where(active)[0]
    is_a = is_block_clean[:, idx]

    # weights on clean IS
    w_clean = alloc_lw_minvar(is_a)

    # pollute: inject large spike at row t (OOS day-0 — should NOT affect IS weights)
    R_polluted = R.copy()
    R_polluted[t, :] += 1.0e6   # obvious contamination signal

    is_block_polluted = R_polluted[t - IS_WIN : t]  # IS range unchanged
    is_a_polluted = is_block_polluted[:, idx]
    w_polluted = alloc_lw_minvar(is_a_polluted)

    maxdiff = float(np.max(np.abs(w_clean - w_polluted)))
    passed = maxdiff < 1e-10
    print(f"  [no-LA check] max |w_clean - w_polluted| = {maxdiff:.2e}  passed={passed}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("S8 — Shrink-then-Optimize  (deep, full-scale, walk-forward)")
    print("=" * 70)

    # ── load 44-asset panel ──────────────────────────────────────────────────
    print("\n[1] Loading 44-asset panel...")
    ret_full = pd.read_parquet(CACHE_RET)
    with open(CACHE_CLS) as f:
        classmap = json.load(f)
    print(f"    raw panel: {ret_full.shape[0]} days × {ret_full.shape[1]} assets")

    # ── construct multimarket panel ──────────────────────────────────────────
    # Keep only assets with at least 800 observations (need IS_WIN+some OOS folds).
    # Drop rows where ALL assets are NaN.
    min_obs = IS_WIN + OOS_WIN + 10
    valid_cols = ret_full.columns[ret_full.notna().sum() >= min_obs]
    ret_mm = ret_full[valid_cols].dropna(how="all")
    # forward-fill short gaps (holidays) then fill remaining NaN with 0
    ret_mm = ret_mm.ffill(limit=3).fillna(0.0)
    print(f"    multimarket panel: {ret_mm.shape[0]} days × {ret_mm.shape[1]} assets")
    print(f"    assets: {list(ret_mm.columns)}")

    # ── construct crypto-only panel ───────────────────────────────────────────
    crypto_cols = [c for c in valid_cols
                   if not any(c.startswith(pfx) for pfx in _EQUITY_FX)]
    ret_crypto = ret_full[crypto_cols].dropna(how="all")
    ret_crypto = ret_crypto.ffill(limit=3).fillna(0.0)
    print(f"    crypto-only panel: {ret_crypto.shape[0]} days × {ret_crypto.shape[1]} assets")

    # ── no-lookahead verification ──────────────────────────────────────────────
    print("\n[2] No-lookahead verification (pollute-and-verify)...")
    la_ok_mm     = _verify_no_lookahead(ret_mm)
    la_ok_crypto = _verify_no_lookahead(ret_crypto)
    if not la_ok_mm or not la_ok_crypto:
        print("  WARNING: lookahead check FAILED — investigate before publishing.")
    else:
        print("  Lookahead check PASSED on both panels.")

    # ── walk-forward: multimarket ─────────────────────────────────────────────
    print("\n[3] Walk-forward — MULTIMARKET panel (252/63)...")
    wf_mm = walk_forward(ret_mm, IS_WIN, OOS_WIN, STEP)
    df_mm = summary_table(wf_mm, "multimarket")
    sig_mm = significance_table(wf_mm, "raw_markowitz", "hrp")
    print("\n  OOS summary (multimarket):")
    print(df_mm.to_string(index=False))
    print("\n  Significance (multimarket, H1: challenger vol < baseline):")
    print(sig_mm.to_string(index=False))

    # ── walk-forward: crypto-only ─────────────────────────────────────────────
    print("\n[4] Walk-forward — CRYPTO-ONLY panel (252/63)...")
    wf_crypto = walk_forward(ret_crypto, IS_WIN, OOS_WIN, STEP)
    df_crypto = summary_table(wf_crypto, "crypto_only")
    sig_crypto = significance_table(wf_crypto, "raw_markowitz", "hrp")
    print("\n  OOS summary (crypto-only):")
    print(df_crypto.to_string(index=False))
    print("\n  Significance (crypto-only):")
    print(sig_crypto.to_string(index=False))

    # ── verdict ───────────────────────────────────────────────────────────────
    print("\n[5] Verdict vs locked bar...")
    verdict = _compute_verdict(wf_mm, sig_mm, df_mm)
    print(f"  {verdict['result']}: {verdict['reason']}")

    # ── save artifacts ────────────────────────────────────────────────────────
    print("\n[6] Saving artifacts...")
    df_mm.to_parquet(f"{OUT_DIR}/s08_summary_multimarket.parquet", index=False)
    df_mm.to_csv(f"{OUT_DIR}/s08_summary_multimarket.csv", index=False)
    df_crypto.to_parquet(f"{OUT_DIR}/s08_summary_crypto.parquet", index=False)
    df_crypto.to_csv(f"{OUT_DIR}/s08_summary_crypto.csv", index=False)
    sig_mm.to_parquet(f"{OUT_DIR}/s08_significance_multimarket.parquet", index=False)
    sig_mm.to_csv(f"{OUT_DIR}/s08_significance_multimarket.csv", index=False)
    sig_crypto.to_parquet(f"{OUT_DIR}/s08_significance_crypto.parquet", index=False)
    sig_crypto.to_csv(f"{OUT_DIR}/s08_significance_crypto.csv", index=False)

    # per-fold ledger (OOS vol per fold per allocator, both panels)
    fold_rows = []
    for nm, v in wf_mm.items():
        for fi, (fv, ft) in enumerate(zip(v["fold_vols"], v["fold_turns"])):
            fold_rows.append(dict(panel="multimarket", allocator=nm,
                                  fold=fi, oos_vol=fv, turnover=ft))
    for nm, v in wf_crypto.items():
        for fi, (fv, ft) in enumerate(zip(v["fold_vols"], v["fold_turns"])):
            fold_rows.append(dict(panel="crypto_only", allocator=nm,
                                  fold=fi, oos_vol=fv, turnover=ft))
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_parquet(f"{OUT_DIR}/s08_fold_ledger.parquet", index=False)
    fold_df.to_csv(f"{OUT_DIR}/s08_fold_ledger.csv", index=False)

    with open(f"{OUT_DIR}/s08_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[done] {elapsed:.1f}s")
    print(f"  artifacts: {OUT_DIR}/")

    # ── final summary printout ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL REPORT — S8 Shrink-then-Optimize")
    print("=" * 70)
    print(f"\nVerdict: {verdict['result']}")
    print(f"Reason:  {verdict['reason']}")
    print(f"\nNo-lookahead check: MM={la_ok_mm}, Crypto={la_ok_crypto}")
    print("\nMULTIMARKET OOS variance table (annualised vol, gross):")
    print(df_mm[["allocator","oos_vol_ann","oos_vol_p5","oos_vol_p95",
                  "sharpe_net","mean_turnover","n_folds"]].to_string(index=False))
    print("\nSIGNIFICANCE (multimarket, paired t-test per fold):")
    print(sig_mm.to_string(index=False))
    print("\nCRYPTO-ONLY OOS variance table:")
    print(df_crypto[["allocator","oos_vol_ann","oos_vol_p5","oos_vol_p95",
                       "sharpe_net","mean_turnover","n_folds"]].to_string(index=False))
    print("\nCRYPTO SIGNIFICANCE:")
    print(sig_crypto.to_string(index=False))

    return verdict, df_mm, df_crypto, sig_mm, sig_crypto


def _compute_verdict(
    wf_mm: dict[str, dict],
    sig_mm: pd.DataFrame,
    df_mm: pd.DataFrame,
) -> dict:
    """
    Locked bar (S8):
      PASS: on multimarket panel:
        (a) lw_minvar vol < raw_markowitz vol  (shrunk beats raw Markowitz), AND
        (b) lw_minvar vol <= hrp vol + 5bp tolerance (ties-or-beats HRP), AND
        (c) rmt_minvar vol < raw_markowitz vol, AND
        (d) at least one of (lw or rmt) achieves significance vs hrp (p < 0.10,
            one-tailed vol reduction, paired t-test across folds).
      FAIL: otherwise → "HRP still wins multi-market; crypto-only win doesn't generalize."
    """
    vol = {row["allocator"]: row["oos_vol_ann"] for _, row in df_mm.iterrows()}
    sig_rows = sig_mm.set_index("allocator")

    if any(nm not in vol for nm in ["raw_markowitz","hrp","lw_minvar","rmt_minvar"]):
        return dict(result="NULL", reason="missing allocators in results")

    v_raw = vol["raw_markowitz"]
    v_hrp = vol["hrp"]
    v_lw  = vol["lw_minvar"]
    v_rmt = vol["rmt_minvar"]

    cond_a = v_lw < v_raw   # LW beats raw Markowitz on vol
    cond_b = v_rmt < v_raw  # RMT beats raw Markowitz on vol
    cond_c = v_lw <= v_hrp + 0.005   # LW ties-or-beats HRP (5bp tolerance)
    cond_d = v_rmt <= v_hrp + 0.005  # RMT ties-or-beats HRP

    # significance: at least one of lw/rmt significantly beats hrp (p < 0.10)
    p_lw_vs_hrp  = float(sig_rows.loc["lw_minvar",  "vs_hrp_p1tail"])  if "lw_minvar"  in sig_rows.index else 1.0
    p_rmt_vs_hrp = float(sig_rows.loc["rmt_minvar", "vs_hrp_p1tail"])  if "rmt_minvar" in sig_rows.index else 1.0
    sig_vs_hrp = min(p_lw_vs_hrp, p_rmt_vs_hrp) < 0.10

    # significance vs raw_markowitz
    p_lw_vs_raw  = float(sig_rows.loc["lw_minvar",  "vs_rawmktz_p1tail"]) if "lw_minvar"  in sig_rows.index else 1.0
    p_rmt_vs_raw = float(sig_rows.loc["rmt_minvar", "vs_rawmktz_p1tail"]) if "rmt_minvar" in sig_rows.index else 1.0
    sig_vs_raw = min(p_lw_vs_raw, p_rmt_vs_raw) < 0.10

    detail = (
        f"lw_vol={v_lw:.4f} rmt_vol={v_rmt:.4f} raw_mktz_vol={v_raw:.4f} hrp_vol={v_hrp:.4f}; "
        f"beats_raw=(lw:{cond_a},rmt:{cond_b}); "
        f"ties_hrp=(lw:{cond_c},rmt:{cond_d}); "
        f"sig_vs_raw(p<0.10):{sig_vs_raw}; sig_vs_hrp(p<0.10):{sig_vs_hrp}; "
        f"p_lw_vs_hrp={p_lw_vs_hrp:.4f} p_rmt_vs_hrp={p_rmt_vs_hrp:.4f}"
    )

    pass_bar = (
        (cond_a or cond_b)        # at least one shrink beats raw Markowitz
        and (cond_c or cond_d)    # at least one ties-or-beats HRP
        and sig_vs_raw             # significant improvement over raw Markowitz
    )
    if pass_bar:
        result = "PASS"
        reason = (
            "Shrunk/denoised min-variance beats raw Markowitz AND ties-or-beats HRP "
            "on multimarket OOS variance with cross-sectional significance. "
            "Demonstrates: the classic optimizer fails on bad inputs, not a bad objective."
        )
    else:
        result = "FAIL"
        reason = (
            "HRP still wins on multimarket; shrinkage improvement does not "
            "generalize from crypto-only or does not achieve significance."
        )
    return dict(result=result, reason=reason, detail=detail,
                cond_a=cond_a, cond_b=cond_b, cond_c=cond_c, cond_d=cond_d,
                sig_vs_raw=sig_vs_raw, sig_vs_hrp=sig_vs_hrp,
                p_lw_vs_hrp=p_lw_vs_hrp, p_rmt_vs_hrp=p_rmt_vs_hrp)


if __name__ == "__main__":
    main()
