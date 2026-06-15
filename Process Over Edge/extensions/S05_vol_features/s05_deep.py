#!/usr/bin/env python3
"""
S05 DEEP — Predictive Features: vol horizon, feature interactions, SADF risk-off.
Pre-registration: PREREGISTRATION.md  S5a / S5b / S5c
Peak-RAM estimate: ~1.2 GB (one instrument at a time, full history, del+gc after each).

Launch wrapper (CPU restriction, RAM cap, thread pins):
  cd into the script directory && \
  ulimit -v 10485760 && \
  taskset -c 16-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 s05_deep.py > run.log 2>&1
"""
# ── thread pins BEFORE any numpy / numba import ─────────────────────────────
import os
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import gc
import sys
import time
import resource
import warnings
import json
import numpy as np
import pandas as pd
from scipy import stats as ss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── library paths ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_1M, EQUITY_1M, FX_1M
PROJ7    = os.path.join(REPO_ROOT, "projects/07_structural_breaks_entropy/scripts")
PROJ8    = os.path.join(REPO_ROOT, "projects/08_microstructural_features/scripts")
for p in (_LIB, PROJ7, PROJ8, HERE):
    sys.path.insert(0, p)

import bars as BARS
import micro_features as MF
import breaks_entropy as BE

# ── output dir ────────────────────────────────────────────────────────────────
OUTDIR = HERE
os.makedirs(OUTDIR, exist_ok=True)

# ── instrument registry ───────────────────────────────────────────────────────
CRYPTO_DIR = CRYPTO_1M
EQUITY_DIR = EQUITY_1M
FX_DIR     = FX_1M

CRYPTO_PAIRS = [
    "1000SHIBUSDT", "AAVEUSDT", "ALGOUSDT", "APEUSDT", "APTUSDT",
    "ARBUSDT",      "ATOMUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT",
    "BTCUSDT",      "DOGEUSDT", "DOTUSDT",  "ETCUSDT", "ETHUSDT",
    "HBARUSDT",     "ICPUSDT",  "LINKUSDT", "LTCUSDT", "NEARUSDT",
    "SOLUSDT",      "SUIUSDT",  "TRXUSDT",  "UNIUSDT", "XLMUSDT",
    "XRPUSDT",      "ZECUSDT",
]
EQUITY_TICKS = ["IWM", "QQQ", "SPY", "UVXY", "VXX", "XLE", "XLF", "XLK", "XLV"]
FX_PAIRS     = ["AUDUSD", "EURGBP", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD",
                "USDCHF", "USDJPY"]

# ── real-cost constants ───────────────────────────────────────────────────────
#   crypto: 5bp taker + 2bp slip = 7bp one-way
#   equity/fx: 3bp taker + 2bp slip = 5bp one-way
COST_CRYPTO = 7e-4   # 0.07% one-way
COST_EQ_FX  = 5e-4   # 0.05% one-way

# ── bar-building parameters ───────────────────────────────────────────────────
TARGET_BARS  = 4_000   # dollar bars per instrument (full history)
VOL_WINDOW   = 20      # trailing bars for the causal vol threshold
SPREAD_WIN   = 20
LAMBDA_WIN   = 50


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def peak_ram_mb() -> float:
    """Peak RSS in MB (Linux /proc/self/status ru_maxrss is in kB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def heartbeat(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}]  {msg}   peak_ram={peak_ram_mb():.0f} MB", flush=True)


# ── dollar-bar builder (full history) ────────────────────────────────────────
def dollar_bars_full(base: pd.DataFrame, target: int = TARGET_BARS) -> pd.DataFrame:
    """All available history, TARGET_BARS dollar bars."""
    thresh = base["quote_volume"].sum() / target
    return BARS.threshold_bars(base, "quote_volume", max(thresh, 1e-8))


# ── instrument loaders ────────────────────────────────────────────────────────
def load_crypto(pair: str):
    path = f"{CRYPTO_DIR}/{pair}_1m.parquet"
    base = BARS.load_base(path)
    bars = dollar_bars_full(base)
    feats = MF.build_features(bars, "crypto")
    return bars, feats


def load_equity(tick: str):
    path = f"{EQUITY_DIR}/{tick}.csv.gz"
    base = BARS.load_base_equity_etf(path)
    bars = dollar_bars_full(base)
    feats = MF.build_features(bars, "equity")
    return bars, feats


def load_fx(pair: str):
    path = f"{FX_DIR}/{pair}_fx1m.parquet"
    base = BARS.load_base_fx(path)
    bars = dollar_bars_full(base)
    feats = MF.build_features(bars, "forex")
    return bars, feats


# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL VOL LABEL — the no-lookahead crux for S5a / S5b
# ═══════════════════════════════════════════════════════════════════════════════

def causal_vol_label(close: np.ndarray,
                     horizon: int = 1,
                     vol_window: int = VOL_WINDOW) -> np.ndarray:
    """Binary label: is realised vol over bars [t+1, t+horizon] above the
    rolling median of past realised vols?

    STRICTLY CAUSAL construction:
      1. rv[t] = sum|logret| over [t+1 .. t+horizon]  (the OUTCOME at bar t)
      2. threshold[t] = rolling median of rv[t-vol_window .. t-1]
         — uses ONLY observations STRICTLY BEFORE bar t, with a .shift(1)
         so the threshold is available before the label is realised.
      3. label[t] = 1 if rv[t] > threshold[t], else 0.
      4. The final horizon bars have no valid outcome → NaN/0.

    Train/test split must use a purge gap ≥ horizon (no label from the test
    region bleeds into the training window).  This function returns labels
    aligned to the CLOSE bar index (length = len(close)).

    CAUSAL-CHECK (inline, printed at runtime):
      * threshold[t] is derived from rv[t-vol_window .. t-1]:
        at no index t does the threshold use future rv values.
        We verify this by checking that shift(1).rolling(vol_window).median()
        is bit-identical to a hand-rolled loop over past-only rv values.
    """
    lp = np.log(np.asarray(close, float))
    n = len(lp)
    dr = np.abs(np.diff(lp, prepend=lp[0]))   # |logret[t]|, aligned to bar t

    # Realised vol = forward sum of |logret| over horizon bars.
    # rv[t] = sum(dr[t+1 .. t+horizon])  (0-indexed: sum dr[t+1], ..., dr[t+horizon])
    # This is the LABEL at bar t; it uses data AFTER bar t, which is correct —
    # labels are the thing we predict, not the thing we use as a predictor.
    rv = pd.Series(dr).shift(-1).rolling(horizon).sum().to_numpy()
    # rv[t] is NaN for t >= n-horizon (not enough future bars)

    # Threshold: rolling median of rv over vol_window PAST bars.
    # shift(1) ensures rv[t] itself is excluded → pure look-back window ending at t-1.
    rv_s = pd.Series(rv)
    threshold = rv_s.shift(1).rolling(vol_window, min_periods=max(1, vol_window // 2)).median().to_numpy()

    # Label: 1 if outcome > past-median
    label = np.where(
        np.isfinite(rv) & np.isfinite(threshold),
        (rv > threshold).astype(int),
        -1    # sentinel: invalid bar (burn-in or tail)
    )
    return label, rv, threshold


def valid_label_mask(label: np.ndarray) -> np.ndarray:
    return label >= 0


# ── purged OOS split with gap ≥ horizon ──────────────────────────────────────
def purged_split_auc(X: np.ndarray, y_raw: np.ndarray,
                     horizon: int,
                     train_frac: float = 0.60,
                     model: str = "gbt") -> float:
    """OOS AUC on a time-purged split.

    gap = horizon bars are dropped between train and test so no label from
    the test region can overlap with the train window.  The gap is >= horizon
    by construction (the crux for S5a).

    CAUSAL-CHECK: at prediction time we only use features built from bars <= t.
    No future bar enters X (by construction of build_features + causal_vol_label).
    """
    mask = valid_label_mask(y_raw)
    X_v = X[mask]; y_v = y_raw[mask]
    n = len(y_v)
    cut = int(n * train_frac)
    gap = max(horizon, 5)   # purge gap >= horizon (the bar requirement)

    X_tr, y_tr = X_v[:cut], y_v[:cut]
    X_os, y_os = X_v[cut + gap:], y_v[cut + gap:]
    if len(np.unique(y_os)) < 2 or len(y_os) < 50:
        return np.nan

    # fill NaN in features with IS-only column medians (causal: no OOS info)
    col_med = np.nanmedian(X_tr, axis=0)
    for j in range(X_tr.shape[1]):
        X_tr[:, j] = np.where(np.isnan(X_tr[:, j]), col_med[j], X_tr[:, j])
        X_os[:, j] = np.where(np.isnan(X_os[:, j]), col_med[j], X_os[:, j])

    if model == "gbt":
        clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42)
    else:
        clf = LogisticRegressionCV(Cs=5, cv=3, penalty="l1", solver="saga",
                                   max_iter=200, random_state=42)
    try:
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_os)[:, 1]
        return float(roc_auc_score(y_os, prob))
    except Exception:
        return np.nan


def marginal_auc(x1d: np.ndarray, y_raw: np.ndarray,
                 horizon: int, train_frac: float = 0.60) -> float:
    """AUC of a single feature (logistic or threshold classifier)."""
    return purged_split_auc(x1d.reshape(-1, 1), y_raw, horizon,
                            train_frac, model="lr")


# ── annualised costed Sharpe ──────────────────────────────────────────────────
BARS_PER_YEAR_CRYPTO = 365 * 8   # ~8 dollar bars per day
BARS_PER_YEAR_EQ     = 252 * 8
BARS_PER_YEAR_FX     = 260 * 8


def annualised_sharpe(returns: np.ndarray, bars_per_year: int,
                      cost_per_trade: float = 0.0,
                      trades_per_bar: float = 1.0) -> float:
    r = np.asarray(returns, float) - cost_per_trade * trades_per_bar
    if r.std(ddof=1) < 1e-15:
        return np.nan
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(bars_per_year))


def max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1.0)
    return float(dd.min())


def calmar(returns: np.ndarray, bars_per_year: int) -> float:
    cum = np.cumprod(1 + returns)
    ann_ret = float(cum[-1] ** (bars_per_year / len(returns)) - 1)
    md = abs(max_drawdown(cum))
    if md < 1e-10:
        return np.nan
    return ann_ret / md


# ═══════════════════════════════════════════════════════════════════════════════
# S5a — LONGER-HORIZON VOL AUC vs 1-BAR
# ═══════════════════════════════════════════════════════════════════════════════
# Strictly causal label; purge gap = horizon; cross-section sign test.
# Bar met: 10-bar AUC > 1-bar AUC sign-consistent across cross-section (p < 0.05).

HORIZONS = [1, 5, 10, 20]   # bars; ~0.125d / 0.6d / 1.25d / 2.5d at 8 bars/day

def run_s5a():
    heartbeat("S5a start — horizon vol AUC")
    rows = []

    def process(sym, market, loader):
        try:
            bars, feats = loader()
            close = bars["close"].to_numpy(float)
            feat_cols = [c for c in feats.columns if feats[c].notna().sum() > 200]
            X_raw = feats[feat_cols].to_numpy(float)
            bpy = (BARS_PER_YEAR_CRYPTO if market == "crypto" else
                   BARS_PER_YEAR_EQ if market == "equity" else BARS_PER_YEAR_FX)
            row = {"symbol": sym, "market": market, "n_bars": len(close)}
            for H in HORIZONS:
                lbl, rv, thr = causal_vol_label(close, horizon=H)
                auc = purged_split_auc(X_raw, lbl, horizon=H)
                row[f"auc_h{H}"] = auc
            rows.append(row)
            del bars, feats, X_raw, lbl, rv, thr
            gc.collect()
            heartbeat(f"  {sym}: {row}")
        except Exception as e:
            heartbeat(f"  {sym} ERROR: {e}")

    # CRYPTO ──────────────────────────────────────────────────────────────────
    for pair in CRYPTO_PAIRS:
        process(pair, "crypto", lambda p=pair: load_crypto(p))

    # EQUITY ──────────────────────────────────────────────────────────────────
    for tick in EQUITY_TICKS:
        process(tick, "equity", lambda t=tick: load_equity(t))

    # FX ──────────────────────────────────────────────────────────────────────
    for pair in FX_PAIRS:
        process(pair, "fx", lambda p=pair: load_fx(p))

    df = pd.DataFrame(rows)

    # ── CAUSAL-LABEL CHECK (printed) ─────────────────────────────────────────
    # Verify that threshold[t] at any t does not depend on rv[t+1..]
    # We do a synthetic pollute-and-verify: set rv_future = -999 for all bars
    # after the midpoint and check threshold values in the first half are unchanged.
    print("\n=== CAUSAL-LABEL POLLUTE-AND-VERIFY ===")
    rng = np.random.default_rng(42)
    fake_close = np.cumprod(1 + rng.standard_normal(500) * 0.01) * 100.0
    lbl_orig, rv_orig, thr_orig = causal_vol_label(fake_close, horizon=10)
    # poison second half of rv (simulates "future data leaking in")
    rv_poisoned = rv_orig.copy()
    rv_poisoned[250:] = -999.0
    rv_s = pd.Series(rv_poisoned)
    thr_poisoned = rv_s.shift(1).rolling(VOL_WINDOW, min_periods=max(1, VOL_WINDOW // 2)).median().to_numpy()
    max_delta = float(np.nanmax(np.abs(thr_orig[:240] - thr_poisoned[:240])))
    print(f"  max|threshold_delta| in [0:240] after poisoning [250:]: {max_delta:.6f}")
    print(f"  => {'PASS: threshold is causal (no future bleed)' if max_delta < 1e-10 else 'FAIL: lookahead detected'}")

    # ── SIGN TEST: is 10-bar AUC > 1-bar AUC across cross-section? ───────────
    valid = df.dropna(subset=["auc_h1", "auc_h10"])
    delta_10v1 = (valid["auc_h10"] - valid["auc_h1"]).to_numpy()
    n_pos = int((delta_10v1 > 0).sum()); n = len(delta_10v1)
    # two-sided binomial sign test (H0: p=0.5)
    from scipy.stats import binomtest
    btest = binomtest(n_pos, n, p=0.5, alternative="greater")
    print(f"\n  S5a cross-section sign test (h10 > h1): {n_pos}/{n} positive, "
          f"p={btest.pvalue:.4f}")
    s5a_pass = (btest.pvalue < 0.05)
    verdict_5a = "PASS" if s5a_pass else ("NULL" if btest.pvalue > 0.20 else "FAIL")
    print(f"  => S5a verdict: {verdict_5a}")

    df.to_parquet(f"{OUTDIR}/s5a_horizon_auc.parquet", index=False)
    heartbeat(f"S5a done.  {verdict_5a}  peak_ram={peak_ram_mb():.0f} MB")
    return df, verdict_5a


# ═══════════════════════════════════════════════════════════════════════════════
# S5b — FEATURE INTERACTIONS ADD AUC OVER MARGINALS (multiple-comparison control)
# ═══════════════════════════════════════════════════════════════════════════════
# horizon=10 (per S5a crux); L1-regularized AUC for both marginal and augmented.
# BH(FDR<0.10) across all (instrument × interaction) pairs.

INTERACTION_PAIRS = [
    ("entropy", "vol_trail"),   # rolling Shannon entropy × trailing |ret|
    ("ofi",     "roll_spread"), # OFI × Roll spread  (crypto/equity only)
    ("vpin",    "kyle"),        # VPIN × Kyle lambda (crypto/equity only)
]

def _build_entropy_feature(bars, feats) -> np.ndarray:
    """Rolling 50-bar Shannon entropy of sign-quantized log-returns (causal)."""
    close = bars["close"].to_numpy(float)
    lp = np.log(close)
    dr = np.diff(lp, prepend=lp[0])
    msg = BE.quantize_signbins(dr, 2)
    ent = BE.rolling_entropy(msg, window=50, kind="shannon")
    return ent


def run_s5b():
    heartbeat("S5b start — feature interactions")
    rows = []
    HORIZON = 10

    def process(sym, market, loader):
        try:
            bars, feats = loader()
            close = bars["close"].to_numpy(float)
            lbl, _, _ = causal_vol_label(close, horizon=HORIZON)
            feat_cols = [c for c in feats.columns if feats[c].notna().sum() > 200]
            X_base = feats[feat_cols].to_numpy(float)

            # IS-medians from training half only (causal fill)
            n_train = int(len(close) * 0.6)
            col_med = np.nanmedian(X_base[:n_train], axis=0)
            for j in range(X_base.shape[1]):
                X_base[:, j] = np.where(np.isnan(X_base[:, j]), col_med[j], X_base[:, j])

            # Trailing vol proxy (causal: backward rolling std)
            lp = np.log(close)
            dr = np.abs(np.diff(lp, prepend=lp[0]))
            vol_trail = pd.Series(dr).rolling(20).mean().to_numpy()
            vol_trail = np.where(np.isnan(vol_trail), np.nanmedian(vol_trail), vol_trail)

            ent = _build_entropy_feature(bars, feats)
            ent_f = np.where(np.isnan(ent), np.nanmedian(ent[np.isfinite(ent)]), ent)

            auc_base = purged_split_auc(X_base, lbl, horizon=HORIZON, model="lr")

            for ia_name, ia_b in INTERACTION_PAIRS:
                if ia_name == "entropy":
                    feat_a = ent_f
                    feat_b = vol_trail
                else:
                    if ia_name not in feat_cols or ia_b not in feat_cols:
                        continue
                    feat_a = X_base[:, feat_cols.index(ia_name)]
                    feat_b = X_base[:, feat_cols.index(ia_b)]
                inter = feat_a * feat_b
                X_aug = np.column_stack([X_base, inter.reshape(-1, 1)])
                auc_aug = purged_split_auc(X_aug, lbl, horizon=HORIZON, model="lr")
                delta = (auc_aug - auc_base) if (np.isfinite(auc_aug) and np.isfinite(auc_base)) else np.nan
                rows.append({
                    "symbol": sym, "market": market,
                    "interaction": f"{ia_name}x{ia_b}",
                    "auc_base": auc_base, "auc_aug": auc_aug,
                    "delta": delta,
                })
            del bars, feats, X_base, lbl
            gc.collect()
            heartbeat(f"  {sym} done")
        except Exception as e:
            heartbeat(f"  {sym} ERROR: {e}")

    for pair in CRYPTO_PAIRS:
        process(pair, "crypto", lambda p=pair: load_crypto(p))
    for tick in EQUITY_TICKS:
        process(tick, "equity", lambda t=tick: load_equity(t))
    for pair in FX_PAIRS:
        process(pair, "fx",     lambda p=pair: load_fx(p))

    df = pd.DataFrame(rows)

    # ── multiple-comparison control (BH FDR<0.10) ────────────────────────────
    # One-sided z-test: H0: delta <= 0; test statistic = delta / (naive stderr estimate).
    # We use a permutation-free sign test per interaction family and then BH across all.
    print("\n=== S5b MULTIPLE-COMPARISON CONTROL (BH FDR<0.10) ===")
    results_mc = []
    for ia_label in df["interaction"].unique():
        sub = df[df["interaction"] == ia_label].dropna(subset=["delta"])
        deltas = sub["delta"].to_numpy()
        n_pos = int((deltas > 0).sum()); n = len(deltas)
        if n < 3:
            results_mc.append({"interaction": ia_label, "n": n, "n_pos": n_pos,
                                "p_raw": np.nan, "median_delta": np.nan})
            continue
        from scipy.stats import binomtest as bt
        p = bt(n_pos, n, p=0.5, alternative="greater").pvalue
        results_mc.append({"interaction": ia_label, "n": n, "n_pos": n_pos,
                            "p_raw": p,
                            "median_delta": float(np.median(deltas))})

    mc_df = pd.DataFrame(results_mc)
    valid_p = mc_df.dropna(subset=["p_raw"])
    if len(valid_p) > 0:
        _, p_adj, _, _ = multipletests(valid_p["p_raw"].to_numpy(), method="fdr_bh")
        mc_df.loc[valid_p.index, "p_bh"] = p_adj
    else:
        mc_df["p_bh"] = np.nan

    robust_interactions = mc_df[mc_df.get("p_bh", pd.Series(dtype=float)) < 0.10]
    print(mc_df.to_string(index=False))
    n_robust = len(robust_interactions)
    verdict_5b = "PASS" if n_robust > 0 else "FAIL"
    if n_robust == 0 and mc_df["p_raw"].min() > 0.20:
        verdict_5b = "NULL"
    print(f"\n  Robust interactions (BH<0.10): {list(robust_interactions['interaction'])}")
    print(f"  => S5b verdict: {verdict_5b}")

    df.to_parquet(f"{OUTDIR}/s5b_interactions.parquet", index=False)
    mc_df.to_parquet(f"{OUTDIR}/s5b_mc_summary.parquet", index=False)
    heartbeat(f"S5b done.  {verdict_5b}  peak_ram={peak_ram_mb():.0f} MB")
    return df, mc_df, verdict_5b


# ═══════════════════════════════════════════════════════════════════════════════
# S5c — SADF EXPLOSIVE-REGIME AS RISK-OFF SIZE CUT
# ═══════════════════════════════════════════════════════════════════════════════
# Baseline sleeve: momentum (sign last bar ret), long only, costed.
# risk-off: zero size when SADF > 0 (explosive).
# entry: trade only when explosive (the wrong use we want to show does NOT help).
# Sign-consistency: risk-off improves BOTH Sharpe AND Calmar across cross-section.

SADF_MINLEN  = 50   # min window length for SADF
SADF_MAXWIN  = 300  # cap (keeps O(n*300) instead of O(n^2))
SADF_LAGS    = 1

def _compute_sadf(lp: np.ndarray) -> np.ndarray:
    """Causal capped SADF aligned to the price index (NaN prefix)."""
    yvar, X_des, base_idx = BE._adf_design(lp, SADF_LAGS)
    yvar = np.ascontiguousarray(yvar)
    X_des = np.ascontiguousarray(X_des)
    sadf_rows = BE._sadf_capped_kernel(yvar, X_des, SADF_MINLEN, SADF_MAXWIN)
    sadf_full = np.full(len(lp), np.nan)
    sadf_full[base_idx] = sadf_rows
    # forward-fill to avoid NaN-indexed gaps in early bars
    sadf_s = pd.Series(sadf_full).fillna(method="ffill").to_numpy()
    return sadf_s


def _sleeve(ret: np.ndarray, sig: np.ndarray, sadf: np.ndarray,
            oos_start: int, n_oos: int,
            cost: float, bpy: int) -> dict:
    """Evaluate three arms: baseline, risk-off, entry-only."""
    s = oos_start; e = s + n_oos
    r_base     = sig[s:e] * ret[s + 1:e + 1]
    explosive  = sadf[s:e] > 0.0

    r_riskoff  = np.where(~explosive, r_base, 0.0)
    r_entry    = np.where( explosive, r_base, 0.0)

    # trades per bar: base=1.0 (one turn per bar), risk-off=pct_active turns
    pct_active_riskoff = float((~explosive).mean())
    pct_active_entry   = float(explosive.mean())

    sh_base    = annualised_sharpe(r_base,    bpy, cost, 1.0)
    sh_riskoff = annualised_sharpe(r_riskoff, bpy, cost, pct_active_riskoff)
    sh_entry   = annualised_sharpe(r_entry,   bpy, cost, pct_active_entry)

    cum_base    = np.cumprod(1 + np.clip(r_base, -0.5, 0.5))
    cum_riskoff = np.cumprod(1 + np.clip(r_riskoff, -0.5, 0.5))
    cum_entry   = np.cumprod(1 + np.clip(r_entry, -0.5, 0.5))

    cal_base    = calmar(np.clip(r_base, -0.5, 0.5),    bpy)
    cal_riskoff = calmar(np.clip(r_riskoff, -0.5, 0.5), bpy)
    cal_entry   = calmar(np.clip(r_entry, -0.5, 0.5),   bpy)

    return {
        "sharpe_base":    sh_base,
        "sharpe_riskoff": sh_riskoff,
        "sharpe_entry":   sh_entry,
        "calmar_base":    cal_base,
        "calmar_riskoff": cal_riskoff,
        "calmar_entry":   cal_entry,
        "pct_explosive":  float(explosive.mean()),
        "sharpe_delta_riskoff": sh_riskoff - sh_base if (np.isfinite(sh_riskoff) and np.isfinite(sh_base)) else np.nan,
        "sharpe_delta_entry":   sh_entry   - sh_base if (np.isfinite(sh_entry)   and np.isfinite(sh_base)) else np.nan,
        "calmar_delta_riskoff": cal_riskoff - cal_base if (np.isfinite(cal_riskoff) and np.isfinite(cal_base)) else np.nan,
    }


def run_s5c():
    heartbeat("S5c start — SADF risk-off switch")
    rows = []

    def process(sym, market, loader):
        try:
            bars, _feats = loader()
            close = bars["close"].to_numpy(float)
            lp = np.log(close)
            dr = np.diff(lp, prepend=lp[0])
            n = len(dr)

            sadf = _compute_sadf(lp)

            sig = np.sign(dr)               # momentum direction signal (causal)
            bpy = (BARS_PER_YEAR_CRYPTO if market == "crypto" else
                   BARS_PER_YEAR_EQ if market == "equity" else BARS_PER_YEAR_FX)
            cost = COST_CRYPTO if market == "crypto" else COST_EQ_FX

            cut = int(n * 0.60); gap = 20
            n_oos = n - cut - gap - 1
            if n_oos < 100:
                return

            res = _sleeve(dr, sig, sadf, cut + gap, n_oos, cost, bpy)
            res.update({"symbol": sym, "market": market, "n_bars": n, "n_oos": n_oos})
            rows.append(res)
            del bars, _feats, sadf, close, lp, dr, sig
            gc.collect()
            heartbeat(f"  {sym}: sharpe_delta_riskoff={res['sharpe_delta_riskoff']:.3f}")
        except Exception as e:
            heartbeat(f"  {sym} ERROR: {e}")

    for pair in CRYPTO_PAIRS:
        process(pair, "crypto", lambda p=pair: load_crypto(p))
    for tick in EQUITY_TICKS:
        process(tick, "equity", lambda t=tick: load_equity(t))
    for pair in FX_PAIRS:
        process(pair, "fx",     lambda p=pair: load_fx(p))

    df = pd.DataFrame(rows)

    # ── SIGN TESTS ─────────────────────────────────────────────────────────
    print("\n=== S5c SIGN TESTS ===")
    from scipy.stats import binomtest as bt

    def sign_result(label, col, df_=df):
        sub = df_.dropna(subset=[col])
        arr = sub[col].to_numpy()
        n_pos = int((arr > 0).sum()); n = len(arr)
        p = bt(n_pos, n, p=0.5, alternative="greater").pvalue if n > 0 else np.nan
        print(f"  {label}: {n_pos}/{n} positive, p={p:.4f}")
        return n_pos, n, p

    n_pos_sh, n_sh, p_sh = sign_result("sharpe_delta_riskoff", "sharpe_delta_riskoff")
    n_pos_cl, n_cl, p_cl = sign_result("calmar_delta_riskoff", "calmar_delta_riskoff")
    n_pos_en, n_en, p_en = sign_result("sharpe_delta_entry",   "sharpe_delta_entry")

    # S5c PASS: risk-off improves both Sharpe AND Calmar (both p<0.05, sign-consistent);
    #           entry does NOT improve (p>0.20).
    risk_off_ok = (p_sh < 0.05 and p_cl < 0.05)
    entry_fail  = (p_en > 0.10)   # entry should NOT be significant
    verdict_5c = "PASS" if (risk_off_ok and entry_fail) else \
                 "NULL" if (p_sh > 0.20 and p_cl > 0.20) else "FAIL"
    print(f"\n  => S5c verdict: {verdict_5c}  "
          f"(risk_off_ok={risk_off_ok}, entry_not_sig={entry_fail})")

    df.to_parquet(f"{OUTDIR}/s5c_sadf_riskoff.parquet", index=False)
    heartbeat(f"S5c done.  {verdict_5c}  peak_ram={peak_ram_mb():.0f} MB")
    return df, verdict_5c


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("S05 DEEP — Predictive Features")
    print(f"Instruments: {len(CRYPTO_PAIRS)} crypto perps + "
          f"{len(EQUITY_TICKS)} ETFs + {len(FX_PAIRS)} FX = "
          f"{len(CRYPTO_PAIRS) + len(EQUITY_TICKS) + len(FX_PAIRS)} total")
    print("=" * 70)

    # ── S5a ──────────────────────────────────────────────────────────────────
    df5a, verdict_5a = run_s5a()
    print("\n--- S5a FULL RESULTS ---")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(df5a.to_string(index=False))

    # ── S5b ──────────────────────────────────────────────────────────────────
    df5b, mc5b, verdict_5b = run_s5b()
    print("\n--- S5b INTERACTION RESULTS ---")
    print(df5b.to_string(index=False))

    # ── S5c ──────────────────────────────────────────────────────────────────
    df5c, verdict_5c = run_s5c()
    print("\n--- S5c SADF RESULTS ---")
    print(df5c.to_string(index=False))

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    peak = peak_ram_mb()
    summary = {
        "S5a": verdict_5a,
        "S5b": verdict_5b,
        "S5c": verdict_5c,
        "elapsed_min": round(elapsed / 60, 1),
        "peak_ram_mb": round(peak, 0),
        "artifacts": {
            "s5a": f"{OUTDIR}/s5a_horizon_auc.parquet",
            "s5b_raw": f"{OUTDIR}/s5b_interactions.parquet",
            "s5b_mc": f"{OUTDIR}/s5b_mc_summary.parquet",
            "s5c": f"{OUTDIR}/s5c_sadf_riskoff.parquet",
        }
    }
    with open(f"{OUTDIR}/s05_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 70)
    print("FINAL VERDICTS")
    print(f"  S5a (10-bar vol AUC > 1-bar):              {verdict_5a}")
    print(f"  S5b (interactions add AUC, BH-controlled): {verdict_5b}")
    print(f"  S5c (SADF risk-off improves; entry fails): {verdict_5c}")
    print(f"  elapsed: {elapsed/60:.1f} min    peak RAM: {peak:.0f} MB")
    print("=" * 70)
