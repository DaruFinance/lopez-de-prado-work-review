import os; os.sched_setaffinity(0, range(16,32))
#!/usr/bin/env python3
"""
S05a v3 — Fair causal AR(1) baseline for horizon vol predictability.

BUG FIXED vs v2 (found by independent referee):
  v2's AR(1) baseline used:
      rv_lag = pd.Series(rv).shift(1)
  where rv is the FORWARD realized-vol series (rv[t] = sum(dr[t+1..t+H])).
  So rv_lag[t] = rv[t-1] = sum(dr[t..t+H-1]) — contains H-1 FUTURE bars
  relative to decision time t.  This inflated baseline AUC (0.70-0.82) and
  artificially widened the micro-features-vs-baseline excess gap.

FIX (one change only, v2 machinery otherwise unchanged):
  Replace the AR(1) baseline feature with a TRAILING, fully-causal
  realized-vol predictor:
      rv_trailing[t] = sum(|dr[t-H+1 .. t]|)
                     = pd.Series(dr).rolling(H).sum()   (NO shift)
  This uses only bars through t — fully causal at decision time t.

ALSO FIXED: np.bool_ is not JSON serializable crash in summary dump.

Launch:
  cd into the script directory && \\
  ulimit -v 10485760 && \\
  taskset -c 16-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -u s05a_v3.py > run_s05a_v3.log 2>&1
"""
# ── thread-pin env (belt-and-suspenders, taskset handles the OS side) ────────
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
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── library paths ─────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_1M, EQUITY_1M, FX_1M
PROJ8 = os.path.join(REPO_ROOT, "projects/08_microstructural_features/scripts")
for p in (_LIB, PROJ8, HERE):
    sys.path.insert(0, p)

import bars as BARS
import micro_features as MF

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

# ── bar-building parameters ───────────────────────────────────────────────────
TARGET_BARS = 4_000
VOL_WINDOW  = 20       # trailing bars for threshold
N_WFO_FOLDS = 5        # rolling WFO windows

# ── horizons ─────────────────────────────────────────────────────────────────
HORIZONS = [5, 10, 20]   # referees require H=5 AND H=10 AND H=20


# ═══════════════════════════════════════════════════════════════════════════════
# JSON encoder — handles np.bool_, np.integer, np.floating
# ═══════════════════════════════════════════════════════════════════════════════

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if np.isfinite(obj) else None
        return super().default(obj)


def _json_dump(obj, fh, **kw):
    json.dump(obj, fh, cls=_NumpyEncoder, **kw)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def heartbeat(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}]  {msg}   peak_ram={peak_ram_mb():.0f} MB", flush=True)


# ── dollar-bar builder (full history) ────────────────────────────────────────
def dollar_bars_full(base: pd.DataFrame, target: int = TARGET_BARS) -> pd.DataFrame:
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
# CORRECTED FORWARD-VOL LABEL  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════════════════════

def forward_rv_label(close: np.ndarray,
                     horizon: int,
                     vol_window: int = VOL_WINDOW):
    """
    CORRECTED label: rv[t] = sum(|logret| over bars t+1 .. t+H).

    Formula: pd.Series(dr).rolling(H).sum().shift(-H)
      - rolling(H).sum() at index i = sum(dr[i-H+1 .. i])  (trailing H bars)
      - shift(-H)        shifts that back H positions:
            result[t] = sum(dr[t+1 .. t+H])
      This is a genuinely forward-looking window: bar t's label is built
      entirely from future bars t+1 .. t+H. The result is NaN for t >= N-H.

    Threshold: rolling median of rv[t-vol_window .. t-1] (strictly causal).
    Label: 1 if rv[t] > threshold[t], else 0. Sentinel -1 for invalid bars.

    Returns (label, rv, threshold, dr).
      dr is also returned so the AR(1) baseline can use it for trailing RV.
    """
    lp = np.log(np.asarray(close, float))
    n  = len(lp)
    dr = np.abs(np.diff(lp, prepend=lp[0]))   # |logret[t]|, bar t

    # CORRECTED: genuine H-bar forward realized vol
    rv = pd.Series(dr).rolling(horizon).sum().shift(-horizon).to_numpy()
    # rv[t] is NaN for t >= n-horizon

    # Threshold: rolling median of rv over vol_window PAST bars
    rv_s = pd.Series(rv)
    threshold = (rv_s.shift(1)
                     .rolling(vol_window, min_periods=max(1, vol_window // 2))
                     .median()
                     .to_numpy())

    label = np.where(
        np.isfinite(rv) & np.isfinite(threshold),
        (rv > threshold).astype(int),
        -1
    )
    return label, rv, threshold, dr


def valid_mask(label: np.ndarray) -> np.ndarray:
    return label >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# POLLUTE-AND-VERIFY  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════════════════════

def pollute_and_verify(rng_seed: int = 42) -> dict:
    """
    Real pollute-and-verify:
      - Poison dr[t+1] for a specific bar t=200 (a single future bar).
      - Confirm that rv[t=200] (correct formula) CHANGES (uses that bar).
      - Confirm that features at bar t=200 do NOT change (features are causal).
      - Confirm that rv[t<200] does NOT change (no backward contamination).

    Returns dict with pass/fail results for each check.
    """
    rng = np.random.default_rng(rng_seed)
    close = np.cumprod(1 + rng.standard_normal(400) * 0.01) * 100.0
    H = 10
    t_poison = 200

    # baseline labels and features
    lbl_orig, rv_orig, thr_orig, dr_orig = forward_rv_label(close, horizon=H)

    def _make_bars_df(c, rng_state):
        n = len(c)
        vol = rng_state.uniform(1e4, 1e6, n)
        dollar_ = c * vol
        buy_d = dollar_ * rng_state.uniform(0.3, 0.7, n)
        sell_d = dollar_ - buy_d
        ticks_ = rng_state.integers(50, 500, n).astype(float)
        return pd.DataFrame({
            "open":         c,
            "high":         c * 1.001,
            "low":          c * 0.999,
            "close":        c,
            "volume":       vol,
            "dollar":       dollar_,
            "ticks":        ticks_,
            "buy_dollar":   buy_d,
            "sell_dollar":  sell_d,
        })

    rng2 = np.random.default_rng(rng_seed + 1)
    bars_df = _make_bars_df(close, rng2)
    feats_orig = MF.build_features(bars_df, "crypto")
    feat_at_t_orig = feats_orig.iloc[t_poison].to_numpy(float)

    # Poison: set close[t_poison+1] to a wildly different value
    close_poisoned = close.copy()
    close_poisoned[t_poison + 1] = close[t_poison] * 10.0

    lbl_pois, rv_pois, thr_pois, dr_pois = forward_rv_label(close_poisoned, horizon=H)

    rng3 = np.random.default_rng(rng_seed + 1)
    bars_df_pois2 = _make_bars_df(close_poisoned, rng3)
    feats_pois = MF.build_features(bars_df_pois2, "crypto")
    feat_at_t_pois = feats_pois.iloc[t_poison].to_numpy(float)

    rv_at_t_orig = rv_orig[t_poison]
    rv_at_t_pois = rv_pois[t_poison]
    check1_rv_changed = not np.isclose(rv_at_t_orig, rv_at_t_pois, rtol=1e-6)

    safe_past = t_poison + 1 - H - 1
    if safe_past > 0:
        max_delta_past = float(np.nanmax(np.abs(rv_orig[:safe_past] - rv_pois[:safe_past])))
    else:
        max_delta_past = 0.0
    check2_past_clean = (max_delta_past < 1e-10)

    feat_delta = np.nanmax(np.abs(feat_at_t_orig - feat_at_t_pois))
    check3_feats_causal = (feat_delta < 1e-6)

    result = {
        "rv_at_200_orig":     float(rv_at_t_orig),
        "rv_at_200_poisoned": float(rv_at_t_pois),
        "check1_rv_uses_future":    bool(check1_rv_changed),
        "check2_past_rv_clean":     bool(check2_past_clean),
        "check3_features_causal":   bool(check3_feats_causal),
        "max_delta_past_rv":        float(max_delta_past),
        "max_delta_features_at_t":  float(feat_delta),
        "PASS": bool(check1_rv_changed and check2_past_clean and check3_feats_causal),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FAIR CAUSAL AR(1) BASELINE  — v3 fix
# ═══════════════════════════════════════════════════════════════════════════════

def ar1_vol_auc_causal(dr: np.ndarray, label: np.ndarray,
                       horizon: int, wfo_folds: int = N_WFO_FOLDS) -> float:
    """
    FAIR causal AR(1) baseline.

    Feature at decision time t:
        rv_trailing[t] = sum(|dr[t-H+1 .. t]|)
                       = pd.Series(dr).rolling(H).sum()   (NO shift)

    This uses ONLY bars through t — fully causal.

    v2 BUG: used pd.Series(rv_forward).shift(1) where rv_forward[t] =
    sum(dr[t+1..t+H]).  So rv_forward[t-1] = sum(dr[t..t+H-1]), which
    contains H-1 future bars relative to t.  That inflated baseline AUC.

    Uses same rolling WFO structure as micro-feature model.
    Returns median OOS AUC across folds (non-overlapping test stride=H).
    """
    # FIXED: trailing (causal) RV — uses bars [t-H+1 .. t], no future info
    rv_trailing = pd.Series(dr).rolling(horizon).sum().to_numpy()   # NO shift

    mask = valid_mask(label)
    idx  = np.where(mask)[0]
    if len(idx) < 200:
        return np.nan

    rv_lag_v = rv_trailing[idx]
    label_v  = label[idx]

    fold_aucs = []
    n = len(idx)
    fold_size = n // (wfo_folds + 1)
    if fold_size < 50:
        return np.nan

    for fold in range(wfo_folds):
        train_end  = (fold + 1) * fold_size
        test_start = train_end + horizon   # purge gap
        test_end   = train_end + 2 * fold_size
        if test_end > n:
            test_end = n
        if test_end - test_start < 50:
            continue

        test_idx_full = np.arange(test_start, test_end)
        test_idx = test_idx_full[::horizon]   # stride = H -> non-overlapping labels

        x_tr = rv_lag_v[:train_end]
        y_tr = label_v[:train_end]
        x_te = rv_lag_v[test_idx]
        y_te = label_v[test_idx]

        if len(np.unique(y_te)) < 2 or len(y_te) < 10:
            continue

        x_tr = np.where(np.isnan(x_tr), np.nanmedian(x_tr), x_tr)
        med_tr = np.nanmedian(x_tr) if np.isfinite(np.nanmedian(x_tr)) else 0.0
        x_te = np.where(np.isnan(x_te), med_tr, x_te)

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0, max_iter=200)
            clf.fit(x_tr.reshape(-1, 1), y_tr)
            prob = clf.predict_proba(x_te.reshape(-1, 1))[:, 1]
            fold_aucs.append(float(roc_auc_score(y_te, prob)))
        except Exception:
            continue

    return float(np.median(fold_aucs)) if fold_aucs else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING WALK-FORWARD AUC  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════════════════════

def wfo_auc(X: np.ndarray, y_raw: np.ndarray,
            horizon: int,
            wfo_folds: int = N_WFO_FOLDS) -> tuple:
    """
    Rolling WFO with non-overlapping test points (stride = H).

    Returns (median_auc, auc_band_low, auc_band_high, effective_n_test, fold_aucs).
    """
    mask = valid_mask(y_raw)
    idx  = np.where(mask)[0]
    if len(idx) < 200:
        return np.nan, np.nan, np.nan, 0, []

    X_v = X[idx]
    y_v = y_raw[idx]
    n   = len(idx)

    fold_size = n // (wfo_folds + 1)
    if fold_size < 50:
        return np.nan, np.nan, np.nan, 0, []

    fold_aucs = []
    total_test_pts = 0

    for fold in range(wfo_folds):
        train_end  = (fold + 1) * fold_size
        test_start = train_end + horizon
        test_end   = train_end + 2 * fold_size
        if test_end > n:
            test_end = n
        if test_end - test_start < 50:
            continue

        test_idx_full = np.arange(test_start, test_end)
        test_idx = test_idx_full[::horizon]

        X_tr = X_v[:train_end].copy()
        y_tr = y_v[:train_end]
        X_te = X_v[test_idx].copy()
        y_te = y_v[test_idx]

        if len(np.unique(y_te)) < 2 or len(y_te) < 10:
            continue

        col_med = np.nanmedian(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            X_tr[:, j] = np.where(np.isnan(X_tr[:, j]), col_med[j], X_tr[:, j])
            X_te[:, j] = np.where(np.isnan(X_te[:, j]), col_med[j], X_te[:, j])

        try:
            clf = LogisticRegressionCV(Cs=5, cv=3, penalty="l1", solver="saga",
                                       max_iter=300, random_state=42)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            fold_aucs.append(float(roc_auc_score(y_te, prob)))
            total_test_pts += len(y_te)
        except Exception:
            continue

    if not fold_aucs:
        return np.nan, np.nan, np.nan, 0, []

    med_auc  = float(np.median(fold_aucs))
    low_auc  = float(np.percentile(fold_aucs, 25))
    high_auc = float(np.percentile(fold_aucs, 75))
    eff_n    = total_test_pts

    return med_auc, low_auc, high_auc, eff_n, fold_aucs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN S5a v3
# ═══════════════════════════════════════════════════════════════════════════════

def run_s5a_v3():
    heartbeat("S5a v3 start — fair causal AR(1) baseline")

    # ── POLLUTE-AND-VERIFY first ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("POLLUTE-AND-VERIFY (unchanged from v2)")
    print("="*70)
    pv = pollute_and_verify()
    print(f"  rv[t=200] original:     {pv['rv_at_200_orig']:.6f}")
    print(f"  rv[t=200] after poison: {pv['rv_at_200_poisoned']:.6f}")
    print(f"  CHECK 1  rv uses future bar:        {'PASS' if pv['check1_rv_uses_future']  else 'FAIL'}")
    print(f"  CHECK 2  past rv uncontaminated:    {'PASS' if pv['check2_past_rv_clean']   else 'FAIL'}")
    print(f"  CHECK 3  features causal at t=200:  {'PASS' if pv['check3_features_causal'] else 'FAIL'}")
    print(f"    max|delta| past rv:    {pv['max_delta_past_rv']:.2e}")
    print(f"    max|delta| features:   {pv['max_delta_features_at_t']:.2e}")
    print(f"  => POLLUTE-AND-VERIFY: {'PASS' if pv['PASS'] else 'FAIL'}")
    pv_pass = pv["PASS"]

    # ── Per-instrument processing ─────────────────────────────────────────────
    rows = []

    def process(sym, market, loader):
        try:
            bars, feats = loader()
            close = bars["close"].to_numpy(float)
            lp = np.log(close)

            feat_cols = [c for c in feats.columns if feats[c].notna().sum() > 200]
            X_raw = feats[feat_cols].to_numpy(float)

            row = {"symbol": sym, "market": market, "n_bars": len(close)}

            for H in HORIZONS:
                lbl, rv, thr, dr = forward_rv_label(close, horizon=H)

                # Micro-feature WFO AUC (non-overlapping test stride=H)
                med_auc, lo, hi, eff_n, fold_aucs = wfo_auc(X_raw, lbl, H)

                # FAIR causal AR(1) baseline (v3 fix)
                ar1_auc = ar1_vol_auc_causal(dr, lbl, H)

                # Excess AUC: micro features vs fair causal AR(1) baseline
                excess = (med_auc - ar1_auc) if (np.isfinite(med_auc) and np.isfinite(ar1_auc)) else np.nan

                row[f"auc_micro_h{H}"]   = med_auc
                row[f"auc_lo_h{H}"]      = lo
                row[f"auc_hi_h{H}"]      = hi
                row[f"eff_n_h{H}"]        = eff_n
                row[f"auc_ar1_h{H}"]      = ar1_auc
                row[f"auc_excess_h{H}"]   = excess
                row[f"n_folds_h{H}"]      = len(fold_aucs)

            rows.append(row)
            del bars, feats, X_raw, lbl, rv, thr, dr
            gc.collect()

            line = (f"  {sym:20s}  "
                    + "  ".join(
                        f"H{H}: micro={row.get(f'auc_micro_h{H}', np.nan):.3f} "
                        f"fair_ar1={row.get(f'auc_ar1_h{H}', np.nan):.3f} "
                        f"exc={row.get(f'auc_excess_h{H}', np.nan):+.3f}"
                        for H in HORIZONS))
            heartbeat(line)

        except Exception as e:
            import traceback
            heartbeat(f"  {sym} ERROR: {e}\n{traceback.format_exc()}")

    for pair in CRYPTO_PAIRS:
        process(pair, "crypto", lambda p=pair: load_crypto(p))
    for tick in EQUITY_TICKS:
        process(tick, "equity", lambda t=tick: load_equity(t))
    for pair in FX_PAIRS:
        process(pair, "fx",     lambda p=pair: load_fx(p))

    df = pd.DataFrame(rows)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIGN TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SIGN TESTS: excess AUC > 0 (micro beats FAIR causal AR(1) baseline)")
    print("Per-market + cross-market. Bonferroni/BH over H in {5,10,20}.")
    print("="*70)

    markets = ["crypto", "equity", "fx", "all"]
    sign_rows = []

    for H in HORIZONS:
        col = f"auc_excess_h{H}"
        for mkt in markets:
            if mkt == "all":
                sub = df.dropna(subset=[col])
            else:
                sub = df[df["market"] == mkt].dropna(subset=[col])

            arr = sub[col].to_numpy()
            n = len(arr)
            if n < 3:
                sign_rows.append({
                    "H": H, "market": mkt, "n": n,
                    "n_pos": int((arr > 0).sum()) if n > 0 else 0,
                    "p_raw": np.nan, "median_excess": np.nan,
                })
                continue

            n_pos = int((arr > 0).sum())
            p_raw = binomtest(n_pos, n, p=0.5, alternative="greater").pvalue

            if mkt == "crypto":
                n_eff = max(int(np.sqrt(n)), 1)
                if n_pos >= n_eff:
                    p_eff = binomtest(min(n_pos, n_eff), n_eff, p=0.5, alternative="greater").pvalue
                else:
                    p_eff = 1.0
            else:
                p_eff = p_raw
                n_eff = n

            sign_rows.append({
                "H": H, "market": mkt, "n": n, "n_eff": n_eff,
                "n_pos": n_pos, "p_raw": float(p_raw), "p_eff": float(p_eff),
                "median_excess": float(np.median(arr)),
            })

    sign_df = pd.DataFrame(sign_rows)

    all_rows = sign_df[sign_df["market"] == "all"].dropna(subset=["p_eff"])
    if len(all_rows) > 0:
        _, p_bh, _, _ = multipletests(all_rows["p_eff"].to_numpy(), method="fdr_bh")
        sign_df.loc[all_rows.index, "p_bh_h"] = p_bh
        p_bonf = all_rows["p_eff"].to_numpy() * len(all_rows)
        sign_df.loc[all_rows.index, "p_bonf_h"] = np.minimum(p_bonf, 1.0)

    print(sign_df.to_string(index=False))

    # ═══════════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    verdict_by_H = {}
    for H in HORIZONS:
        all_row = sign_df[(sign_df["market"] == "all") & (sign_df["H"] == H)]
        if all_row.empty or all_row.get("p_bh_h", pd.Series([np.nan])).isna().all():
            verdict_by_H[H] = "INSUFFICIENT_DATA"
            continue
        p_bh_val  = all_row["p_bh_h"].iloc[0] if "p_bh_h" in all_row.columns else np.nan
        med_exc   = all_row["median_excess"].iloc[0]
        if np.isnan(p_bh_val):
            verdict_by_H[H] = "INSUFFICIENT_DATA"
        elif p_bh_val < 0.05 and med_exc > 0:
            verdict_by_H[H] = "PASS"
        elif p_bh_val > 0.20:
            verdict_by_H[H] = "NULL"
        else:
            verdict_by_H[H] = "MARGINAL"

    for H, v in verdict_by_H.items():
        row_all = sign_df[(sign_df["market"] == "all") & (sign_df["H"] == H)]
        if not row_all.empty:
            p_bh_val = row_all["p_bh_h"].iloc[0] if "p_bh_h" in row_all.columns else np.nan
            med_exc  = row_all["median_excess"].iloc[0]
            p_bh_str = f"{p_bh_val:.4f}" if np.isfinite(p_bh_val) else "n/a"
            print(f"  H={H:2d}: verdict={v:15s}  median_excess_AUC={med_exc:+.4f}  p_BH={p_bh_str}")
        else:
            print(f"  H={H:2d}: verdict={v}")

    print()
    print("Cross-instrument median AUC by horizon (FAIR comparison):")
    for H in HORIZONS:
        col_micro = f"auc_micro_h{H}"
        col_ar1   = f"auc_ar1_h{H}"
        med_micro = df[col_micro].median() if col_micro in df.columns else np.nan
        med_ar1   = df[col_ar1].median()   if col_ar1   in df.columns else np.nan
        exc = (med_micro - med_ar1) if (np.isfinite(med_micro) and np.isfinite(med_ar1)) else np.nan
        exc_str = f"{exc:+.4f}" if np.isfinite(exc) else "n/a"
        print(f"  H={H:2d}: micro={med_micro:.4f}  fair_AR1={med_ar1:.4f}  excess={exc_str}")

    # HONEST VERDICT
    print()
    print("HONEST VERDICT (v3 — corrected baseline):")
    negatives = sum(1 for v in verdict_by_H.values() if v in ("NULL", "MARGINAL"))
    positives = sum(1 for v in verdict_by_H.values() if v == "PASS")
    if negatives == len(HORIZONS):
        print("  NEGATIVE HOLDS: micro features do NOT beat the fair causal AR(1) "
              "baseline at any horizon (BH-corrected). The earlier over-stated "
              "excess (v2) was entirely due to the future-peeking baseline.")
    elif positives > 0 and negatives > 0:
        print("  MIXED: micro features beat the fair baseline at some horizons "
              "but not all. See per-horizon verdicts above.")
    elif positives == len(HORIZONS):
        print("  POSITIVE: micro features beat the fair causal AR(1) baseline "
              "at ALL horizons (BH-corrected). Surprising given the corrected baseline.")
    else:
        print("  INSUFFICIENT DATA to draw a conclusion.")

    # ── Save artifacts ────────────────────────────────────────────────────────
    df.to_parquet(f"{OUTDIR}/s5a_v3_horizon_auc.parquet", index=False)
    sign_df.to_parquet(f"{OUTDIR}/s5a_v3_sign_tests.parquet", index=False)

    with open(f"{OUTDIR}/s5a_v3_pollute_verify.json", "w") as fh:
        _json_dump(pv, fh, indent=2)

    summary = {
        "version": "v3",
        "baseline": "fair_causal_trailing_rv",
        "baseline_formula": "pd.Series(dr).rolling(H).sum()  # NO shift — fully causal",
        "v2_bug": ("v2 used pd.Series(rv_forward).shift(1) which = sum(dr[t..t+H-1]) "
                   "— H-1 future bars at decision time t"),
        "verdict_by_horizon": {str(k): v for k, v in verdict_by_H.items()},
        "pollute_and_verify": bool(pv["PASS"]),
        "peak_ram_mb": round(float(peak_ram_mb()), 0),
        "artifacts": {
            "main":           f"{OUTDIR}/s5a_v3_horizon_auc.parquet",
            "sign_tests":     f"{OUTDIR}/s5a_v3_sign_tests.parquet",
            "pollute_verify": f"{OUTDIR}/s5a_v3_pollute_verify.json",
            "summary":        f"{OUTDIR}/s5a_v3_summary.json",
        }
    }
    with open(f"{OUTDIR}/s5a_v3_summary.json", "w") as fh:
        _json_dump(summary, fh, indent=2)

    heartbeat(f"S5a v3 done.  peak_ram={peak_ram_mb():.0f} MB")
    return df, sign_df, verdict_by_H, pv_pass


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    print("="*70)
    print("S05a v3 — Fair Causal AR(1) Baseline (Referee Fix)")
    print(f"Instruments: {len(CRYPTO_PAIRS)} crypto + "
          f"{len(EQUITY_TICKS)} ETFs + {len(FX_PAIRS)} FX = "
          f"{len(CRYPTO_PAIRS) + len(EQUITY_TICKS) + len(FX_PAIRS)} total")
    print(f"Horizons:    {HORIZONS}  (BH-corrected across all H)")
    print(f"WFO folds:   {N_WFO_FOLDS}  (non-overlapping test stride = H)")
    print()
    print("v2 baseline BUG: pd.Series(rv_forward).shift(1)")
    print("  rv_forward[t-1] = sum(dr[t..t+H-1])  <- H-1 FUTURE bars!")
    print("v3 baseline FIX: pd.Series(dr).rolling(H).sum()  [NO shift]")
    print("  rv_trailing[t]  = sum(dr[t-H+1..t])  <- fully causal")
    print("="*70)

    df5a, sign_df, verdict_by_H, pv_pass = run_s5a_v3()

    elapsed = time.time() - t0
    peak    = peak_ram_mb()

    print("\n" + "="*70)
    print("FINAL RESULTS")
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)
    print(df5a.to_string(index=False))
    print()
    print("SIGN TEST SUMMARY:")
    print(sign_df.to_string(index=False))
    print()
    print(f"Pollute-and-verify: {'PASS' if pv_pass else 'FAIL'}")
    print(f"Elapsed: {elapsed/60:.1f} min    Peak RAM: {peak:.0f} MB")
    print("="*70)
