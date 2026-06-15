import os; os.sched_setaffinity(0, range(16, 32))
"""
s04a_v2.py — S4a (CPCV-chosen meta-label cutoff vs single-split) CORRECTED.

Referees rejected the original s04_deep.py::S4a for:
  1. Structural leakage: global p_oof reused for threshold selection inside CPCV
     and single-split arms (META-lookahead). Fixed: each path fits meta-clf FRESH
     on that path's train indices; predicts on its test indices. No global p_oof.
  2. Tautological variance: CPCV evaluates 2/6 OOS per path, single-split 1/6 ->
     ~2x variance reduction is pure CLT, not information aggregation. Fixed:
     (a) apples-to-apples comparison via normalized variance (var * n_oos, removes
     the 1/n scaling), AND (b) permuted-threshold null to isolate the residual.
  3. Bias metric used IS reference (thresh=0.50 on all events). Fixed: use a
     genuinely held-out forward block (last 1/6 of data) that neither arm touches.
     Renamed to "threshold-selection sensitivity" and interpreted accordingly.
  4. Degenerate instruments (NaN OOS PF, too few events) are flagged and excluded.
  5. Effective-N of cross-section (eigenvalue participation ratio of instrument
     OOS PF series) reported; sign-test p restated against effective-N.
  6. Median strategy PF disclosed prominently (net loss context).

Architecture:
  - fit_meta_clf(X_tr, y_tr): fits BaggingClassifier on given train slice.
  - predict_proba_1(clf, X_te): predict on test slice.
  - For each path (CPCV or single-split): re-fit clf FRESH on tr_idx, predict
    p_te on te_idx. No p_oof ever crosses path boundaries.
  - Permuted-threshold null (EFFICIENT): pre-cache (p_te, pnl_te, n_oos) for
    each path once. The null only re-draws random thresholds using cached probas
    (no classifier re-fitting during the null). This is the correct null: it
    asks "is IS threshold selection better than random threshold selection?",
    holding fixed the classifier outputs.
  - Equal-OOS comparison: report var_norm = var * n_oos (invariant to OOS size).

Real costs throughout:
  Crypto: 8 bp/side (5 bp taker + 2 bp slip + 1 bp funding proxy)
  Equity/FX: 7 bp/side

Instruments: 27 crypto perps + 7 equity ETFs + 8 FX pairs = 42 total.
One instrument at a time (RAM-bounded). Prints peak RAM.

Outputs (S04_metalabel/):
  per_instrument_s4a_v2.parquet
  summary_s4a_v2.json
"""

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import sys, gc, json, time, warnings
import resource
import numpy as np
import pandas as pd
from scipy import stats as ss
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_1M, FX_1M, DATA_CACHE
sys.path.insert(0, _LIB)
sys.path.insert(0, os.path.join(REPO_ROOT, "projects/03_meta_labeling/scripts"))
sys.path.insert(0, HERE)

import bars as BARS
import overfit as OF
import tbm

OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)

CRYPTO_DIR   = CRYPTO_1M
FX_DIR       = FX_1M
EQUITY_CACHE = DATA_CACHE

# ─────────────────────────────────────────────────────────────────────────────
# Instrument registry
# ─────────────────────────────────────────────────────────────────────────────
CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT",
    "UNIUSDT", "ATOMUSDT", "ETCUSDT", "AAVEUSDT", "ALGOUSDT",
    "NEARUSDT", "XLMUSDT", "TRXUSDT", "DOGEUSDT", "ARBUSDT",
    "APTUSDT", "SUIUSDT", "ICPUSDT", "HBARUSDT", "APEUSDT",
    "1000SHIBUSDT", "ZECUSDT",
]

EQUITY_TICKERS = ["SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLV"]

FX_PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "USDCAD", "USDCHF", "EURGBP", "NZDUSD",
]

# ─────────────────────────────────────────────────────────────────────────────
# Cost parameters
# ─────────────────────────────────────────────────────────────────────────────
COST_CRYPTO = 8.0 / 1e4    # 8 bp/side
COST_EQ_FX  = 7.0 / 1e4    # 7 bp/side

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline parameters
# ─────────────────────────────────────────────────────────────────────────────
N_TARGET_BARS  = 20_000
VOL_SPAN       = 50
FAST, SLOW     = 20, 60
MAX_HOLD       = 50
N_SPLITS       = 6          # used for CPCV n_groups AND single-split arm
EMBARGO_PCT    = 0.01
LABEL_SPAN     = 3
N_ESTIMATORS   = 100

# CPCV: C(6,2) = 15 paths, each with 2/6 OOS
CPCV_N_GROUPS  = 6
CPCV_K_TEST    = 2

# Single-split arm: N_SPLITS - 1 = 5 consecutive-pair paths, each with 1/6 OOS
N_SINGLE       = N_SPLITS - 1   # 5 paths

# Threshold grid for IS selection
THRESH_GRID = np.array([0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60])

# Held-out forward block: last HOLDOUT_FRAC of events (reference for sensitivity metric)
HOLDOUT_FRAC   = 1.0 / N_SPLITS   # 1/6 ~ 16.7%

# Permutation null: number of random-threshold draws per path-set
# Efficient null: reuses cached (p_te, pnl_te) per path; no clf re-fitting.
N_PERM         = 500
RNG_SEED       = 42

T0_GLOBAL = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def peak_ram_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB


def heartbeat(msg):
    elapsed = (time.time() - T0_GLOBAL) / 60.0
    ram = peak_ram_gb()
    print(f"[{elapsed:6.1f}m | RAM {ram:.2f} GB] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bar builders  (identical to s04_deep.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_dollar_bars_crypto(pair):
    path = os.path.join(CRYPTO_DIR, f"{pair}_1m.parquet")
    base = tbm.load_base_crypto(path)
    return BARS.matched_bars(base, N_TARGET_BARS)["dollar"]


def build_dollar_bars_fx(pair):
    path = os.path.join(FX_DIR, f"{pair}_fx1m.parquet")
    base = BARS.load_base_fx(path)
    return BARS.matched_bars(base, N_TARGET_BARS)["dollar"]


def build_equity_bars(ticker):
    path = os.path.join(EQUITY_CACHE, f"daily_etf_ohlcv_{ticker}.parquet")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    out = pd.DataFrame({
        "open":   df["open"].to_numpy(np.float64),
        "high":   df["high"].to_numpy(np.float64),
        "low":    df["low"].to_numpy(np.float64),
        "close":  df["close"].to_numpy(np.float64),
        "volume": df["volume"].to_numpy(np.float64),
    }, index=df.index)
    out["dollar"]      = out["volume"] * out["close"]
    out["ticks"]       = out["volume"]
    out["buy_dollar"]  = out["dollar"] / 2.0
    out["sell_dollar"] = out["dollar"] / 2.0
    out = out[(out["close"] > 0) & (out["volume"] > 0)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Meta-clf helpers (no global OOF — everything per-path)
# ─────────────────────────────────────────────────────────────────────────────
def fit_meta_clf(X_tr, y_tr, n_estimators=N_ESTIMATORS, seed=0):
    """Fit BaggingClassifier on (X_tr, y_tr). Returns (clf, pi) or (None, 0)."""
    if len(X_tr) < 30 or int(y_tr.sum()) < 5 or int((1 - y_tr).sum()) < 5:
        return None, 0
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
        n_estimators=n_estimators, max_samples=0.8, max_features=0.8,
        bootstrap=True, n_jobs=1, random_state=seed)
    clf.fit(X_tr, y_tr)
    pi = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
    return clf, pi


def predict_proba_1(clf, pi, X_te):
    """Predict P(class=1). clf=None returns 0.5."""
    if clf is None:
        return np.full(len(X_te), 0.5)
    return clf.predict_proba(X_te)[:, pi]


def pf_from_proba(p_te, pnl_te, thresh):
    """PF of meta-filtered trades on (p_te, pnl_te) at thresh."""
    act = p_te >= thresh
    pnl_m = pnl_te[act]
    if len(pnl_m) == 0:
        return np.nan
    gp = pnl_m[pnl_m > 0].sum()
    gn = -pnl_m[pnl_m < 0].sum()
    return float(gp / gn) if gn > 0 else np.nan


def best_is_thresh_from_proba(p_val, pnl_val, thresh_grid):
    """IS threshold selection: pick thresh maximising PF on precomputed p_val."""
    best_t = 0.50
    best_pf = -np.inf
    for t in thresh_grid:
        pf = pf_from_proba(p_val, pnl_val, t)
        if np.isfinite(pf) and pf > best_pf:
            best_pf = pf
            best_t = t
    return best_t


def is_proba_for_path(X_tr, y_tr, pnl_tr, n_estimators=50, seed=0):
    """Compute IS validation probas for threshold selection on path's train set.
    Returns (p_val, pnl_val) on inner last-20% holdout of tr, or (None, None)."""
    n_tr = len(X_tr)
    cut = int(n_tr * 0.8)
    if cut < 20 or (n_tr - cut) < 5:
        return None, None
    clf_is, pi_is = fit_meta_clf(X_tr[:cut], y_tr[:cut], n_estimators=n_estimators, seed=seed)
    if clf_is is None:
        return None, None
    p_val = predict_proba_1(clf_is, pi_is, X_tr[cut:])
    return p_val, pnl_tr[cut:]


# ─────────────────────────────────────────────────────────────────────────────
# Effective-N: eigenvalue participation ratio
# ─────────────────────────────────────────────────────────────────────────────
def eigenvalue_participation_ratio(X):
    """PR = (sum lambda)^2 / sum(lambda^2) for correlation matrix of X.
    X: (n_series, n_instruments) — each row is one series over instruments."""
    X = np.asarray(X, float)
    N = X.shape[1] if X.ndim == 2 else X.shape[0]
    if N < 2:
        return float(N)
    # Correlation among instruments (columns)
    corr = np.corrcoef(X.T if X.ndim == 2 else X)
    corr = np.nan_to_num(corr, nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(corr), 0, None)
    s2 = np.sum(np.square(lam))
    if s2 < 1e-12:
        return float(N)
    return float(np.sum(lam) ** 2 / s2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-instrument S4a (leakage-free + efficient null)
# ─────────────────────────────────────────────────────────────────────────────
def run_s4a_instrument_v2(bars, cost_per_side, min_events=80, rng_seed=RNG_SEED):
    """
    Leakage-free CPCV vs single-split threshold-selection comparison.

    Held-out forward block: last HOLDOUT_FRAC of events is set aside as a
    genuine OOS reference. Neither arm evaluates on it. It is used only to
    compute the holdout_pf_default (clf trained on all working set, thresh=0.50).

    Working set (first 1 - HOLDOUT_FRAC of events) is split for both arms.

    CPCV arm (C(6,2)=15 paths, each with 2/6 of working set as OOS):
      For each path:
        1. Fit clf FRESH on X[tr_idx], y[tr_idx].           <- FIX: no global OOF
        2. Compute IS validation probas on inner 80/20 of tr (for thresh selection).
        3. Pick IS threshold from precomputed IS probas.
        4. Predict OOS probas on X[te_idx].
        5. Compute OOS PF at chosen threshold.
      Cache (p_te, pnl_te, n_oos, is_thresh) for efficient null.

    Single-split arm (5 consecutive fold pairs, each with 1/6 OOS):
      Same structure as CPCV path, using fold f as IS, fold f+1 as OOS.

    CLT-normalized variance (var * n_oos): removes the ~2x OOS-size difference.

    Permuted-threshold null (EFFICIENT — no clf re-fitting):
      Using the cached (p_te, pnl_te) per path, draw N_PERM random thresholds.
      Compute the variance of OOS PFs under random threshold selection.
      Residual = real_var_norm_red - null_var_norm_red.
      This isolates the benefit of IS threshold selection from pure sample size.

    Threshold-selection sensitivity:
      |arm_mean_pf - holdout_pf_default|. This measures how much the IS-selected
      threshold deviates from the default (not "bias" in the estimator sense).
    """
    close = bars["close"].to_numpy(np.float64)
    high  = bars["high"].to_numpy(np.float64)
    low   = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    vol = tbm.ewma_vol(close, VOL_SPAN)

    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    ev_all = tbm.crossover_events(side_full)
    warm = SLOW + 12
    ev_all = ev_all[(ev_all > warm) & (ev_all < n - 1)]
    if len(ev_all) < max(min_events, 60):
        return None

    side_all = side_full[ev_all]
    feat = tbm.build_features(bars, side_full, vol, FAST, SLOW)
    X_all = feat.iloc[ev_all].to_numpy(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Triple-barrier labels, fixed barrier config 1.5/1.0 for S4a
    tb = tbm.triple_barrier(close, high, low, openp, ev_all, side_all, vol, 1.5, 1.0, MAX_HOLD)
    pnl_all = tb["ret_gross"].to_numpy() - 2.0 * cost_per_side
    y_all   = (pnl_all > 0).astype(int)

    N_all = len(ev_all)

    # ── Held-out forward block ──
    holdout_start = int(np.floor(N_all * (1.0 - HOLDOUT_FRAC)))
    work_idx = np.arange(holdout_start)
    hold_idx = np.arange(holdout_start, N_all)

    N_work = len(work_idx)
    if N_work < max(min_events, 60) or len(hold_idx) < 10:
        return None

    y_work = y_all[work_idx]
    if y_work.sum() < 15 or (1 - y_work).sum() < 15:
        return None

    # ── CPCV arm: fit clf per path, cache (p_te, pnl_te, n_oos, is_thresh) ──
    # Also cache is_p_val for efficient null (reuse the IS probas for thresh draw)
    cpcv_paths = []   # list of dicts: {oos_pf, n_oos, is_thresh, p_te, pnl_te, is_p_val, is_pnl_val}

    for tr_rel, te_rel in OF.cpcv_splits(N_work, CPCV_N_GROUPS, CPCV_K_TEST,
                                          EMBARGO_PCT, LABEL_SPAN):
        tr_abs = work_idx[tr_rel]
        te_abs = work_idx[te_rel]
        if len(tr_abs) < 30 or len(te_abs) < 5:
            continue

        # FIX (leakage): fresh clf on tr_abs only
        clf, pi = fit_meta_clf(X_all[tr_abs], y_all[tr_abs], seed=0)
        if clf is None:
            continue
        p_te = predict_proba_1(clf, pi, X_all[te_abs])

        # IS threshold selection: inner 80/20 of tr
        p_val, pnl_val = is_proba_for_path(X_all[tr_abs], y_all[tr_abs], pnl_all[tr_abs],
                                            n_estimators=50, seed=0)
        if p_val is None:
            is_thresh = 0.50
            is_p_val = None
            is_pnl_val = None
        else:
            is_thresh = best_is_thresh_from_proba(p_val, pnl_val, THRESH_GRID)
            is_p_val = p_val
            is_pnl_val = pnl_val

        oos_pf = pf_from_proba(p_te, pnl_all[te_abs], is_thresh)
        if np.isfinite(oos_pf):
            cpcv_paths.append(dict(
                oos_pf=oos_pf,
                n_oos=len(te_abs),
                is_thresh=is_thresh,
                p_te=p_te,
                pnl_te=pnl_all[te_abs],
                is_p_val=is_p_val,
                is_pnl_val=is_pnl_val,
            ))

    if len(cpcv_paths) < 3:
        return None

    cpcv_oos_pfs   = np.array([p["oos_pf"] for p in cpcv_paths])
    cpcv_n_oos_arr = np.array([p["n_oos"]  for p in cpcv_paths], float)
    cpcv_mean_noos = float(cpcv_n_oos_arr.mean())

    # ── Single-split arm: same structure ──
    splits_work = list(OF.purged_kfold_splits(N_work, N_SPLITS, EMBARGO_PCT, LABEL_SPAN))

    single_paths = []

    for f in range(N_SPLITS - 1):
        is_rel  = splits_work[f][1]      # test fold of split f = IS tuning
        oos_rel = splits_work[f + 1][1]  # test fold of split f+1 = OOS eval
        is_abs  = work_idx[is_rel]
        oos_abs = work_idx[oos_rel]
        if len(is_abs) < 30 or len(oos_abs) < 5:
            continue

        # FIX (leakage): fresh clf on is_abs only
        clf_s, pi_s = fit_meta_clf(X_all[is_abs], y_all[is_abs], seed=0)
        if clf_s is None:
            continue
        p_te_s = predict_proba_1(clf_s, pi_s, X_all[oos_abs])

        # IS threshold selection: inner 80/20 of is_abs
        p_val_s, pnl_val_s = is_proba_for_path(X_all[is_abs], y_all[is_abs], pnl_all[is_abs],
                                                 n_estimators=50, seed=0)
        if p_val_s is None:
            is_thresh_s = 0.50
            is_p_val_s = None
            is_pnl_val_s = None
        else:
            is_thresh_s = best_is_thresh_from_proba(p_val_s, pnl_val_s, THRESH_GRID)
            is_p_val_s = p_val_s
            is_pnl_val_s = pnl_val_s

        oos_pf_s = pf_from_proba(p_te_s, pnl_all[oos_abs], is_thresh_s)
        if np.isfinite(oos_pf_s):
            single_paths.append(dict(
                oos_pf=oos_pf_s,
                n_oos=len(oos_abs),
                is_thresh=is_thresh_s,
                p_te=p_te_s,
                pnl_te=pnl_all[oos_abs],
                is_p_val=is_p_val_s,
                is_pnl_val=is_pnl_val_s,
            ))

    if len(single_paths) < 2:
        return None

    single_oos_pfs   = np.array([p["oos_pf"] for p in single_paths])
    single_n_oos_arr = np.array([p["n_oos"]  for p in single_paths], float)
    single_mean_noos = float(single_n_oos_arr.mean())

    # ── Variance metrics ──
    cpcv_var   = float(np.var(cpcv_oos_pfs))
    single_var = float(np.var(single_oos_pfs))

    # CLT-normalized: var * n_oos  (removes 1/n scaling)
    cpcv_var_norm   = cpcv_var   * cpcv_mean_noos
    single_var_norm = single_var * single_mean_noos

    raw_var_red  = single_var - cpcv_var
    norm_var_red = single_var_norm - cpcv_var_norm

    # ── Permuted-threshold null (EFFICIENT: no clf re-fitting) ──
    # For each permutation: draw a random threshold for each path from THRESH_GRID.
    # Compute PF on the cached (p_te, pnl_te). This directly asks:
    #   "How much variance reduction is achievable by purely random threshold assignment?"
    # Any excess of the real method over this null = genuine IS selection benefit.
    rng = np.random.default_rng(rng_seed)
    null_cpcv_var_norms   = []
    null_single_var_norms = []

    for _ in range(N_PERM):
        # CPCV null
        nc_pfs = []
        for path in cpcv_paths:
            rand_t = rng.choice(THRESH_GRID)
            pf_n = pf_from_proba(path["p_te"], path["pnl_te"], rand_t)
            if np.isfinite(pf_n):
                nc_pfs.append(pf_n)
        if len(nc_pfs) >= 3:
            nc_arr = np.array(nc_pfs)
            # Use same n_oos weights (paths that returned finite may differ, use mean of valid)
            null_cpcv_var_norms.append(float(np.var(nc_arr) * cpcv_mean_noos))

        # Single-split null
        ns_pfs = []
        for path in single_paths:
            rand_t = rng.choice(THRESH_GRID)
            pf_sn = pf_from_proba(path["p_te"], path["pnl_te"], rand_t)
            if np.isfinite(pf_sn):
                ns_pfs.append(pf_sn)
        if len(ns_pfs) >= 2:
            ns_arr = np.array(ns_pfs)
            null_single_var_norms.append(float(np.var(ns_arr) * single_mean_noos))

    valid_null = min(len(null_cpcv_var_norms), len(null_single_var_norms))
    if valid_null >= 20:
        null_cpcv_vn   = np.array(null_cpcv_var_norms[:valid_null])
        null_single_vn = np.array(null_single_var_norms[:valid_null])
        null_norm_var_red = float(np.mean(null_single_vn - null_cpcv_vn))
        residual_var_red  = norm_var_red - null_norm_var_red
        # Also compute median and std of null for reporting
        null_var_red_arr  = null_single_vn - null_cpcv_vn
        null_var_red_std  = float(np.std(null_var_red_arr))
    else:
        null_norm_var_red = np.nan
        residual_var_red  = np.nan
        null_var_red_std  = np.nan

    # ── Holdout block reference: threshold-selection sensitivity ──
    # Fit clf on ALL working set (not used in any arm evaluation).
    clf_hold, pi_hold = fit_meta_clf(X_all[work_idx], y_all[work_idx], seed=0)
    if clf_hold is not None and len(hold_idx) >= 5:
        p_hold = predict_proba_1(clf_hold, pi_hold, X_all[hold_idx])
        holdout_pf_default = pf_from_proba(p_hold, pnl_all[hold_idx], 0.50)
    else:
        holdout_pf_default = np.nan

    thresh_sens_cpcv   = abs(float(np.mean(cpcv_oos_pfs)) - holdout_pf_default) \
        if np.isfinite(holdout_pf_default) else np.nan
    thresh_sens_single = abs(float(np.mean(single_oos_pfs)) - holdout_pf_default) \
        if np.isfinite(holdout_pf_default) else np.nan

    # ── Net loss disclosure ──
    gp_all = pnl_all[pnl_all > 0].sum()
    gn_all = -pnl_all[pnl_all < 0].sum()
    full_pf_no_filter = float(gp_all / gn_all) if gn_all > 0 else np.nan

    return dict(
        n_events_total=N_all,
        n_work=N_work,
        n_holdout=len(hold_idx),

        cpcv_n_paths=len(cpcv_paths),
        cpcv_mean_pf=float(np.mean(cpcv_oos_pfs)),
        cpcv_median_pf=float(np.median(cpcv_oos_pfs)),
        cpcv_var=cpcv_var,
        cpcv_mean_noos=cpcv_mean_noos,
        cpcv_var_norm=cpcv_var_norm,

        single_n_paths=len(single_paths),
        single_mean_pf=float(np.mean(single_oos_pfs)),
        single_median_pf=float(np.median(single_oos_pfs)),
        single_var=single_var,
        single_mean_noos=single_mean_noos,
        single_var_norm=single_var_norm,

        raw_var_reduction=raw_var_red,
        norm_var_reduction=norm_var_red,
        null_norm_var_reduction=null_norm_var_red,
        null_var_reduction_std=null_var_red_std,
        residual_var_reduction=residual_var_red,

        holdout_pf_default=holdout_pf_default,
        thresh_sens_cpcv=thresh_sens_cpcv,
        thresh_sens_single=thresh_sens_single,

        full_pf_no_filter=full_pf_no_filter,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional statistics
# ─────────────────────────────────────────────────────────────────────────────
def sign_test_with_effn(n_pos, n_total, eff_n=None):
    """Binomial sign test. Returns raw p and effective-N-adjusted p."""
    if n_total < 4:
        return dict(n_pos=n_pos, n_total=n_total, p_raw=np.nan, p_effn=np.nan, eff_n=eff_n)
    try:
        p_raw = float(ss.binomtest(n_pos, n_total, 0.5, alternative='two-sided').pvalue)
    except AttributeError:
        p_raw = float(2 * min(ss.binom.cdf(n_pos, n_total, 0.5),
                               ss.binom.sf(n_pos - 1, n_total, 0.5)))
    p_effn = np.nan
    if eff_n is not None and np.isfinite(eff_n) and eff_n >= 2:
        eff_int = max(2, int(round(eff_n)))
        n_pos_eff = min(int(round(n_pos * eff_int / max(n_total, 1))), eff_int)
        try:
            p_effn = float(ss.binomtest(n_pos_eff, eff_int, 0.5, alternative='two-sided').pvalue)
        except AttributeError:
            p_effn = float(2 * min(ss.binom.cdf(n_pos_eff, eff_int, 0.5),
                                    ss.binom.sf(n_pos_eff - 1, eff_int, 0.5)))
    return dict(n_pos=n_pos, n_total=n_total, p_raw=p_raw, p_effn=p_effn, eff_n=eff_n)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    heartbeat("Starting S04a-v2 deep run (leakage-free + null-corrected)")

    instruments = (
        [("crypto", s) for s in CRYPTO_PAIRS] +
        [("equity", t) for t in EQUITY_TICKERS] +
        [("fx", p)     for p in FX_PAIRS]
    )
    heartbeat(f"Panel: {len(instruments)} instruments "
              f"({len(CRYPTO_PAIRS)} crypto, {len(EQUITY_TICKERS)} equity, {len(FX_PAIRS)} FX)")

    rows = []
    n_ok = 0
    n_skip = 0
    n_degen = 0

    for mkt, sym in instruments:
        heartbeat(f"Processing {mkt}:{sym} ...")
        try:
            if mkt == "crypto":
                bars = build_dollar_bars_crypto(sym)
                cost = COST_CRYPTO
            elif mkt == "equity":
                bars = build_equity_bars(sym)
                cost = COST_EQ_FX
            else:
                bars = build_dollar_bars_fx(sym)
                cost = COST_EQ_FX

            if len(bars) < 1500:
                heartbeat(f"  SKIP {sym}: too few bars ({len(bars)})")
                n_skip += 1
                del bars; gc.collect()
                continue

            min_ev = 50 if mkt == "equity" else 80
            r = run_s4a_instrument_v2(bars, cost, min_events=min_ev, rng_seed=RNG_SEED)

            if r is None:
                heartbeat(f"  SKIP {sym}: insufficient events or degenerate labels")
                n_skip += 1
                del bars; gc.collect()
                continue

            is_degen = (
                not np.isfinite(r["cpcv_mean_pf"]) or
                not np.isfinite(r["single_mean_pf"]) or
                r["cpcv_n_paths"] < 3 or
                r["single_n_paths"] < 2
            )
            r["degenerate"] = is_degen
            if is_degen:
                heartbeat(f"  DEGENERATE {sym}: excluded from stats")
                n_degen += 1

            bpy = len(bars) / max(
                (bars.index[-1] - bars.index[0]).total_seconds() / (365.25 * 86400), 1e-6)
            r.update(dict(market=mkt, symbol=sym, n_bars=len(bars), bpy=bpy))
            rows.append(r)

            heartbeat(f"  OK {sym}: cpcv_pf={r['cpcv_mean_pf']:.3f}  "
                      f"single_pf={r['single_mean_pf']:.3f}  "
                      f"raw_vred={r['raw_var_reduction']:+.4f}  "
                      f"norm_vred={r['norm_var_reduction']:+.4f}  "
                      f"resid={r['residual_var_reduction']:+.4f}")
            n_ok += 1

        except Exception as e:
            heartbeat(f"  ERROR {sym}: {e}")
            import traceback; traceback.print_exc()
            n_skip += 1

        if 'bars' in locals():
            del bars
        gc.collect()

    heartbeat(f"Loop done: {n_ok} processed, {n_skip} skipped, {n_degen} degenerate")

    # ── Save per-instrument results ──
    df = pd.DataFrame(rows)
    out_pq = os.path.join(OUT_DIR, "per_instrument_s4a_v2.parquet")
    df.to_parquet(out_pq, index=False)
    heartbeat(f"Saved {out_pq} ({len(df)} rows)")

    # ── Cross-sectional summary ──
    df_ok = df[~df["degenerate"]].copy() if "degenerate" in df.columns else df.copy()
    df_ok = df_ok.dropna(subset=["norm_var_reduction", "residual_var_reduction"])
    heartbeat(f"Clean instruments for stats: {len(df_ok)}")

    summary = dict(
        n_instruments_total=len(df),
        n_degenerate=n_degen,
        n_clean=len(df_ok),
    )

    # Disclosed: median strategy PF without any meta-filter
    med_full_pf = float(df_ok["full_pf_no_filter"].median()) if len(df_ok) > 0 else np.nan
    summary["disclosed_median_full_pf_no_filter"] = med_full_pf
    heartbeat(f"  DISCLOSED: median strategy PF (no meta-filter) = {med_full_pf:.4f}  "
              f"({'net loss' if med_full_pf < 1.0 else 'net gain'}; "
              f"this study evaluates ESTIMATOR VARIANCE of a {'loss-making' if med_full_pf < 1.0 else 'profitable'} strategy)")

    if len(df_ok) >= 4:
        # ── Effective-N ──
        # Use cpcv_mean_pf and single_mean_pf series as the cross-sectional
        # observations. PR = eigenvalue participation ratio among instruments.
        pf_mat = np.column_stack([
            df_ok["cpcv_mean_pf"].to_numpy(),
            df_ok["single_mean_pf"].to_numpy(),
        ])
        eff_n_val = eigenvalue_participation_ratio(pf_mat.T)
        summary["eff_n_cross_section"] = eff_n_val
        heartbeat(f"  Effective cross-section N (PR): {eff_n_val:.1f} of {len(df_ok)} nominal")

        # ── Raw variance reduction (tautological) ──
        raw_vals = df_ok["raw_var_reduction"]
        n_pos_raw = int((raw_vals > 0).sum())
        n_tot_raw = int(raw_vals.notna().sum())
        st_raw = sign_test_with_effn(n_pos_raw, n_tot_raw, eff_n=eff_n_val)

        # ── CLT-normalized variance reduction ──
        norm_vals = df_ok["norm_var_reduction"]
        n_pos_norm = int((norm_vals > 0).sum())
        n_tot_norm = int(norm_vals.notna().sum())
        st_norm = sign_test_with_effn(n_pos_norm, n_tot_norm, eff_n=eff_n_val)

        # ── Residual variance reduction (over permuted null) ──
        resid_vals = df_ok["residual_var_reduction"].dropna()
        n_pos_resid = int((resid_vals > 0).sum())
        n_tot_resid = int(len(resid_vals))
        st_resid = sign_test_with_effn(n_pos_resid, n_tot_resid, eff_n=eff_n_val)

        summary.update(dict(
            # OOS PF
            median_cpcv_mean_pf=float(df_ok["cpcv_mean_pf"].median()),
            median_single_mean_pf=float(df_ok["single_mean_pf"].median()),
            median_cpcv_mean_noos=float(df_ok["cpcv_mean_noos"].median()),
            median_single_mean_noos=float(df_ok["single_mean_noos"].median()),

            # Raw variance (tautological — OOS sizes differ)
            median_cpcv_var=float(df_ok["cpcv_var"].median()),
            median_single_var=float(df_ok["single_var"].median()),
            median_raw_var_red=float(raw_vals.median()),
            sign_test_raw=st_raw,
            tautology_note=(
                "Raw var reduction is tautological: CPCV OOS size ~2x single-split "
                f"(median {df_ok['cpcv_mean_noos'].median():.0f} vs "
                f"{df_ok['single_mean_noos'].median():.0f} events). "
                "CLT alone predicts lower variance for larger OOS. Use norm_var_red."
            ),

            # CLT-normalized (apples-to-apples)
            median_cpcv_var_norm=float(df_ok["cpcv_var_norm"].median()),
            median_single_var_norm=float(df_ok["single_var_norm"].median()),
            median_norm_var_red=float(norm_vals.median()),
            sign_test_norm=st_norm,

            # Residual over permuted null
            median_null_norm_var_red=float(df_ok["null_norm_var_reduction"].dropna().median()),
            median_residual_var_red=float(resid_vals.median()) if len(resid_vals) > 0 else np.nan,
            sign_test_residual=st_resid,

            # Threshold-selection sensitivity
            median_thresh_sens_cpcv=float(df_ok["thresh_sens_cpcv"].dropna().median()),
            median_thresh_sens_single=float(df_ok["thresh_sens_single"].dropna().median()),
        ))

        # ── VERDICT ──
        # Primary test: is residual_var_red > 0 significantly (after CLT correction
        # and after permuted-null correction)?
        # PASS: residual > 0 for >55% instruments AND p_effn < 0.05
        # PARTIAL: CLT-normalized reduction significant but null not netted
        # FAIL: residual not significant; the advantage is a sample-size artifact

        resid_frac = n_pos_resid / max(n_tot_resid, 1)
        p_resid    = st_resid.get("p_effn", np.nan)
        if p_resid is np.nan or not np.isfinite(p_resid):
            p_resid = st_resid.get("p_raw", np.nan)

        norm_frac = n_pos_norm / max(n_tot_norm, 1)
        p_norm    = st_norm.get("p_effn", np.nan)
        if p_norm is np.nan or not np.isfinite(p_norm):
            p_norm = st_norm.get("p_raw", np.nan)

        resid_pass = (np.isfinite(p_resid) and p_resid < 0.05 and resid_frac > 0.55)
        norm_pass  = (np.isfinite(p_norm)  and p_norm  < 0.05 and norm_frac  > 0.55)

        if resid_pass:
            verdict = (
                "PASS — CPCV threshold selection reduces estimator variance beyond the "
                "pure sample-size effect (residual significant after permuted-threshold null)."
            )
        elif norm_pass:
            verdict = (
                "FAIL→ CLT-normalized variance IS lower for CPCV (p_effn={:.3f}, "
                "{}/{} instruments), but residual over the permuted-threshold null is "
                "NOT significant (p={:.3f}, {}/{} instruments). "
                "The CPCV variance reduction is largely a sample-size artifact; "
                "the residual IS-selection benefit is small or insignificant."
            ).format(p_norm, n_pos_norm, n_tot_norm, p_resid, n_pos_resid, n_tot_resid)
        else:
            verdict = (
                "FAIL→ No significant variance reduction at either the raw, "
                "CLT-normalized, or null-corrected level. "
                "CPCV threshold selection shows no detectable advantage over random "
                "threshold assignment. The result is consistent with a sample-size artifact."
            )

        summary["verdict"] = verdict

    else:
        summary["verdict"] = "INSUFFICIENT_DATA"

    # ── Final metadata ──
    peak_ram = peak_ram_gb()
    elapsed_min = (time.time() - T0_GLOBAL) / 60.0
    summary["peak_ram_gb"] = peak_ram
    summary["elapsed_min"] = elapsed_min

    out_json = os.path.join(OUT_DIR, "summary_s4a_v2.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    heartbeat(f"Saved {out_json}")

    # ── Human-readable output ──
    print("\n" + "=" * 72)
    print("S4a-v2 RESULTS  (leakage-free + null-corrected variance comparison)")
    print("=" * 72)
    print(f"  Instruments total:            {summary.get('n_instruments_total', 0)}")
    print(f"  Degenerate/excluded:          {summary.get('n_degenerate', 0)}")
    print(f"  Clean:                        {summary.get('n_clean', 0)}")
    print(f"  Effective cross-section N:    {summary.get('eff_n_cross_section', np.nan):.1f}")
    print()
    print(f"  *** DISCLOSED CONTEXT ***")
    print(f"  Median strategy PF (no filter): {summary.get('disclosed_median_full_pf_no_filter', np.nan):.4f}")
    print(f"  (this study tests ESTIMATOR VARIANCE of a loss-making strategy, not edge)")
    print()
    if "median_cpcv_var" in summary:
        print(f"  --- Variance (raw — TAUTOLOGICAL) ---")
        print(f"  Median CPCV var:              {summary['median_cpcv_var']:.4f}")
        print(f"  Median single var:            {summary['median_single_var']:.4f}")
        print(f"  Median raw var reduction:     {summary['median_raw_var_red']:+.4f}")
        st = summary['sign_test_raw']
        print(f"  Sign test raw:                {st['n_pos']}/{st['n_total']}"
              f"  p_raw={st['p_raw']:.4f}  p_effn={st['p_effn']:.4f}")
        print(f"  NOTE: {summary.get('tautology_note','')[:90]}...")
        print()
        print(f"  --- CLT-normalized variance (var * n_oos, apples-to-apples) ---")
        print(f"  Median CPCV var_norm:         {summary['median_cpcv_var_norm']:.2f}")
        print(f"  Median single var_norm:       {summary['median_single_var_norm']:.2f}")
        print(f"  Median norm var reduction:    {summary['median_norm_var_red']:+.4f}")
        sn = summary['sign_test_norm']
        print(f"  Sign test norm:               {sn['n_pos']}/{sn['n_total']}"
              f"  p_raw={sn['p_raw']:.4f}  p_effn={sn['p_effn']:.4f}")
        print()
        print(f"  --- Residual over permuted-threshold null ---")
        print(f"  Median null norm var red:     {summary.get('median_null_norm_var_red', np.nan):+.4f}")
        print(f"  Median residual var red:      {summary.get('median_residual_var_red', np.nan):+.4f}")
        sr = summary.get('sign_test_residual', {})
        print(f"  Sign test residual:           {sr.get('n_pos','?')}/{sr.get('n_total','?')}"
              f"  p_raw={sr.get('p_raw', np.nan):.4f}  p_effn={sr.get('p_effn', np.nan):.4f}")
        print()
        print(f"  --- Threshold-selection sensitivity (replaces old 'bias') ---")
        print(f"  Median CPCV sensitivity:      {summary.get('median_thresh_sens_cpcv', np.nan):.4f}")
        print(f"  Median single sensitivity:    {summary.get('median_thresh_sens_single', np.nan):.4f}")
        print(f"  (ref = holdout forward block, thresh=0.50; neither arm touches it)")
    print()
    print(f"  VERDICT: {summary.get('verdict', 'N/A')}")
    print()
    print(f"  Peak RAM: {peak_ram:.2f} GB")
    print(f"  Elapsed:  {elapsed_min:.1f} min")
    print()
    print(f"  Artifacts:")
    print(f"    {out_pq}")
    print(f"    {out_json}")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    main()
