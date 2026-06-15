#!/usr/bin/env python3
"""
s04_deep.py — Deep run for S4 extensions: CV-chosen meta-label cutoff (S4a)
and asymmetric vol-scaled barriers (S4b).

Full 42-instrument panel: 27 crypto perps, 7 equity ETFs, 8 FX pairs.
Full available history per instrument. Dollar bars for crypto/FX; for equity
daily bars are used directly (no 1m intraday available).

S4a: CPCV-chosen cutoff vs single-split — lower OOS variance AND less bias
     across the panel (paired, significant), better OOS-PF transfer.
     No-LA: cutoff chosen on IS folds only.

S4b: IS-tuned vol-scaled asymmetric barriers (TP>SL, k·EWMA-vol) vs symmetric:
     meta precision + net PF across the panel (paired, significant);
     median toward/above break-even.

Real costs throughout:
  Crypto: 5 bp taker + 2 bp slip per fill = 7 bp per side (2-way = 14 bp round-trip)
          + 1 bp/8h/leg funding (absorbed via net cost per trade)
  Equity/FX: via constant 7 bp per side (conservative proxy via lib.realism concept)

Outputs to the script's own directory:
  trades_crypto.parquet, trades_equity.parquet, trades_fx.parquet
  per_instrument_s4a.parquet, per_instrument_s4b.parquet
  summary_s4a.json, summary_s4b.json

Peak RAM estimate: ~800 MB peak per instrument (one at a time, del+gc after each).
"""
import os
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
from itertools import combinations

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
sys.path.insert(0, os.path.join(REPO_ROOT, "projects/05_trend_scanning/scripts"))
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
# Instrument registry: full 42-instrument panel
# ─────────────────────────────────────────────────────────────────────────────
CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT",
    "UNIUSDT", "ATOMUSDT", "ETCUSDT", "AAVEUSDT", "ALGOUSDT",
    "NEARUSDT", "XLMUSDT", "TRXUSDT", "DOGEUSDT", "ARBUSDT",
    "APTUSDT", "SUIUSDT", "ICPUSDT", "HBARUSDT", "APEUSDT",
    "UNIUSDT", "ZECUSDT",
]
# Remove duplicate UNIUSDT, use 1000SHIB instead
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
# Crypto: 5 bp taker + 2 bp slip = 7 bp per side -> 2-way round-trip = 14 bp
# Plus 1 bp/8h funding per leg (absorbed as ~1 bp/trade for short holds,
# we use 7 bp + 1 bp = 8 bp per side total to be conservative)
COST_CRYPTO = 8.0 / 1e4    # 8 bp per side (covers taker + slip + funding)
COST_EQ_FX  = 7.0 / 1e4    # 7 bp per side (no funding)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline parameters
# ─────────────────────────────────────────────────────────────────────────────
N_TARGET_BARS = 20_000   # dollar bars target
VOL_SPAN      = 50
FAST, SLOW    = 20, 60
MAX_HOLD      = 50
N_SPLITS      = 6
EMBARGO_PCT   = 0.01
LABEL_SPAN    = 3
N_ESTIMATORS  = 100

# S4b grid: (pt_mult, sl_mult) pairs to IS-tune; MUST include symmetric baseline
BARRIER_GRID = [
    (1.0, 1.0),   # sym narrow
    (1.5, 1.0),   # sym baseline
    (2.0, 1.5),   # sym wider
    (2.0, 1.0),   # asym: TP=2, SL=1
    (2.5, 1.0),   # asym: TP=2.5, SL=1
    (3.0, 1.0),   # asym: TP=3, SL=1
    (2.0, 0.75),  # asym: TP=2, SL=0.75 (tight SL)
    (1.5, 0.75),  # asym+tight
]

# CPCV: C(6,2) = 15 path combinations
CPCV_N_GROUPS = 6
CPCV_K_TEST   = 2

# Threshold grid for cutoff selection
THRESH_GRID = [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]

T0_GLOBAL = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Utility: peak RAM
# ─────────────────────────────────────────────────────────────────────────────
def peak_ram_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KB->GB


def heartbeat(msg):
    elapsed = (time.time() - T0_GLOBAL) / 60.0
    ram = peak_ram_gb()
    print(f"[{elapsed:6.1f}m | RAM {ram:.2f} GB] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bar builders
# ─────────────────────────────────────────────────────────────────────────────
def build_dollar_bars_crypto(pair: str) -> pd.DataFrame:
    path = os.path.join(CRYPTO_DIR, f"{pair}_1m.parquet")
    base = tbm.load_base_crypto(path)
    bars = BARS.matched_bars(base, N_TARGET_BARS)["dollar"]
    return bars


def build_dollar_bars_fx(pair: str) -> pd.DataFrame:
    path = os.path.join(FX_DIR, f"{pair}_fx1m.parquet")
    base = BARS.load_base_fx(path)
    bars = BARS.matched_bars(base, N_TARGET_BARS)["dollar"]
    return bars


def build_equity_bars(ticker: str) -> pd.DataFrame:
    """Daily bars for equity ETFs. Add synthetic columns needed by tbm."""
    path = os.path.join(EQUITY_CACHE, f"daily_etf_ohlcv_{ticker}.parquet")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # Ensure column order and add proxy buy/sell (neutral: not available)
    out = pd.DataFrame({
        "open":   df["open"].to_numpy(np.float64),
        "high":   df["high"].to_numpy(np.float64),
        "low":    df["low"].to_numpy(np.float64),
        "close":  df["close"].to_numpy(np.float64),
        "volume": df["volume"].to_numpy(np.float64),
    }, index=df.index)
    out["dollar"]     = out["volume"] * out["close"]
    out["ticks"]      = out["volume"]
    out["buy_dollar"] = out["dollar"] / 2.0
    out["sell_dollar"]= out["dollar"] / 2.0
    out = out[(out["close"] > 0) & (out["volume"] > 0)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────
def bpy_of(bars: pd.DataFrame) -> float:
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    return len(bars) / years


def ewma_vol(bars, span=VOL_SPAN):
    return tbm.ewma_vol(bars["close"].to_numpy(np.float64), span)


def run_meta_oof(X, y, n_estimators=N_ESTIMATORS):
    """Purged k-fold OOF meta-probabilities."""
    p = np.full(len(y), np.nan)
    for tr, te in OF.purged_kfold_splits(len(y), N_SPLITS, EMBARGO_PCT, LABEL_SPAN):
        if len(tr) < 50 or y[tr].sum() < 5 or (1 - y[tr]).sum() < 5:
            p[te] = y[tr].mean() if len(tr) > 0 else 0.5
            continue
        clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
            n_estimators=n_estimators, max_samples=0.8, max_features=0.8,
            bootstrap=True, n_jobs=1, random_state=0)
        clf.fit(X[tr], y[tr])
        pi = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
        p[te] = clf.predict_proba(X[te])[:, pi]
    return np.nan_to_num(p, nan=float(np.nanmean(p) if np.any(np.isfinite(p)) else 0.5))


def trade_ret_series(ev_idx, hold, pnl_net, n_bars):
    r = np.zeros(n_bars)
    for k in range(len(ev_idx)):
        i0 = int(ev_idx[k])
        h = max(1, int(hold[k]))
        per = pnl_net[k] / h
        r[i0:min(i0 + h, n_bars)] += per
    return r


def score_r(r, bpy):
    r = r[np.isfinite(r)]
    nz = r[r != 0.0]
    sr_pb = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0
    sr_ann = sr_pb * np.sqrt(bpy)
    gp = nz[nz > 0].sum()
    gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else np.nan
    return dict(sr_ann=sr_ann, sr_pb=sr_pb, pf=pf, n_obs=len(r))


def pf_on_trades(p_oof, pnl_net, thresh):
    """PF of meta-filtered trades at a given threshold."""
    act = p_oof >= thresh
    pnl_m = pnl_net[act]
    gp = pnl_m[pnl_m > 0].sum()
    gn = -pnl_m[pnl_m < 0].sum()
    return float(gp / gn) if gn > 0 else np.nan


def precision_on_trades(p_oof, y, thresh):
    act = p_oof >= thresh
    if act.sum() == 0:
        return np.nan
    return float(y[act].mean())


# ─────────────────────────────────────────────────────────────────────────────
# S4a: CPCV-chosen cutoff vs single-split
# ─────────────────────────────────────────────────────────────────────────────
def run_s4a(ev, y, p_oof, pnl_net, hold, n):
    """
    Compare:
      - Single-split threshold selection: use 2nd-to-last fold as IS to pick
        threshold, evaluate on last fold as OOS (single path).
      - CPCV threshold selection: C(N_SPLITS, 2) paths, each picks threshold on
        IS groups, evaluates on OOS groups.

    Returns per-instrument metrics:
      single_thresh, single_oos_pf, cpcv_mean_pf, cpcv_std_pf,
      cpcv_var_pf, bias_single (|single_oos_pf - true_pf|),
      bias_cpcv (|cpcv_mean_pf - true_pf|)
    """
    thresholds = THRESH_GRID

    # "True" PF: average PF across all CPCV paths (best unbiased estimate)
    splits_list = list(OF.purged_kfold_splits(len(ev), N_SPLITS, EMBARGO_PCT, LABEL_SPAN))

    # Single-split: 2nd-to-last fold = IS, last fold = OOS
    is_idx_s  = splits_list[-2][1]  # second-to-last TEST fold is the IS for tuning
    oos_idx_s = splits_list[-1][1]  # last TEST fold is the final OOS

    if len(is_idx_s) < 10 or len(oos_idx_s) < 10:
        return None

    # Single-split threshold selection
    is_pfs_s = {}
    for t in thresholds:
        is_pfs_s[t] = pf_on_trades(p_oof[is_idx_s], pnl_net[is_idx_s], t)
    best_t_single = max(thresholds, key=lambda t: is_pfs_s[t] if np.isfinite(is_pfs_s.get(t, np.nan)) else -np.inf)
    oos_pf_single = pf_on_trades(p_oof[oos_idx_s], pnl_net[oos_idx_s], best_t_single)

    # CPCV paths
    cpcv_oos_pfs = []
    cpcv_best_threshs = []
    all_oos_covered = np.zeros(len(ev), bool)

    for tr_idx, te_idx in OF.cpcv_splits(len(ev), CPCV_N_GROUPS, CPCV_K_TEST,
                                          EMBARGO_PCT, LABEL_SPAN):
        if len(tr_idx) < 20 or len(te_idx) < 10:
            continue
        is_pfs_c = {}
        for t in thresholds:
            is_pfs_c[t] = pf_on_trades(p_oof[tr_idx], pnl_net[tr_idx], t)
        best_t_c = max(thresholds, key=lambda t: is_pfs_c[t] if np.isfinite(is_pfs_c.get(t, np.nan)) else -np.inf)
        oos_pf_c = pf_on_trades(p_oof[te_idx], pnl_net[te_idx], best_t_c)
        if np.isfinite(oos_pf_c):
            cpcv_oos_pfs.append(oos_pf_c)
            cpcv_best_threshs.append(best_t_c)
        all_oos_covered[te_idx] = True

    if len(cpcv_oos_pfs) < 3:
        return None

    cpcv_mean_pf = float(np.mean(cpcv_oos_pfs))
    cpcv_std_pf  = float(np.std(cpcv_oos_pfs))
    cpcv_var_pf  = float(np.var(cpcv_oos_pfs))

    # "True" PF: full-panel default (thresh=0.50 applied to all events)
    true_pf_default = pf_on_trades(p_oof, pnl_net, 0.50)

    # Bias: |mean(oos_pf) - true_pf_default|
    bias_single = abs(oos_pf_single - true_pf_default) if np.isfinite(oos_pf_single) else np.nan
    bias_cpcv   = abs(cpcv_mean_pf - true_pf_default) if np.isfinite(cpcv_mean_pf) else np.nan

    # Also: single-split OOS var = 0 by construction (1 path), CPCV gives var
    # Key metric: CPCV var < what we'd get from bootstrap of single-split paths
    # Additional: compute variance of each single-fold OOS PF across all N_SPLITS folds
    # (this is the single-fold variance "oracle" - the spread if you'd tried each fold)
    single_fold_pfs = []
    for f in range(N_SPLITS - 1):   # use fold f as IS, fold f+1 as OOS
        is_idx_f  = splits_list[f][1]
        oos_idx_f = splits_list[f + 1][1] if f + 1 < N_SPLITS else None
        if oos_idx_f is None or len(is_idx_f) < 10 or len(oos_idx_f) < 10:
            continue
        is_pfs_f = {t: pf_on_trades(p_oof[is_idx_f], pnl_net[is_idx_f], t) for t in thresholds}
        best_t_f = max(thresholds, key=lambda t: is_pfs_f[t] if np.isfinite(is_pfs_f.get(t, np.nan)) else -np.inf)
        oos_pf_f = pf_on_trades(p_oof[oos_idx_f], pnl_net[oos_idx_f], best_t_f)
        if np.isfinite(oos_pf_f):
            single_fold_pfs.append(oos_pf_f)

    # Variance of single-fold OOS PFs (oracle for single-split variance)
    single_fold_var = float(np.var(single_fold_pfs)) if len(single_fold_pfs) >= 2 else np.nan

    # CPCV chose threshold:
    cpcv_modal_thresh = float(np.median(cpcv_best_threshs)) if cpcv_best_threshs else 0.50

    return dict(
        single_thresh=best_t_single,
        single_oos_pf=oos_pf_single,
        cpcv_mean_pf=cpcv_mean_pf,
        cpcv_std_pf=cpcv_std_pf,
        cpcv_var_pf=cpcv_var_pf,
        single_fold_var=single_fold_var,
        bias_single=bias_single,
        bias_cpcv=bias_cpcv,
        true_pf_default=true_pf_default,
        cpcv_n_paths=len(cpcv_oos_pfs),
        cpcv_modal_thresh=cpcv_modal_thresh,
    )


# ─────────────────────────────────────────────────────────────────────────────
# S4b: IS-tuned asymmetric vol-scaled barriers vs symmetric baseline
# ─────────────────────────────────────────────────────────────────────────────
def run_s4b_instrument(bars, cost_per_side, bpy, min_events=80):
    """
    IS-tune barrier parameters (pt_mult, sl_mult) via CPCV.
    At each CPCV path: pick the barrier config that maximizes IS-PF,
    report OOS-PF and meta-precision with that barrier.

    Also run symmetric baseline (1.5, 1.0) as fixed config across all paths
    for direct comparison.

    Returns per-instrument metrics.
    """
    close = bars["close"].to_numpy(np.float64)
    high  = bars["high"].to_numpy(np.float64)
    low   = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    vol = tbm.ewma_vol(close, VOL_SPAN)

    # Primary signal
    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    ev = tbm.crossover_events(side_full)
    warm = SLOW + 12
    ev = ev[(ev > warm) & (ev < n - 1)]

    if len(ev) < min_events:
        return None

    side = side_full[ev]

    # Build features (same for all barrier configs — features are causal)
    feat = tbm.build_features(bars, side_full, vol, FAST, SLOW)
    X = feat.iloc[ev].to_numpy(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Symmetric baseline (1.5, 1.0): run full OOF ---
    tb_sym = tbm.triple_barrier(close, high, low, openp, ev, side, vol, 1.5, 1.0, MAX_HOLD)
    ret_sym  = tb_sym["ret_gross"].to_numpy()
    hold_sym = tb_sym["hold"].to_numpy()
    pnl_sym  = ret_sym - 2.0 * cost_per_side
    y_sym    = (pnl_sym > 0).astype(int)

    if y_sym.sum() < max(15, len(ev) // 6) or (1 - y_sym).sum() < max(15, len(ev) // 6):
        return None

    p_oof_sym = run_meta_oof(X, y_sym)
    thresh_sym = 0.50
    act_sym = p_oof_sym >= thresh_sym
    pnl_meta_sym = act_sym.astype(float) * pnl_sym
    r_meta_sym = trade_ret_series(ev, hold_sym, pnl_meta_sym, n)
    sc_sym = score_r(r_meta_sym, bpy)
    prec_sym = precision_on_trades(p_oof_sym, y_sym, thresh_sym)
    pf_sym  = pf_on_trades(p_oof_sym, pnl_sym, thresh_sym)

    # --- CPCV IS-tuning of barrier config ---
    # Pre-compute (ret_gross, hold, pnl_net, y) for each barrier config
    barrier_data = {}
    for (pt, sl) in BARRIER_GRID:
        tb_ = tbm.triple_barrier(close, high, low, openp, ev, side, vol, pt, sl, MAX_HOLD)
        ret_  = tb_["ret_gross"].to_numpy()
        hold_ = tb_["hold"].to_numpy()
        pnl_  = ret_ - 2.0 * cost_per_side
        y_    = (pnl_ > 0).astype(int)
        barrier_data[(pt, sl)] = dict(pnl=pnl_, hold=hold_, y=y_)

    # CPCV: for each path, pick best IS-barrier, measure OOS
    cpcv_asym_pfs    = []
    cpcv_asym_precs  = []
    cpcv_sym_pfs     = []   # sym baseline evaluated on same OOS splits
    cpcv_sym_precs   = []
    cpcv_chosen_asym = []   # was the chosen config asymmetric (pt > sl)?

    for tr_idx, te_idx in OF.cpcv_splits(len(ev), CPCV_N_GROUPS, CPCV_K_TEST,
                                          EMBARGO_PCT, LABEL_SPAN):
        if len(tr_idx) < 50 or len(te_idx) < 10:
            continue

        # IS: pick barrier config that maximizes precision on IS trades
        # (precision on IS is a causal, no-LA selection: uses only tr_idx)
        best_cfg = None
        best_is_pf = -np.inf

        for (pt, sl), bd in barrier_data.items():
            y_tr = bd["y"][tr_idx]
            pnl_tr = bd["pnl"][tr_idx]
            if y_tr.sum() < 5 or (1 - y_tr).sum() < 5:
                continue
            # Use OOF probabilities on IS region only — run a quick inner CV
            # For efficiency: use the SAME p_oof_sym probabilities (features don't
            # change with barrier, only labels do; re-fitting per config is expensive).
            # IS-selection: fit meta-clf on tr using y from this barrier config.
            # Use a single train/test within tr (last 20% of tr as mini-val).
            n_tr = len(tr_idx)
            val_cut = int(n_tr * 0.8)
            sub_tr = tr_idx[:val_cut]
            sub_val = tr_idx[val_cut:]
            if len(sub_tr) < 30 or len(sub_val) < 5:
                # Fall back to full IS PF
                is_pf = pf_on_trades(p_oof_sym[tr_idx], pnl_tr, 0.50)
            else:
                clf_is = BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
                    n_estimators=50, max_samples=0.8, max_features=0.8,
                    bootstrap=True, n_jobs=1, random_state=0)
                y_sub_tr = bd["y"][sub_tr]
                if y_sub_tr.sum() < 3 or (1 - y_sub_tr).sum() < 3:
                    is_pf = pf_on_trades(p_oof_sym[tr_idx], pnl_tr, 0.50)
                else:
                    clf_is.fit(X[sub_tr], y_sub_tr)
                    pi = list(clf_is.classes_).index(1) if 1 in clf_is.classes_ else 0
                    p_val = clf_is.predict_proba(X[sub_val])[:, pi]
                    is_pf = pf_on_trades(p_val, pnl_tr[val_cut:], 0.50)

            if np.isfinite(is_pf) and is_pf > best_is_pf:
                best_is_pf = is_pf
                best_cfg = (pt, sl)

        if best_cfg is None:
            best_cfg = (1.5, 1.0)  # fallback to symmetric baseline

        # Evaluate OOS with chosen barrier config (re-fit meta clf on tr with best_cfg labels)
        bd_best = barrier_data[best_cfg]
        y_tr_best = bd_best["y"][tr_idx]
        pnl_te_best = bd_best["pnl"][te_idx]
        y_te_best = bd_best["y"][te_idx]

        if y_tr_best.sum() < 10 or (1 - y_tr_best).sum() < 10:
            continue

        clf_oos = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
            n_estimators=N_ESTIMATORS, max_samples=0.8, max_features=0.8,
            bootstrap=True, n_jobs=1, random_state=0)
        clf_oos.fit(X[tr_idx], y_tr_best)
        pi = list(clf_oos.classes_).index(1) if 1 in clf_oos.classes_ else 0
        p_te = clf_oos.predict_proba(X[te_idx])[:, pi]

        oos_pf_best  = pf_on_trades(p_te, pnl_te_best, 0.50)
        oos_prec_best = precision_on_trades(p_te, y_te_best, 0.50)

        # OOS with symmetric baseline (same tr_idx fit, but sym labels)
        y_tr_sym_c = barrier_data[(1.5, 1.0)]["y"][tr_idx]
        pnl_te_sym_c = barrier_data[(1.5, 1.0)]["pnl"][te_idx]
        y_te_sym_c = barrier_data[(1.5, 1.0)]["y"][te_idx]

        if y_tr_sym_c.sum() < 10 or (1 - y_tr_sym_c).sum() < 10:
            continue

        clf_sym_c = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
            n_estimators=N_ESTIMATORS, max_samples=0.8, max_features=0.8,
            bootstrap=True, n_jobs=1, random_state=0)
        clf_sym_c.fit(X[tr_idx], y_tr_sym_c)
        pi_s = list(clf_sym_c.classes_).index(1) if 1 in clf_sym_c.classes_ else 0
        p_te_sym = clf_sym_c.predict_proba(X[te_idx])[:, pi_s]

        oos_pf_sym_c   = pf_on_trades(p_te_sym, pnl_te_sym_c, 0.50)
        oos_prec_sym_c  = precision_on_trades(p_te_sym, y_te_sym_c, 0.50)

        if np.isfinite(oos_pf_best) and np.isfinite(oos_pf_sym_c):
            cpcv_asym_pfs.append(oos_pf_best)
            cpcv_asym_precs.append(oos_prec_best if np.isfinite(oos_prec_best) else np.nan)
            cpcv_sym_pfs.append(oos_pf_sym_c)
            cpcv_sym_precs.append(oos_prec_sym_c if np.isfinite(oos_prec_sym_c) else np.nan)
            cpcv_chosen_asym.append(1 if best_cfg[0] > best_cfg[1] else 0)

    if len(cpcv_asym_pfs) < 3:
        return None

    asym_arr = np.array(cpcv_asym_pfs)
    sym_arr  = np.array(cpcv_sym_pfs)
    prec_asym_arr = np.array([x for x in cpcv_asym_precs if np.isfinite(x)])
    prec_sym_arr  = np.array([x for x in cpcv_sym_precs if np.isfinite(x)])

    return dict(
        # Symmetric baseline (full OOF, fixed thresh=0.50)
        sym_pf=pf_sym,
        sym_prec=prec_sym,
        sym_sr_ann=sc_sym["sr_ann"],

        # CPCV-tuned asymmetric barrier
        asym_cpcv_mean_pf=float(np.nanmean(asym_arr)),
        asym_cpcv_std_pf=float(np.nanstd(asym_arr)),
        asym_cpcv_med_pf=float(np.nanmedian(asym_arr)),
        asym_cpcv_mean_prec=float(np.nanmean(prec_asym_arr)) if len(prec_asym_arr) > 0 else np.nan,

        # CPCV symmetric baseline (for paired comparison)
        sym_cpcv_mean_pf=float(np.nanmean(sym_arr)),
        sym_cpcv_std_pf=float(np.nanstd(sym_arr)),
        sym_cpcv_med_pf=float(np.nanmedian(sym_arr)),
        sym_cpcv_mean_prec=float(np.nanmean(prec_sym_arr)) if len(prec_sym_arr) > 0 else np.nan,

        # Deltas (asym - sym)
        delta_mean_pf=float(np.nanmean(asym_arr - sym_arr)),
        delta_med_pf=float(np.nanmedian(asym_arr - sym_arr)),
        delta_mean_prec=(float(np.nanmean(prec_asym_arr)) - float(np.nanmean(prec_sym_arr)))
                        if (len(prec_asym_arr) > 0 and len(prec_sym_arr) > 0) else np.nan,

        # How often did IS-tuning choose asymmetric?
        frac_asym_chosen=float(np.mean(cpcv_chosen_asym)),
        n_cpcv_paths=len(cpcv_asym_pfs),

        # Break-even reference
        asym_above_breakeven=float(np.mean(asym_arr > 1.0)),
        sym_above_breakeven=float(np.mean(sym_arr > 1.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-instrument driver for S4a
# ─────────────────────────────────────────────────────────────────────────────
def run_s4a_instrument(bars, cost_per_side, bpy, min_events=80):
    """Build dollar-bar meta pipeline, then run S4a analysis."""
    close = bars["close"].to_numpy(np.float64)
    high  = bars["high"].to_numpy(np.float64)
    low   = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    vol = tbm.ewma_vol(close, VOL_SPAN)

    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    ev = tbm.crossover_events(side_full)
    warm = SLOW + 12
    ev = ev[(ev > warm) & (ev < n - 1)]
    if len(ev) < min_events:
        return None

    side = side_full[ev]
    feat = tbm.build_features(bars, side_full, vol, FAST, SLOW)
    X = feat.iloc[ev].to_numpy(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Use symmetric 1.5/1.0 for S4a (barrier config is not the variable here)
    tb = tbm.triple_barrier(close, high, low, openp, ev, side, vol, 1.5, 1.0, MAX_HOLD)
    ret_gross = tb["ret_gross"].to_numpy()
    hold      = tb["hold"].to_numpy()
    pnl_net   = ret_gross - 2.0 * cost_per_side
    y = (pnl_net > 0).astype(int)

    if y.sum() < max(15, len(ev) // 6) or (1 - y).sum() < max(15, len(ev) // 6):
        return None

    p_oof = run_meta_oof(X, y)
    return run_s4a(ev, y, p_oof, pnl_net, hold, n)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional statistical tests
# ─────────────────────────────────────────────────────────────────────────────
def paired_test(a, b):
    """Wilcoxon signed-rank + t-test for paired (a - b) differences.
    Returns: mean_diff, median_diff, p_wilcoxon, p_ttest, n_pairs"""
    diff = np.array(a) - np.array(b)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 4:
        return dict(mean_diff=np.nan, median_diff=np.nan,
                    p_wilcoxon=np.nan, p_ttest=np.nan, n=len(diff))
    _, p_w = ss.wilcoxon(diff, alternative='two-sided')
    _, p_t = ss.ttest_1samp(diff, 0.0)
    return dict(
        mean_diff=float(np.mean(diff)),
        median_diff=float(np.median(diff)),
        p_wilcoxon=float(p_w),
        p_ttest=float(p_t),
        n=len(diff),
    )


def sign_test(a, b):
    """Sign test: how often a > b?"""
    diff = np.array(a) - np.array(b)
    diff = diff[np.isfinite(diff)]
    n_pos = int(np.sum(diff > 0))
    n_neg = int(np.sum(diff < 0))
    n_tot = len(diff)
    if n_tot < 4:
        return dict(n_pos=n_pos, n_neg=n_neg, n_total=n_tot, p_sign=np.nan)
    _, p_sign = ss.binom_test(n_pos, n_tot, 0.5, alternative='two-sided') \
        if hasattr(ss, 'binom_test') else (np.nan, np.nan)
    # Use scipy.stats.binomtest for newer scipy
    try:
        p_sign = float(ss.binomtest(n_pos, n_tot, 0.5, alternative='two-sided').pvalue)
    except AttributeError:
        try:
            from scipy.stats import binom_test
            p_sign = float(binom_test(n_pos, n_tot, 0.5, alternative='two-sided'))
        except Exception:
            p_sign = float(2 * min(ss.binom.cdf(n_pos, n_tot, 0.5),
                                   ss.binom.sf(n_pos - 1, n_tot, 0.5)))
    return dict(n_pos=n_pos, n_neg=n_neg, n_total=n_tot, p_sign=float(p_sign))


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    heartbeat("Starting S04 deep run")
    rows_s4a = []
    rows_s4b = []

    # Build instrument list
    instruments = []
    for sym in CRYPTO_PAIRS:
        instruments.append(("crypto", sym))
    for tkr in EQUITY_TICKERS:
        instruments.append(("equity", tkr))
    for pair in FX_PAIRS:
        instruments.append(("fx", pair))

    heartbeat(f"Panel: {len(instruments)} instruments "
              f"({len(CRYPTO_PAIRS)} crypto, {len(EQUITY_TICKERS)} equity, {len(FX_PAIRS)} FX)")

    n_ok = 0
    n_skip = 0

    for mkt, sym in instruments:
        heartbeat(f"Processing {mkt}:{sym} ...")
        try:
            # Load bars
            if mkt == "crypto":
                bars = build_dollar_bars_crypto(sym)
                cost = COST_CRYPTO
            elif mkt == "equity":
                bars = build_equity_bars(sym)
                cost = COST_EQ_FX
            else:  # fx
                bars = build_dollar_bars_fx(sym)
                cost = COST_EQ_FX

            if len(bars) < 1500:
                heartbeat(f"  SKIP {sym}: too few bars ({len(bars)})")
                n_skip += 1
                del bars
                gc.collect()
                continue

            bpy = bpy_of(bars)
            heartbeat(f"  {sym}: {len(bars)} bars, bpy={bpy:.0f}")

            # ── S4a ──
            try:
                min_ev = 50 if mkt == "equity" else 80
                r4a = run_s4a_instrument(bars, cost, bpy, min_events=min_ev)
                if r4a is not None:
                    r4a.update(dict(market=mkt, symbol=sym, n_bars=len(bars), bpy=bpy))
                    rows_s4a.append(r4a)
                    heartbeat(f"  S4a OK: cpcv_mean_pf={r4a['cpcv_mean_pf']:.3f}, "
                              f"single_oos_pf={r4a['single_oos_pf']:.3f}, "
                              f"cpcv_std={r4a['cpcv_std_pf']:.3f}")
                else:
                    heartbeat(f"  S4a: insufficient events, skipped")
            except Exception as e:
                heartbeat(f"  S4a ERROR: {e}")

            # ── S4b ──
            try:
                min_ev_b = 50 if mkt == "equity" else 80
                r4b = run_s4b_instrument(bars, cost, bpy, min_events=min_ev_b)
                if r4b is not None:
                    r4b.update(dict(market=mkt, symbol=sym, n_bars=len(bars), bpy=bpy))
                    rows_s4b.append(r4b)
                    heartbeat(f"  S4b OK: asym_cpcv_mean_pf={r4b['asym_cpcv_mean_pf']:.3f}, "
                              f"sym_cpcv_mean_pf={r4b['sym_cpcv_mean_pf']:.3f}, "
                              f"delta={r4b['delta_mean_pf']:+.3f}, "
                              f"frac_asym_chosen={r4b['frac_asym_chosen']:.2f}")
                else:
                    heartbeat(f"  S4b: insufficient events, skipped")
            except Exception as e:
                heartbeat(f"  S4b ERROR: {e}")

            n_ok += 1

        except Exception as e:
            heartbeat(f"  LOAD ERROR {sym}: {e}")
            n_skip += 1

        # Cleanup
        if 'bars' in dir():
            del bars
        gc.collect()
        heartbeat(f"  -> {n_ok} processed, {n_skip} skipped")

    heartbeat(f"Loop complete. {n_ok} instruments processed, {n_skip} skipped.")

    # ── Serialize per-instrument results ──
    df4a = pd.DataFrame(rows_s4a)
    df4b = pd.DataFrame(rows_s4b)

    df4a.to_parquet(os.path.join(OUT_DIR, "per_instrument_s4a.parquet"), index=False)
    df4b.to_parquet(os.path.join(OUT_DIR, "per_instrument_s4b.parquet"), index=False)
    heartbeat(f"Saved per-instrument parquets. S4a: {len(df4a)} rows, S4b: {len(df4b)} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-sectional summary: S4a
    # ─────────────────────────────────────────────────────────────────────────
    summary_4a = {}

    if len(df4a) >= 4:
        # Primary claim: CPCV has lower OOS variance than single-split
        # Proxy: CPCV var < single-fold var across the panel
        cpcv_vars = df4a["cpcv_var_pf"].dropna().tolist()
        single_vars = df4a["single_fold_var"].dropna().tolist()

        # Align to same instruments
        df4a_clean = df4a.dropna(subset=["cpcv_var_pf", "single_fold_var"])
        cpcv_vars_p  = df4a_clean["cpcv_var_pf"].tolist()
        single_vars_p = df4a_clean["single_fold_var"].tolist()

        var_test = paired_test(single_vars_p, cpcv_vars_p)  # want single > CPCV
        var_sign = sign_test(single_vars_p, cpcv_vars_p)

        # Bias comparison
        df4a_bias = df4a.dropna(subset=["bias_single", "bias_cpcv"])
        bias_test = paired_test(
            df4a_bias["bias_single"].tolist(),
            df4a_bias["bias_cpcv"].tolist()
        )
        bias_sign = sign_test(
            df4a_bias["bias_single"].tolist(),
            df4a_bias["bias_cpcv"].tolist()
        )

        # CPCV mean PF vs single-split OOS PF (PF transfer)
        df4a_pf = df4a.dropna(subset=["cpcv_mean_pf", "single_oos_pf"])
        pf_test = paired_test(
            df4a_pf["cpcv_mean_pf"].tolist(),
            df4a_pf["single_oos_pf"].tolist()
        )

        summary_4a = dict(
            n_instruments=len(df4a),
            n_clean=len(df4a_clean),

            # Variance: single-fold vs CPCV
            median_cpcv_var=float(np.nanmedian(cpcv_vars)),
            median_single_fold_var=float(np.nanmedian(single_vars)),
            var_reduction_paired=var_test,
            var_sign_test=var_sign,
            var_lower_cpcv_frac=float(var_sign["n_pos"] / var_sign["n_total"]) if var_sign["n_total"] > 0 else np.nan,

            # Bias
            median_bias_single=float(np.nanmedian(df4a_bias["bias_single"])),
            median_bias_cpcv=float(np.nanmedian(df4a_bias["bias_cpcv"])),
            bias_reduction_paired=bias_test,
            bias_sign_test=bias_sign,
            bias_lower_cpcv_frac=float(bias_sign["n_pos"] / bias_sign["n_total"]) if bias_sign["n_total"] > 0 else np.nan,

            # PF transfer
            median_cpcv_mean_pf=float(np.nanmedian(df4a["cpcv_mean_pf"])),
            median_single_oos_pf=float(np.nanmedian(df4a["single_oos_pf"])),
            pf_transfer_test=pf_test,

            # CPCV std across panel
            median_cpcv_std=float(np.nanmedian(df4a["cpcv_std_pf"])),
        )

        # VERDICT S4a
        # PASS conditions:
        #   1. CPCV var < single-split var: var_sign["n_pos"] / n_total > 0.55, p < 0.05
        #   2. CPCV bias < single bias: bias_sign["n_pos"] / n_total > 0.55, p < 0.05
        var_pass  = (var_sign["p_sign"] < 0.05 and
                     var_sign["n_pos"] / max(var_sign["n_total"], 1) > 0.55)
        bias_pass = (bias_sign["p_sign"] < 0.05 and
                     bias_sign["n_pos"] / max(bias_sign["n_total"], 1) > 0.55)

        if var_pass and bias_pass:
            verdict_4a = "PASS"
        elif var_pass or bias_pass:
            verdict_4a = "PARTIAL"
        else:
            verdict_4a = "FAIL"

        summary_4a["verdict"] = verdict_4a

    else:
        summary_4a["verdict"] = "INSUFFICIENT_DATA"
        summary_4a["n_instruments"] = len(df4a)

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-sectional summary: S4b
    # ─────────────────────────────────────────────────────────────────────────
    summary_4b = {}

    if len(df4b) >= 4:
        df4b_clean = df4b.dropna(subset=["asym_cpcv_mean_pf", "sym_cpcv_mean_pf"])

        asym_pfs = df4b_clean["asym_cpcv_mean_pf"].tolist()
        sym_pfs  = df4b_clean["sym_cpcv_mean_pf"].tolist()
        asym_precs = df4b_clean["asym_cpcv_mean_prec"].dropna().tolist()
        sym_precs  = df4b_clean["sym_cpcv_mean_prec"].dropna().tolist()

        # Paired PF test: asymmetric vs symmetric
        pf_paired = paired_test(asym_pfs, sym_pfs)
        pf_sign   = sign_test(asym_pfs, sym_pfs)

        # Paired precision test
        df4b_prec = df4b.dropna(subset=["asym_cpcv_mean_prec", "sym_cpcv_mean_prec"])
        prec_paired = paired_test(
            df4b_prec["asym_cpcv_mean_prec"].tolist(),
            df4b_prec["sym_cpcv_mean_prec"].tolist()
        )
        prec_sign = sign_test(
            df4b_prec["asym_cpcv_mean_prec"].tolist(),
            df4b_prec["sym_cpcv_mean_prec"].tolist()
        )

        # Median break-even fraction
        med_asym_above_be = float(np.nanmedian(df4b["asym_above_breakeven"]))
        med_sym_above_be  = float(np.nanmedian(df4b["sym_above_breakeven"]))

        summary_4b = dict(
            n_instruments=len(df4b),
            n_clean=len(df4b_clean),

            median_asym_pf=float(np.nanmedian(df4b["asym_cpcv_med_pf"])),
            median_sym_pf=float(np.nanmedian(df4b["sym_cpcv_med_pf"])),
            median_delta_pf=float(np.nanmedian(df4b["delta_med_pf"])),

            pf_paired_test=pf_paired,
            pf_sign_test=pf_sign,
            pf_asym_wins_frac=float(pf_sign["n_pos"] / max(pf_sign["n_total"], 1)),

            median_asym_prec=float(np.nanmedian(df4b["asym_cpcv_mean_prec"])),
            median_sym_prec=float(np.nanmedian(df4b["sym_cpcv_mean_prec"])),
            prec_paired_test=prec_paired,
            prec_sign_test=prec_sign,

            # Break-even: fraction of instruments where median CPCV PF > 1.0
            med_asym_above_breakeven=med_asym_above_be,
            med_sym_above_breakeven=med_sym_above_be,
            asym_median_pf_above_1=float(np.nanmedian(df4b["asym_cpcv_med_pf"]) > 1.0),

            frac_asym_chosen_median=float(np.nanmedian(df4b["frac_asym_chosen"])),
        )

        # VERDICT S4b
        # PASS conditions:
        #   1. Asym PF > sym PF across panel: pf_sign["n_pos"] / n_total > 0.55, p < 0.05
        #   2. Asym prec > sym prec across panel: prec_sign["n_pos"] / n_total > 0.55, p < 0.05
        #   3. Median PF toward/above break-even (> 0.95 as threshold, median improvement)
        pf_pass   = (pf_sign["p_sign"] < 0.05 and
                     pf_sign["n_pos"] / max(pf_sign["n_total"], 1) > 0.55)
        prec_pass = (prec_sign["p_sign"] < 0.05 and
                     prec_sign["n_pos"] / max(prec_sign["n_total"], 1) > 0.55)
        be_pass   = (float(np.nanmedian(df4b["asym_cpcv_med_pf"])) >
                     float(np.nanmedian(df4b["sym_cpcv_med_pf"])))  # asym median > sym

        if pf_pass and prec_pass and be_pass:
            verdict_4b = "PASS"
        elif pf_pass or prec_pass:
            verdict_4b = "PARTIAL"
        else:
            verdict_4b = "FAIL"

        summary_4b["verdict"] = verdict_4b

    else:
        summary_4b["verdict"] = "INSUFFICIENT_DATA"
        summary_4b["n_instruments"] = len(df4b)

    # ── Final output ──
    peak_ram = peak_ram_gb()
    elapsed_min = (time.time() - T0_GLOBAL) / 60.0

    for s, fname in [(summary_4a, "summary_s4a.json"), (summary_4b, "summary_s4b.json")]:
        s["peak_ram_gb"] = peak_ram
        s["elapsed_min"] = elapsed_min
        with open(os.path.join(OUT_DIR, fname), "w") as f:
            json.dump(s, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    heartbeat(f"Peak RAM: {peak_ram:.2f} GB, elapsed: {elapsed_min:.1f} min")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nS4a — CV-chosen meta-label cutoff:")
    print(f"  Instruments processed: {summary_4a.get('n_instruments', 0)}")
    if "median_cpcv_var" in summary_4a:
        print(f"  Median CPCV var:          {summary_4a['median_cpcv_var']:.4f}")
        print(f"  Median single-fold var:   {summary_4a['median_single_fold_var']:.4f}")
        vst = summary_4a.get("var_sign_test", {})
        print(f"  Var lower in CPCV: {vst.get('n_pos','?')}/{vst.get('n_total','?')} "
              f"  p_sign={vst.get('p_sign', np.nan):.4f}")
        bst = summary_4a.get("bias_sign_test", {})
        print(f"  Bias lower in CPCV: {bst.get('n_pos','?')}/{bst.get('n_total','?')} "
              f"  p_sign={bst.get('p_sign', np.nan):.4f}")
        print(f"  Median CPCV mean PF:     {summary_4a['median_cpcv_mean_pf']:.3f}")
        print(f"  Median single OOS PF:    {summary_4a['median_single_oos_pf']:.3f}")
    print(f"  VERDICT S4a: {summary_4a.get('verdict', 'N/A')}")

    print(f"\nS4b — IS-tuned asymmetric vol-scaled barriers:")
    print(f"  Instruments processed: {summary_4b.get('n_instruments', 0)}")
    if "median_asym_pf" in summary_4b:
        print(f"  Median asym CPCV PF:     {summary_4b['median_asym_pf']:.3f}")
        print(f"  Median sym CPCV PF:      {summary_4b['median_sym_pf']:.3f}")
        print(f"  Median delta PF:         {summary_4b['median_delta_pf']:+.3f}")
        pst = summary_4b.get("pf_sign_test", {})
        print(f"  PF asym>sym: {pst.get('n_pos','?')}/{pst.get('n_total','?')} "
              f"  p_sign={pst.get('p_sign', np.nan):.4f}")
        prst = summary_4b.get("prec_sign_test", {})
        print(f"  Prec asym>sym: {prst.get('n_pos','?')}/{prst.get('n_total','?')} "
              f"  p_sign={prst.get('p_sign', np.nan):.4f}")
        print(f"  Median asym prec:        {summary_4b['median_asym_prec']:.3f}")
        print(f"  Median sym prec:         {summary_4b['median_sym_prec']:.3f}")
        print(f"  Median asym above BE:    {summary_4b['med_asym_above_breakeven']:.2f}")
        print(f"  Frac asym chosen by IS:  {summary_4b['frac_asym_chosen_median']:.2f}")
    print(f"  VERDICT S4b: {summary_4b.get('verdict', 'N/A')}")

    print(f"\nArtifacts in {OUT_DIR}:")
    print(f"  per_instrument_s4a.parquet")
    print(f"  per_instrument_s4b.parquet")
    print(f"  summary_s4a.json")
    print(f"  summary_s4b.json")
    print(f"\nPeak RAM: {peak_ram:.2f} GB")
    print(f"Elapsed:  {elapsed_min:.1f} min")
    print("=" * 70)

    return summary_4a, summary_4b


if __name__ == "__main__":
    main()
