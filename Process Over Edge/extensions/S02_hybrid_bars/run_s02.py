#!/usr/bin/env python3
"""
S2 — Hybrid activity-or-time bars (deep, full cross-section)
Peak RAM estimate: ~300 MB (one instrument at a time)

Locked bar (PREREGISTRATION.md S2):
  PASS: hybrid beats plain volume/dollar bars on median excess kurtosis AND
        a quiet-period tail metric across the full cross-section (sign test sig.),
        with no bar-count-uniformity regression.

Cross-section:
  - 27 crypto perps  CRYPTO_1M/*_1m.parquet
  - 9 equity ETFs    EQUITY_1M/*.csv.gz  (RTH session-handled)
  - 8 FX pairs       FX_1M/*_fx1m.parquet

Full available history per instrument.

Outputs under the script's own directory:
  results.parquet   — per-instrument per-method metrics
  summary.json      — sign-test counts, medians, verdict
"""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                  NUMBA_NUM_THREADS="1")

import sys, gc, warnings, json, time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_1M, FX_1M, EQUITY_1M
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)
import bars as B

warnings.filterwarnings("ignore")

# ── Output dir ───────────────────────────────────────────────────────────────
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = f"{OUT_DIR}/run.log"

def hb(msg: str):
    """Heartbeat: print with timestamp and flush."""
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


# ── Data paths ────────────────────────────────────────────────────────────────
CRYPTO_DIR = CRYPTO_1M
FX_DIR     = FX_1M
ETF_DIR    = EQUITY_1M

CRYPTO_FILES = sorted([
    f for f in os.listdir(CRYPTO_DIR) if f.endswith("_1m.parquet")
])
FX_FILES = sorted([
    f for f in os.listdir(FX_DIR) if f.endswith("_fx1m.parquet")
])
# Use consolidated ETF files (full history, RTH)
ETF_TICKERS = ["IWM", "QQQ", "SPY", "UVXY", "VXX", "XLE", "XLF", "XLK", "XLV"]

hb(f"Instrument counts — crypto:{len(CRYPTO_FILES)}  FX:{len(FX_FILES)}  ETF:{len(ETF_TICKERS)}")


# ── Time-cap settings ─────────────────────────────────────────────────────────
# Crypto/FX: max 240 1m-bars = 4h. Chosen as 4× the target bar spacing at
# N_TARGET_DAILY=24 bars/day (one bar every ~60min baseline).
# Equities: session is 390 1m-bars. Cap = 78 bars = 2h (2× baseline at N=24/day).
# Both values are set before seeing any instrument-level result.
TIME_CAP_CRYPTO_FX = 240   # minutes = 4h
TIME_CAP_EQUITY    = 78    # minutes ≈ 2h RTH

# How many bars per day to target (applied per instrument)
N_TARGET_DAILY = 24

# Quiet-period definition: lowest ACTIVITY_PCTILE of hours by bar-count
ACTIVITY_PCTILE = 20   # bottom 20% of hourly activity


# ── Helpers ───────────────────────────────────────────────────────────────────
def exkurt(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 50:
        return np.nan
    return float(ss.kurtosis(r, fisher=True))


def count_cv(bars: pd.DataFrame, tz: str | None = None) -> float:
    """CV of bar count per calendar day."""
    if tz is not None and bars.index.tz is not None:
        idx = bars.index.tz_convert(tz)
    else:
        idx = bars.index
    per = pd.Series(1, index=idx).resample("1D").sum()
    per = per[per > 0]
    if len(per) < 5:
        return np.nan
    return float(per.std(ddof=1) / per.mean())


def log_ret_series(bars: pd.DataFrame) -> np.ndarray:
    """Close-to-close log returns (raw, overnight included for crypto/FX)."""
    return np.diff(np.log(bars["close"].to_numpy()))


def session_log_ret(bars: pd.DataFrame) -> np.ndarray:
    """Within-session log returns for equities (drops overnight gap)."""
    sr = B.session_log_returns(bars)
    return sr.to_numpy()


def quiet_period_kurtosis(bars: pd.DataFrame, tz: str | None = None,
                           pctile: int = 20) -> float:
    """
    Kurtosis of bar returns restricted to the lowest-activity hours.

    Algorithm:
      1. Assign each bar an hour-of-week bucket (168 buckets).
      2. Count bars per bucket over the full sample.
      3. Identify the bottom `pctile`% of buckets by bar count.
      4. Compute kurtosis on returns within those buckets only.

    This isolates quiet periods where volume bars go stale and the time-cap
    of hybrid bars should prevent fat-tailed drift-then-jump returns.
    """
    if tz is not None and bars.index.tz is not None:
        idx_local = bars.index.tz_convert(tz)
    else:
        idx_local = bars.index

    # hour-of-week = dayofweek * 24 + hour
    dow = idx_local.dayofweek   # 0=Mon
    hr  = idx_local.hour
    bucket = dow * 24 + hr      # 0..167

    # Count bars per bucket
    bcount = pd.Series(bucket).value_counts()
    threshold_count = np.percentile(bcount.values, pctile)
    quiet_buckets = set(bcount[bcount <= threshold_count].index.tolist())

    mask = np.array([b in quiet_buckets for b in bucket])
    quiet_bars = bars.iloc[mask]
    if len(quiet_bars) < 50:
        return np.nan

    r = np.diff(np.log(quiet_bars["close"].to_numpy()))
    return exkurt(r)


# ── Hybrid bar builder ────────────────────────────────────────────────────────
def hybrid_dollar_time_bars(df: pd.DataFrame, n_target: int,
                             time_cap: int) -> pd.DataFrame:
    """
    Close a bar when EITHER:
      (a) cumulative dollar volume >= dollar_threshold, OR
      (b) elapsed base-bars >= time_cap.

    Dollar threshold matched to plain dollar bars (same total / n_target).
    time_cap is in base-bar (1m) units.

    Causal by construction: no future information used.
    """
    dol = df["quote_volume"].to_numpy()
    dol_thresh = dol.sum() / max(1, n_target)

    gid = np.zeros(len(dol), dtype=np.int64)
    cum_dol = 0.0
    elapsed  = 0
    bar_id   = 0

    for i in range(len(dol)):
        cum_dol += dol[i]
        elapsed  += 1
        if cum_dol >= dol_thresh or elapsed >= time_cap:
            bar_id  += 1
            cum_dol  = 0.0
            elapsed  = 0
        gid[i] = bar_id

    return B._aggregate(df, gid)


def hybrid_volume_time_bars(df: pd.DataFrame, n_target: int,
                             time_cap: int) -> pd.DataFrame:
    """
    Close a bar when EITHER:
      (a) cumulative volume >= volume_threshold, OR
      (b) elapsed base-bars >= time_cap.

    Volume threshold matched to plain volume bars.
    """
    vol = df["volume"].to_numpy()
    vol_thresh = vol.sum() / max(1, n_target)

    gid = np.zeros(len(vol), dtype=np.int64)
    cum_vol = 0.0
    elapsed  = 0
    bar_id   = 0

    for i in range(len(vol)):
        cum_vol += vol[i]
        elapsed += 1
        if cum_vol >= vol_thresh or elapsed >= time_cap:
            bar_id  += 1
            cum_vol  = 0.0
            elapsed  = 0
        gid[i] = bar_id

    return B._aggregate(df, gid)


# ── Per-instrument computation ────────────────────────────────────────────────
def process_instrument(df: pd.DataFrame, label: str, market: str,
                       time_cap: int, use_session: bool = False) -> dict:
    """
    Returns a dict of metrics for this instrument.
    Computes: plain volume bars, plain dollar bars, hybrid-dollar-time,
              hybrid-volume-time.

    Metrics per bar type:
      - exkurt: excess kurtosis of (session) log returns
      - exkurt_quiet: excess kurtosis of returns in quiet-period hours
      - cv: bar-count CV per calendar day

    sign-comparison flags (hybrid beats baseline):
      - hdt_beats_vol_ek: hybrid-dollar-time exkurt < plain_volume exkurt
      - hdt_beats_dol_ek: hybrid-dollar-time exkurt < plain_dollar exkurt
      - hvt_beats_vol_ek: hybrid-volume-time exkurt < plain_volume exkurt
      - hvt_beats_dol_ek: hybrid-volume-time exkurt < plain_dollar exkurt
      (same for _qp = quiet-period version)
      - uniformity flag: hybrid CV not materially worse than baseline
        defined as hybrid_cv <= baseline_cv * 1.20 (20% slack)
    """
    n_total = len(df)
    n_days  = max(1, (df.index[-1] - df.index[0]).days + 1)
    n_target = max(50, int(round(n_days * N_TARGET_DAILY)))

    # Determine timezone for equities session / day bucketing
    tz = "America/New_York" if use_session else None

    # ── Plain volume bars ──────────────────────────────────────────────────
    vol_thresh = df["volume"].sum() / n_target
    vb = B.threshold_bars(df, "volume", vol_thresh)
    vb_r = session_log_ret(vb) if use_session else log_ret_series(vb)
    vb_ek  = exkurt(vb_r)
    vb_qp  = quiet_period_kurtosis(vb, tz=tz, pctile=ACTIVITY_PCTILE)
    vb_cv  = count_cv(vb, tz=tz)
    del vb; gc.collect()

    # ── Plain dollar bars ──────────────────────────────────────────────────
    dol_thresh = df["quote_volume"].sum() / n_target
    db = B.threshold_bars(df, "quote_volume", dol_thresh)
    db_r = session_log_ret(db) if use_session else log_ret_series(db)
    db_ek  = exkurt(db_r)
    db_qp  = quiet_period_kurtosis(db, tz=tz, pctile=ACTIVITY_PCTILE)
    db_cv  = count_cv(db, tz=tz)
    del db; gc.collect()

    # ── Hybrid dollar-time bars ────────────────────────────────────────────
    hdt = hybrid_dollar_time_bars(df, n_target, time_cap)
    hdt_r = session_log_ret(hdt) if use_session else log_ret_series(hdt)
    hdt_ek  = exkurt(hdt_r)
    hdt_qp  = quiet_period_kurtosis(hdt, tz=tz, pctile=ACTIVITY_PCTILE)
    hdt_cv  = count_cv(hdt, tz=tz)
    n_hdt   = len(hdt)
    del hdt; gc.collect()

    # ── Hybrid volume-time bars ────────────────────────────────────────────
    hvt = hybrid_volume_time_bars(df, n_target, time_cap)
    hvt_r = session_log_ret(hvt) if use_session else log_ret_series(hvt)
    hvt_ek  = exkurt(hvt_r)
    hvt_qp  = quiet_period_kurtosis(hvt, tz=tz, pctile=ACTIVITY_PCTILE)
    hvt_cv  = count_cv(hvt, tz=tz)
    n_hvt   = len(hvt)
    del hvt; gc.collect()

    # ── Sign-comparison flags ──────────────────────────────────────────────
    def beats(x, y):
        """True if x < y (lower kurtosis = better), False if not, None if NaN."""
        if np.isnan(x) or np.isnan(y):
            return None
        return bool(x < y)

    def no_reg(h_cv, b_cv, slack=1.20):
        """True if hybrid CV not materially worse (within 20% of baseline)."""
        if np.isnan(h_cv) or np.isnan(b_cv) or b_cv <= 0:
            return None
        return bool(h_cv <= b_cv * slack)

    result = dict(
        label=label, market=market, n_rows=n_total, n_days=n_days,
        n_target=n_target, time_cap=time_cap,
        n_hdt=n_hdt, n_hvt=n_hvt,
        # plain baselines
        vb_ek=vb_ek,  vb_qp=vb_qp,  vb_cv=vb_cv,
        db_ek=db_ek,  db_qp=db_qp,  db_cv=db_cv,
        # hybrid metrics
        hdt_ek=hdt_ek, hdt_qp=hdt_qp, hdt_cv=hdt_cv,
        hvt_ek=hvt_ek, hvt_qp=hvt_qp, hvt_cv=hvt_cv,
        # ek deltas (hybrid - baseline, negative = improvement)
        hdt_vs_vb_ek_delta=_safe_sub(hdt_ek, vb_ek),
        hdt_vs_db_ek_delta=_safe_sub(hdt_ek, db_ek),
        hvt_vs_vb_ek_delta=_safe_sub(hvt_ek, vb_ek),
        hvt_vs_db_ek_delta=_safe_sub(hvt_ek, db_ek),
        # qp deltas
        hdt_vs_vb_qp_delta=_safe_sub(hdt_qp, vb_qp),
        hdt_vs_db_qp_delta=_safe_sub(hdt_qp, db_qp),
        hvt_vs_vb_qp_delta=_safe_sub(hvt_qp, vb_qp),
        hvt_vs_db_qp_delta=_safe_sub(hvt_qp, db_qp),
        # sign flags
        hdt_beats_vb_ek=beats(hdt_ek, vb_ek),
        hdt_beats_db_ek=beats(hdt_ek, db_ek),
        hvt_beats_vb_ek=beats(hvt_ek, vb_ek),
        hvt_beats_db_ek=beats(hvt_ek, db_ek),
        hdt_beats_vb_qp=beats(hdt_qp, vb_qp),
        hdt_beats_db_qp=beats(hdt_qp, db_qp),
        hvt_beats_vb_qp=beats(hvt_qp, vb_qp),
        hvt_beats_db_qp=beats(hvt_qp, db_qp),
        hdt_no_cv_regress_vs_vb=no_reg(hdt_cv, vb_cv),
        hdt_no_cv_regress_vs_db=no_reg(hdt_cv, db_cv),
        hvt_no_cv_regress_vs_vb=no_reg(hvt_cv, vb_cv),
        hvt_no_cv_regress_vs_db=no_reg(hvt_cv, db_cv),
    )
    return result


def _safe_sub(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)


# ── Sign-test helper ──────────────────────────────────────────────────────────
def sign_test(wins: int, total: int) -> float:
    """Two-sided binomial sign test p-value (H0: p=0.5)."""
    if total == 0:
        return np.nan
    result = ss.binomtest(wins, total, p=0.5, alternative="greater")
    return float(result.pvalue)


# ── Main loop ─────────────────────────────────────────────────────────────────
rows = []
t_start = time.time()

hb("=== Starting S2 deep run ===")

# -- CRYPTO --
hb(f"--- CRYPTO: {len(CRYPTO_FILES)} instruments ---")
for fname in CRYPTO_FILES:
    sym = fname.replace("_1m.parquet", "")
    path = f"{CRYPTO_DIR}/{fname}"
    hb(f"  Loading {sym}...")
    try:
        df = B.load_base(path)
        if len(df) < 20_000:
            hb(f"    skip: only {len(df)} rows")
            continue
        hb(f"    {sym}: {len(df)} rows, {(df.index[-1]-df.index[0]).days}d")
        row = process_instrument(df, sym, "crypto", TIME_CAP_CRYPTO_FX, use_session=False)
        rows.append(row)
        hb(f"    {sym}: hdt_ek={row['hdt_ek']:.2f} vs vb_ek={row['vb_ek']:.2f} db_ek={row['db_ek']:.2f}  "
           f"hdt_qp={row['hdt_qp']:.2f} vs vb_qp={row['vb_qp']:.2f}  "
           f"beats_vb={row['hdt_beats_vb_ek']} beats_db={row['hdt_beats_db_ek']}")
    except Exception as e:
        hb(f"    ERROR {sym}: {e}")
    finally:
        try: del df
        except: pass
        gc.collect()

# -- FX --
hb(f"--- FX: {len(FX_FILES)} instruments ---")
for fname in FX_FILES:
    sym = fname.replace("_fx1m.parquet", "")
    path = f"{FX_DIR}/{fname}"
    hb(f"  Loading {sym}...")
    try:
        df = B.load_base_fx(path)
        if len(df) < 20_000:
            hb(f"    skip: only {len(df)} rows")
            continue
        hb(f"    {sym}: {len(df)} rows, {(df.index[-1]-df.index[0]).days}d")
        row = process_instrument(df, sym, "fx", TIME_CAP_CRYPTO_FX, use_session=False)
        rows.append(row)
        hb(f"    {sym}: hdt_ek={row['hdt_ek']:.2f} vs vb_ek={row['vb_ek']:.2f} db_ek={row['db_ek']:.2f}  "
           f"beats_vb={row['hdt_beats_vb_ek']} beats_db={row['hdt_beats_db_ek']}")
    except Exception as e:
        hb(f"    ERROR {sym}: {e}")
    finally:
        try: del df
        except: pass
        gc.collect()

# -- EQUITIES --
hb(f"--- ETF: {len(ETF_TICKERS)} instruments ---")
for ticker in ETF_TICKERS:
    path = f"{ETF_DIR}/{ticker}.csv.gz"
    hb(f"  Loading {ticker}...")
    try:
        df = B.load_base_equity_etf(path, rth=True)
        if len(df) < 20_000:
            hb(f"    skip: only {len(df)} rows")
            continue
        hb(f"    {ticker}: {len(df)} rows, {(df.index[-1]-df.index[0]).days}d")
        row = process_instrument(df, ticker, "etf", TIME_CAP_EQUITY, use_session=True)
        rows.append(row)
        hb(f"    {ticker}: hdt_ek={row['hdt_ek']:.2f} vs vb_ek={row['vb_ek']:.2f} db_ek={row['db_ek']:.2f}  "
           f"beats_vb={row['hdt_beats_vb_ek']} beats_db={row['hdt_beats_db_ek']}")
    except Exception as e:
        hb(f"    ERROR {ticker}: {e}")
    finally:
        try: del df
        except: pass
        gc.collect()

hb(f"=== All instruments done in {(time.time()-t_start)/60:.1f}m, {len(rows)} results ===")

# ── Save per-instrument results ───────────────────────────────────────────────
df_res = pd.DataFrame(rows)
res_path = f"{OUT_DIR}/results.parquet"
df_res.to_parquet(res_path, index=False)
hb(f"Saved {res_path}")

# ── Sign tests ────────────────────────────────────────────────────────────────
def sign_counts(col: str):
    vals = df_res[col].dropna()
    wins  = int(vals.sum())
    total = int(len(vals))
    return wins, total

def sign_p(col: str):
    w, n = sign_counts(col)
    return sign_test(w, n)

metrics = {}

# Kurtosis sign tests
for tag, beat_col in [
    ("hdt_vs_vb_ek",  "hdt_beats_vb_ek"),
    ("hdt_vs_db_ek",  "hdt_beats_db_ek"),
    ("hvt_vs_vb_ek",  "hvt_beats_vb_ek"),
    ("hvt_vs_db_ek",  "hvt_beats_db_ek"),
    ("hdt_vs_vb_qp",  "hdt_beats_vb_qp"),
    ("hdt_vs_db_qp",  "hdt_beats_db_qp"),
    ("hvt_vs_vb_qp",  "hvt_beats_vb_qp"),
    ("hvt_vs_db_qp",  "hvt_beats_db_qp"),
]:
    w, n = sign_counts(beat_col)
    p    = sign_p(beat_col)
    metrics[tag] = {"wins": w, "total": n, "p_one_sided": round(float(p), 4)}

# Uniformity guard: fraction with no CV regression
for tag, col in [
    ("hdt_cv_guard_vs_vb", "hdt_no_cv_regress_vs_vb"),
    ("hdt_cv_guard_vs_db", "hdt_no_cv_regress_vs_db"),
    ("hvt_cv_guard_vs_vb", "hvt_no_cv_regress_vs_vb"),
    ("hvt_cv_guard_vs_db", "hvt_no_cv_regress_vs_db"),
]:
    vals = df_res[col].dropna()
    metrics[tag] = {
        "no_regress_frac": round(float(vals.mean()), 3),
        "n": int(len(vals)),
    }

# Medians across full cross-section
num_cols = ["vb_ek", "db_ek", "hdt_ek", "hvt_ek",
            "vb_qp", "db_qp", "hdt_qp", "hvt_qp",
            "vb_cv", "db_cv", "hdt_cv", "hvt_cv",
            "hdt_vs_vb_ek_delta", "hdt_vs_db_ek_delta",
            "hdt_vs_vb_qp_delta", "hdt_vs_db_qp_delta"]
medians = {}
for c in num_cols:
    if c in df_res.columns:
        v = df_res[c].dropna()
        medians[c] = {"median": round(float(v.median()), 3),
                      "q25":    round(float(v.quantile(0.25)), 3),
                      "q75":    round(float(v.quantile(0.75)), 3),
                      "n":      int(len(v))}

# Per-market breakdown
market_summary = {}
for mkt in ["crypto", "fx", "etf"]:
    sub = df_res[df_res["market"] == mkt]
    if len(sub) == 0:
        continue
    ms = {}
    for c in ["vb_ek", "db_ek", "hdt_ek", "hvt_ek",
              "hdt_beats_vb_ek", "hdt_beats_db_ek",
              "hdt_beats_vb_qp", "hdt_beats_db_qp"]:
        vals = sub[c].dropna()
        if c.startswith("hdt_beats") or c.startswith("hvt_beats"):
            ms[c] = {"wins": int(vals.sum()), "total": int(len(vals))}
        else:
            ms[c] = {"median": round(float(vals.median()), 3), "n": int(len(vals))}
    market_summary[mkt] = ms

# ── Verdict ───────────────────────────────────────────────────────────────────
# PASS conditions (per locked bar):
#   1. Hybrid beats volume bars on median excess kurtosis (sign test p < 0.05)
#   2. Hybrid beats dollar bars on median excess kurtosis (sign test p < 0.05)
#   3. Hybrid beats volume bars on quiet-period kurtosis (sign test p < 0.05)
#   4. Hybrid beats dollar bars on quiet-period kurtosis (sign test p < 0.05)
#   5. No material uniformity regression (>= 80% of instruments within 20% slack)
# We require conditions 1-4 to hold for EITHER hdt or hvt (the better hybrid).
# Condition 5 must hold for the winning hybrid.

def check_pass(tag_ek_vb, tag_ek_db, tag_qp_vb, tag_qp_db,
               uni_vb, uni_db, alpha=0.05):
    ek_vb = metrics[tag_ek_vb]["p_one_sided"] < alpha
    ek_db = metrics[tag_ek_db]["p_one_sided"] < alpha
    qp_vb = metrics[tag_qp_vb]["p_one_sided"] < alpha
    qp_db = metrics[tag_qp_db]["p_one_sided"] < alpha
    uni   = (metrics[uni_vb]["no_regress_frac"] >= 0.80 and
             metrics[uni_db]["no_regress_frac"] >= 0.80)
    return ek_vb, ek_db, qp_vb, qp_db, uni

hdt_conds = check_pass("hdt_vs_vb_ek", "hdt_vs_db_ek",
                        "hdt_vs_vb_qp", "hdt_vs_db_qp",
                        "hdt_cv_guard_vs_vb", "hdt_cv_guard_vs_db")
hvt_conds = check_pass("hvt_vs_vb_ek", "hvt_vs_db_ek",
                        "hvt_vs_vb_qp", "hvt_vs_db_qp",
                        "hvt_cv_guard_vs_vb", "hvt_cv_guard_vs_db")

hdt_pass = all(hdt_conds)
hvt_pass = all(hvt_conds)
overall_pass = hdt_pass or hvt_pass

verdict = "PASS" if overall_pass else "FAIL"
winning_hybrid = []
if hdt_pass: winning_hybrid.append("hybrid-dollar-time")
if hvt_pass: winning_hybrid.append("hybrid-volume-time")

verdict_detail = {
    "verdict": verdict,
    "winning_hybrids": winning_hybrid,
    "time_cap_crypto_fx_minutes": TIME_CAP_CRYPTO_FX,
    "time_cap_etf_minutes": TIME_CAP_EQUITY,
    "n_instruments": len(rows),
    "hdt_conditions": {
        "ek_beats_vol_bars": hdt_conds[0],
        "ek_beats_dol_bars": hdt_conds[1],
        "qp_beats_vol_bars": hdt_conds[2],
        "qp_beats_dol_bars": hdt_conds[3],
        "uniformity_guard":  hdt_conds[4],
    },
    "hvt_conditions": {
        "ek_beats_vol_bars": hvt_conds[0],
        "ek_beats_dol_bars": hvt_conds[1],
        "qp_beats_vol_bars": hvt_conds[2],
        "qp_beats_dol_bars": hvt_conds[3],
        "uniformity_guard":  hvt_conds[4],
    },
}

summary = {
    "verdict_detail": verdict_detail,
    "sign_tests": metrics,
    "medians": medians,
    "market_breakdown": market_summary,
}

sum_path = f"{OUT_DIR}/summary.json"
with open(sum_path, "w") as fh:
    json.dump(summary, fh, indent=2)
hb(f"Saved {sum_path}")

# ── Print results ─────────────────────────────────────────────────────────────
hb("\n=== S2 RESULTS ===")
hb(f"VERDICT: {verdict}")
if winning_hybrid:
    hb(f"Winning hybrid(s): {', '.join(winning_hybrid)}")
hb(f"N instruments: {len(rows)}")
hb(f"Time cap: crypto/FX={TIME_CAP_CRYPTO_FX}min, ETF={TIME_CAP_EQUITY}min")

hb("\n--- Median excess kurtosis (full cross-section) ---")
for label, col in [("vol bars", "vb_ek"), ("dollar bars", "db_ek"),
                   ("hybrid-dollar-time", "hdt_ek"), ("hybrid-vol-time", "hvt_ek")]:
    m = medians.get(col, {})
    hb(f"  {label:<22}: median={m.get('median','?'):.2f}  [p25={m.get('q25','?'):.2f}, p75={m.get('q75','?'):.2f}]  n={m.get('n','?')}")

hb("\n--- Median excess kurtosis (quiet periods) ---")
for label, col in [("vol bars", "vb_qp"), ("dollar bars", "db_qp"),
                   ("hybrid-dollar-time", "hdt_qp"), ("hybrid-vol-time", "hvt_qp")]:
    m = medians.get(col, {})
    hb(f"  {label:<22}: median={m.get('median','?'):.2f}  [p25={m.get('q25','?'):.2f}, p75={m.get('q75','?'):.2f}]")

hb("\n--- Sign tests (hybrid beats baseline, one-sided, H0: p=0.5) ---")
for tag, v in metrics.items():
    if "wins" in v:
        hb(f"  {tag:<32}: {v['wins']}/{v['total']}  p={v['p_one_sided']:.4f}")
    else:
        hb(f"  {tag:<32}: no-regress={v['no_regress_frac']:.2f}  n={v['n']}")

hb("\n--- HDT conditions met ---")
for k, v in verdict_detail["hdt_conditions"].items():
    hb(f"  {k}: {v}")
hb("--- HVT conditions met ---")
for k, v in verdict_detail["hvt_conditions"].items():
    hb(f"  {k}: {v}")

hb(f"\nArtifacts: {OUT_DIR}/")
hb(f"Total wall time: {(time.time()-t_start)/60:.1f}m")
