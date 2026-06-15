#!/usr/bin/env python3
"""
S3 deep run — FFD vs practitioner shortcuts
Preregistered bar: FFD strictly dominates price-minus-EMA and rolling-z-score
on the JOINT (stationarity pass-rate, memory-retained) frontier for the large majority.

Design:
- Full cross-section: 505+ crypto perps at 1h (binance_um), full history per instrument
- Three transforms of log-price, all strictly causal:
    1. FFD at per-asset min-d  (min_d_search via lib/fracdiff.py)
    2. price - slow causal EMA (spans 200, 500)
    3. rolling causal z-score  (windows 200, 500)
- Stationarity: ADF at 95% (stat < crit_5%)
- Memory: abs(Pearson corr of transform with log-price level)
- Dominance (per instrument): FFD is BOTH stationary AND has higher |corr| than shortcut
  (shortcut must match stationarity to be comparable; if shortcut fails stationarity,
  FFD wins trivially)
- Report: fraction where FFD dominates, fraction where shortcut matches/beats,
  cross-sectional stationarity pass-rates and memory medians, sign-test p-value.

RAM: one series at a time, del+gc after each. Peak ~50 MB per symbol.
Threads: pinned to 1 everywhere.

Output: written to the script's own directory.
  per_instrument.parquet  — one row per symbol x shortcut_variant
  summary.json            — aggregate stats, verdict
  run.log                 — heartbeat log
"""
import os
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import sys, gc, time, json, warnings, logging
import numpy as np
import pandas as pd
from scipy import stats as ss
from numba import njit

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_PERP_1H
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)
import fracdiff as FD

# ── Numba JIT kernels (compiled on first call) ────────────────────────────────
@njit
def _rollz_nb(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score, Numba-compiled."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.nan
    for i in range(window - 1, n):
        mu = 0.0
        for j in range(i - window + 1, i + 1):
            mu += x[j]
        mu /= window
        var = 0.0
        for j in range(i - window + 1, i + 1):
            var += (x[j] - mu) * (x[j] - mu)
        sd = (var / (window - 1)) ** 0.5
        if sd > 0.0:
            out[i] = (x[i] - mu) / sd
    return out


# Warm-up JIT compile (with a tiny dummy array)
_dummy = np.zeros(300, dtype=np.float64)
_rollz_nb(_dummy, 200)
del _dummy

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = CRYPTO_PERP_1H
OUT_DIR  = HERE
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_BARS = 2000          # minimum 1h bars to include an instrument
D_GRID   = np.round(np.arange(0.0, 1.01, 0.05), 2)
TAU      = 1e-4          # weight-truncation threshold (tighter for hourly series)
EMA_SPANS   = [200, 500]
ROLL_WINS   = [200, 500]
HEARTBEAT   = 50         # log every N instruments


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_log_price(path: str) -> np.ndarray | None:
    """Load close prices from parquet, return log-price as float64 array."""
    try:
        df = pd.read_parquet(path, columns=["close"])
        c = df["close"].to_numpy(dtype=float)
        c = c[np.isfinite(c) & (c > 0)]
        if len(c) < MIN_BARS:
            return None
        return np.log(c)
    except Exception as e:
        log.warning(f"  load error {path}: {e}")
        return None


def adf_pass(x: np.ndarray) -> tuple[float, float, bool]:
    """ADF on x; returns (stat, crit_5pct, passed). causal — no lookahead."""
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return np.nan, np.nan, False
    try:
        from statsmodels.tsa.stattools import adfuller
        res = adfuller(x, maxlag=1, regression="c", autolag=None)
        stat  = float(res[0])
        crit5 = float(res[4]["5%"])
        return stat, crit5, bool(stat < crit5)
    except Exception:
        return np.nan, np.nan, False


def memory_corr(transform: np.ndarray, level: np.ndarray) -> float:
    """abs(Pearson corr) between transform and log-price level, on common valid support."""
    mask = np.isfinite(transform) & np.isfinite(level)
    if mask.sum() < 20:
        return np.nan
    return float(abs(np.corrcoef(transform[mask], level[mask])[0, 1]))


def causal_ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EWMA (alpha = 2/(span+1)), returns x - EMA(x, span)."""
    alpha = 2.0 / (span + 1)
    ema = np.empty_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return x - ema


def causal_rollz(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score of x with given window (Numba-accelerated)."""
    return _rollz_nb(np.asarray(x, dtype=np.float64), window)


# ── Per-instrument analysis ───────────────────────────────────────────────────
def analyze_instrument(symbol: str, lp: np.ndarray) -> list[dict]:
    """
    Returns a list of row-dicts (one per shortcut variant) for this instrument.
    Each row records the FFD result and one shortcut result so we can do
    per-row dominance comparisons cleanly.
    """
    # ── FFD min-d ──────────────────────────────────────────────────────────
    res = FD.min_d_search(lp, d_grid=D_GRID, tau=TAU)
    d_star    = res["d_star"]   # may be nan if no d passes ADF

    if not np.isnan(d_star):
        ffd_x        = FD.ffd(lp, d_star, tau=TAU)
        ffd_stat, ffd_crit5, ffd_stationary = adf_pass(ffd_x[np.isfinite(ffd_x)])
        ffd_memory   = memory_corr(ffd_x, lp)
    else:
        ffd_x        = np.full_like(lp, np.nan)
        ffd_stat     = np.nan
        ffd_crit5    = np.nan
        ffd_stationary = False
        ffd_memory   = np.nan

    rows = []

    # ── EMA shortcuts ──────────────────────────────────────────────────────
    for span in EMA_SPANS:
        ema_diff = causal_ema(lp, span)
        ema_stat, ema_crit5, ema_stationary = adf_pass(ema_diff[np.isfinite(ema_diff)])
        ema_memory = memory_corr(ema_diff, lp)

        # dominance: FFD BOTH stationary AND higher memory than shortcut
        # When shortcut fails stationarity, FFD trivially wins on stationarity axis
        # When shortcut passes stationarity, compare memory
        if ffd_stationary and not ema_stationary:
            dom = "ffd_stat_wins"
        elif ffd_stationary and ema_stationary:
            # both stationary — compare memory
            if np.isfinite(ffd_memory) and np.isfinite(ema_memory):
                if ffd_memory > ema_memory:
                    dom = "ffd_mem_wins"
                elif ffd_memory < ema_memory:
                    dom = "shortcut_mem_wins"
                else:
                    dom = "tie"
            else:
                dom = "insufficient_data"
        elif not ffd_stationary and ema_stationary:
            dom = "shortcut_stat_wins"
        else:  # both fail
            dom = "both_fail_stat"

        rows.append(dict(
            symbol=symbol,
            n_bars=len(lp),
            shortcut_type="ema",
            shortcut_param=span,
            d_star=d_star,
            ffd_stationary=int(ffd_stationary),
            ffd_stat=ffd_stat,
            ffd_crit5=ffd_crit5,
            ffd_memory=ffd_memory,
            sc_stationary=int(ema_stationary),
            sc_stat=ema_stat,
            sc_crit5=ema_crit5,
            sc_memory=ema_memory,
            dominance=dom,
        ))

    # ── Rolling z-score shortcuts ──────────────────────────────────────────
    for win in ROLL_WINS:
        rz = causal_rollz(lp, win)
        rz_stat, rz_crit5, rz_stationary = adf_pass(rz[np.isfinite(rz)])
        rz_memory = memory_corr(rz, lp)

        if ffd_stationary and not rz_stationary:
            dom = "ffd_stat_wins"
        elif ffd_stationary and rz_stationary:
            if np.isfinite(ffd_memory) and np.isfinite(rz_memory):
                if ffd_memory > rz_memory:
                    dom = "ffd_mem_wins"
                elif ffd_memory < rz_memory:
                    dom = "shortcut_mem_wins"
                else:
                    dom = "tie"
            else:
                dom = "insufficient_data"
        elif not ffd_stationary and rz_stationary:
            dom = "shortcut_stat_wins"
        else:
            dom = "both_fail_stat"

        rows.append(dict(
            symbol=symbol,
            n_bars=len(lp),
            shortcut_type="rollz",
            shortcut_param=win,
            d_star=d_star,
            ffd_stationary=int(ffd_stationary),
            ffd_stat=ffd_stat,
            ffd_crit5=ffd_crit5,
            ffd_memory=ffd_memory,
            sc_stationary=int(rz_stationary),
            sc_stat=rz_stat,
            sc_crit5=rz_crit5,
            sc_memory=rz_memory,
            dominance=dom,
        ))

    return rows


# ── Build instrument list ─────────────────────────────────────────────────────
all_files = sorted(os.listdir(DATA_DIR))
all_syms  = [f.replace("_1h.parquet", "") for f in all_files if f.endswith("_1h.parquet")]
log.info(f"Total files in data dir: {len(all_syms)}")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows   = []
n_skipped  = 0
n_done     = 0
t_start    = time.time()

for i, sym in enumerate(all_syms):
    path = os.path.join(DATA_DIR, f"{sym}_1h.parquet")
    lp   = load_log_price(path)

    if lp is None:
        n_skipped += 1
        continue

    try:
        rows = analyze_instrument(sym, lp)
        all_rows.extend(rows)
        n_done += 1
    except Exception as e:
        log.warning(f"  ERROR on {sym}: {e}")
        n_skipped += 1

    del lp
    gc.collect()

    if (i + 1) % HEARTBEAT == 0:
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1)
        remaining = (len(all_syms) - i - 1) / max(rate, 1e-9)
        log.info(
            f"[{i+1}/{len(all_syms)}] done={n_done} skipped={n_skipped} "
            f"elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={remaining:.0f}s"
        )

log.info(f"Loop complete. done={n_done} skipped={n_skipped} total_rows={len(all_rows)}")

# ── Save per-instrument parquet ───────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df.to_parquet(os.path.join(OUT_DIR, "per_instrument.parquet"), index=False)
log.info("Saved per_instrument.parquet")

# ── Aggregate statistics ──────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame) -> dict:
    results = {}
    n_instruments = df["symbol"].nunique()
    results["n_instruments"] = n_instruments
    results["n_rows"] = len(df)

    # ── Per-shortcut-type aggregation ────────────────────────────────────
    for sc_type in ["ema", "rollz"]:
        for param in (EMA_SPANS if sc_type == "ema" else ROLL_WINS):
            key = f"{sc_type}_{param}"
            sub = df[(df["shortcut_type"] == sc_type) & (df["shortcut_param"] == param)].copy()
            n = len(sub)
            if n == 0:
                continue

            # Stationarity pass-rates
            ffd_stat_rate = sub["ffd_stationary"].mean()
            sc_stat_rate  = sub["sc_stationary"].mean()

            # Memory medians (abs corr) — only where the transform is stationary
            ffd_mem_median = sub.loc[sub["ffd_stationary"] == 1, "ffd_memory"].median()
            sc_mem_median  = sub.loc[sub["sc_stationary"] == 1, "sc_memory"].median()

            # Dominance counts
            dom_counts = sub["dominance"].value_counts().to_dict()

            # FFD joint dominance: stationary AND higher memory
            # Cases: ffd_stat_wins (shortcut not stationary)
            #        ffd_mem_wins  (both stationary, FFD has more memory)
            ffd_dom_n    = dom_counts.get("ffd_stat_wins", 0) + dom_counts.get("ffd_mem_wins", 0)
            sc_dom_n     = dom_counts.get("shortcut_stat_wins", 0) + dom_counts.get("shortcut_mem_wins", 0)
            tie_n        = dom_counts.get("tie", 0)
            both_fail_n  = dom_counts.get("both_fail_stat", 0)
            insuf_n      = dom_counts.get("insufficient_data", 0)

            ffd_dom_frac   = ffd_dom_n / n
            sc_dom_frac    = sc_dom_n  / n

            # Sign test: on instruments where both passes stationarity,
            # is FFD memory > shortcut memory?
            both_stat = sub[(sub["ffd_stationary"] == 1) & (sub["sc_stationary"] == 1)].copy()
            n_both_stat = len(both_stat)
            if n_both_stat > 0:
                both_stat["ffd_wins_mem"] = (both_stat["ffd_memory"] > both_stat["sc_memory"]).astype(int)
                n_ffd_wins_mem = both_stat["ffd_wins_mem"].sum()
                # sign test: H0 = p=0.5
                sign_p = ss.binomtest(int(n_ffd_wins_mem), int(n_both_stat), 0.5, alternative="greater").pvalue
            else:
                n_ffd_wins_mem = 0
                sign_p = np.nan

            # Overall sign test including stationarity failures
            # H0: FFD does NOT dominate (p=0.5); direction = +1 for FFD win, -1 for shortcut win
            # Use only decisive cases
            decisive = sub[sub["dominance"].isin(["ffd_stat_wins", "ffd_mem_wins",
                                                    "shortcut_stat_wins", "shortcut_mem_wins"])]
            n_decisive = len(decisive)
            n_ffd_decisive = decisive["dominance"].isin(["ffd_stat_wins", "ffd_mem_wins"]).sum()
            overall_sign_p = ss.binomtest(int(n_ffd_decisive), int(n_decisive), 0.5,
                                          alternative="greater").pvalue if n_decisive > 0 else np.nan

            results[key] = dict(
                n_instruments=n,
                ffd_stat_rate=round(ffd_stat_rate, 4),
                sc_stat_rate=round(sc_stat_rate, 4),
                ffd_mem_median=round(float(ffd_mem_median), 4) if np.isfinite(ffd_mem_median) else None,
                sc_mem_median=round(float(sc_mem_median), 4) if np.isfinite(sc_mem_median) else None,
                dom_counts=dom_counts,
                ffd_dom_n=int(ffd_dom_n),
                sc_dom_n=int(sc_dom_n),
                tie_n=int(tie_n),
                both_fail_n=int(both_fail_n),
                ffd_dom_frac=round(ffd_dom_frac, 4),
                sc_dom_frac=round(sc_dom_frac, 4),
                n_both_stationary=int(n_both_stat),
                n_ffd_wins_memory=int(n_ffd_wins_mem),
                sign_test_p_memory=round(float(sign_p), 6) if np.isfinite(sign_p) else None,
                sign_test_p_overall=round(float(overall_sign_p), 6) if np.isfinite(overall_sign_p) else None,
            )

    # ── d* distribution ──────────────────────────────────────────────────────
    # use one row per symbol (first row per symbol)
    sym_df = df.drop_duplicates("symbol")[["symbol", "n_bars", "d_star"]].copy()
    results["d_star"] = dict(
        median=round(float(sym_df["d_star"].median()), 4),
        p25=round(float(sym_df["d_star"].quantile(0.25)), 4),
        p75=round(float(sym_df["d_star"].quantile(0.75)), 4),
        frac_found=round(float(sym_df["d_star"].notna().mean()), 4),
        n_found=int(sym_df["d_star"].notna().sum()),
        n_total=int(len(sym_df)),
    )
    results["n_bars_median"] = int(sym_df["n_bars"].median())

    # ── Verdict ──────────────────────────────────────────────────────────────
    # Bar: FFD strictly dominates for the LARGE MAJORITY (>50%) AND
    # overall sign-test p < 0.05 for both shortcut types
    verdict_parts = []
    pass_bar = True

    for sc_type in ["ema", "rollz"]:
        for param in (EMA_SPANS if sc_type == "ema" else ROLL_WINS):
            key = f"{sc_type}_{param}"
            if key not in results:
                continue
            r = results[key]
            dom_frac = r["ffd_dom_frac"]
            sign_p   = r.get("sign_test_p_overall")
            sig      = (sign_p is not None and sign_p < 0.05)
            majority = dom_frac > 0.50
            verdict_parts.append(
                f"{key}: ffd_dom={dom_frac:.1%} sign_p={sign_p} "
                f"{'MAJORITY+SIG' if (majority and sig) else ('MAJORITY_NOT_SIG' if majority else 'MINORITY')}"
            )
            if not (majority and sig):
                pass_bar = False

    results["verdict_lines"] = verdict_parts
    results["verdict"] = "PASS" if pass_bar else "FAIL"
    results["verdict_notes"] = (
        "FFD strictly dominates (stationarity + memory) for >50% of instruments "
        "with sign-test p<0.05 for all four shortcut variants."
        if pass_bar else
        "At least one shortcut variant matches or beats FFD for a substantial fraction "
        "of instruments. Honest negative: a shortcut can match FFD."
    )

    return results


stats = aggregate(df)
log.info(f"VERDICT: {stats['verdict']}")
for line in stats.get("verdict_lines", []):
    log.info(f"  {line}")

# ── Save summary.json ─────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(stats, f, indent=2, default=str)
log.info(f"Saved summary.json")

# ── Human-readable report ─────────────────────────────────────────────────────
report_lines = [
    "S3 — FFD vs practitioner shortcuts (deep, full cross-section)",
    f"Instruments processed: {stats['n_instruments']}  (skipped {n_skipped})",
    f"Bars per instrument (median): {stats['n_bars_median']}",
    f"d* found: {stats['d_star']['n_found']}/{stats['d_star']['n_total']} "
    f"({stats['d_star']['frac_found']:.1%})  "
    f"d* median={stats['d_star']['median']} IQR=[{stats['d_star']['p25']},{stats['d_star']['p75']}]",
    "",
]

for sc_type in ["ema", "rollz"]:
    for param in (EMA_SPANS if sc_type == "ema" else ROLL_WINS):
        key = f"{sc_type}_{param}"
        if key not in stats:
            continue
        r = stats[key]
        report_lines += [
            f"-- {key} (n={r['n_instruments']}) --",
            f"  Stationarity pass-rate: FFD={r['ffd_stat_rate']:.1%}  shortcut={r['sc_stat_rate']:.1%}",
            f"  Memory (abs corr with level, stationary-only median): "
            f"FFD={r['ffd_mem_median']}  shortcut={r['sc_mem_median']}",
            f"  Dominance counts: {r['dom_counts']}",
            f"  FFD dominates: {r['ffd_dom_n']}/{r['n_instruments']} ({r['ffd_dom_frac']:.1%})",
            f"  Shortcut matches/beats: {r['sc_dom_n']}/{r['n_instruments']} ({r['sc_dom_frac']:.1%})",
            f"  Both-stationary memory sign test: "
            f"FFD wins={r['n_ffd_wins_memory']}/{r['n_both_stationary']} p={r['sign_test_p_memory']}",
            f"  Overall sign test (decisive cases): p={r['sign_test_p_overall']}",
            "",
        ]

report_lines += [
    f"VERDICT: {stats['verdict']}",
    stats["verdict_notes"],
]

report_text = "\n".join(report_lines)
report_path = os.path.join(OUT_DIR, "report.txt")
with open(report_path, "w") as f:
    f.write(report_text)
print(report_text)
log.info(f"Saved report.txt")
log.info("Done.")
