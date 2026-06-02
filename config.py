"""Central data/path configuration for the ldp-review repo.

Every script reads its data roots from here instead of hard-coding absolute
paths. Override any root by exporting the matching ``LDP_*`` environment
variable; otherwise sane repo-relative defaults under ``data/`` are used.

No market data is bundled in this repository (see ``DATA.md``). Point these
roots at your own copies of the public/commercial sources described there.

Example:
    export LDP_CRYPTO_1M=/path/to/binance_perp_1m
    export LDP_EQUITY_1M=/path/to/algoseek_etf_1min
    export LDP_FX_1M=/path/to/histdata_fx_1m

Usage in a script:
    import sys, os
    sys.path.insert(0, <repo root>)   # so ``import config`` resolves
    from config import CRYPTO_1M, EQUITY_1M, FX_1M, REPO_ROOT, LIB
"""
import os

# Repository root (directory containing this file) and the shared library dir.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("LDP_LIB", os.path.join(REPO_ROOT, "lib"))


def _root(env_name, default_subdir):
    """Resolve a data root from ``$env_name`` or a repo-relative default."""
    return os.environ.get(env_name, os.path.join(REPO_ROOT, "data", default_subdir))


# --- Market data roots --------------------------------------------------------
# Crypto Binance USD-M perpetual klines.
CRYPTO_1M = _root("LDP_CRYPTO_1M", "crypto_1m")   # *_1m.parquet
CRYPTO_1H = _root("LDP_CRYPTO_1H", "crypto_1h")   # binance_um/*_1h.parquet
CRYPTO_30M = _root("LDP_CRYPTO_30M", "crypto_30m")  # *_30m.parquet

# Point-in-time delisted Binance USD-M perpetual klines (survivorship-aware
# universe extension), *_1h.parquet. Each file carries only the bars that
# instrument actually traded.
CRYPTO_DELISTED_1H = _root("LDP_CRYPTO_DELISTED_1H", "crypto_delisted_1h")

# US equities (Algoseek ETF 1-minute trade bars), *.csv.gz.
EQUITY_1M = _root("LDP_EQUITY_1M", "equity_1m")

# Forex (HistData, resampled to a 1-minute base), *_fx1m.parquet.
FX_1M = _root("LDP_FX_1M", "fx_1m")

# Raw forex CSVs used by the portfolio study (per-pair files).
FX_RAW = _root("LDP_FX_RAW", "fx_raw")

# Per-strategy daily PnL corpus for the at-scale overfitting harness.
# Layout: <PNL_DAILY>/asset=<TICKER>_equity|<PAIR>_fx/part-*.parquet
PNL_DAILY = _root("LDP_PNL_DAILY", "pnl_daily")

# Banked verified ledgers consumed read-only by the ML studies. These artifacts
# are produced by a separate pipeline (see each study's README) and are not
# bundled here; point the roots at your own copies.
#   XS_LEDGER        : cross-sectional gradient-boosted-tree rotation ledger
#                      (trades.parquet + combos.parquet) for the cross-sectional study.
#   SINGLE_SERIES    : pooled single-series direction-forecasting corpus
#                      (<root>/<group>/<run>/cells.parquet) for the regime comparison.
#   XS_MULTI         : optional multi-market FX/equity cross-sectional summaries
#                      (<root>/market=<m>/summary.parquet).
#   METALABEL_LEDGER : banked per-trade ledger of the edged-primary meta-label run.
XS_LEDGER = _root("LDP_XS_LEDGER", "xs_ledger")
SINGLE_SERIES = _root("LDP_SINGLE_SERIES", "single_series")
XS_MULTI = _root("LDP_XS_MULTI", "xs_multi")
METALABEL_LEDGER = _root("LDP_METALABEL_LEDGER", "metalabel_ledger")

# Read-only structural-edge engine consumed by the meta-labeling scale-up study.
# Supplies the verified sim kernel, the cost model and the causal per-pair data
# loader for the proven structural edges. Produced by a separate pipeline (see the
# study README) and not bundled here; point the root at your own copy. Scripts
# import it lazily and skip gracefully if it is absent.
STRUCTURAL_ENGINE = _root("LDP_STRUCTURAL_ENGINE", "structural_engine")

# Local cache for downloaded/intermediate artifacts (run logs, etc.).
DATA_CACHE = _root("LDP_DATA_CACHE", "cache")
