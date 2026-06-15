"""T_XS — Cross-Sectional engine: shared constants, costs, device, WFO geometry.

THE THESIS (Daniel, 2026-05-30): the *switch of pairs IS the strategy*. Each bar we
rank the whole alive universe by a causal per-pair score and rotate capital into the
best (long top quantile / short bottom), neutralized. This is the literature's strongest
ML-in-finance pattern (cross-sectional return prediction) — NOT a single-series corpus
replicated across pairs.

Costs reused from the verified novel engine (per-fill). The portfolio sim (xs_sim) is
NEW (the novel kernel is single-asset) but uses the SAME cost constants. We NEVER modify
novel/kernel.py or novel/common.py (shared with closed T_NOVEL/T6).

Causality is load-bearing: every feature is shift(1); cross-sectional normalization uses
only bar-t's cross-section; labels are purged (lbl_end < boundary) in the WFO fit.
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

import config as _cfg

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS

# Structural-edge engine: supplies the verified cost constants and metrics
# (its common.py) reused unmodified by this audit. Imported lazily below.
_NOVEL = _cfg.STRUCTURAL_ENGINE
if _NOVEL not in sys.path:
    sys.path.insert(0, _NOVEL)

import numpy as np  # noqa: E402
import common as _c  # noqa: E402  (structural engine common.py — costs, metrics)

# ---- reused cost constants (per fill) ----
TAKER_FILL = _c.TAKER_FILL   # 0.0007 = 0.05% fee + 0.02% slip (rotations are taker)
MAKER_FILL = _c.MAKER_FILL   # 0.0004
# funding: per-bar perp funding is applied directly from the aligned funding-rate panel
# (rate is per-8h; we apply rate * (bar_hours/8) per 1h bar on net exposure).
BAR_HOURS = 1.0
FUNDING_PER_BAR_SCALE = BAR_HOURS / 8.0

# ---- data roots (resolved from config / environment) ----
# Live Binance USD-M perpetual hourly klines and the survivorship-aware delisted /
# spot underlyings panels come straight from config. The funding, open-interest,
# order-flow and inverse (CM) sibling feeds are not part of the bundled config map;
# each resolves from its own LDP_* override, defaulting under the repo data/ dir.
def _root(env_name, default_subdir):
    return os.environ.get(env_name, os.path.join(REPO_ROOT, "data", default_subdir))


PERP_DIR = _cfg.CRYPTO_PERP_1H
PERP_CM_DIR = _root("LDP_CRYPTO_PERP_CM_1H", "crypto_perp_cm_1h")
FUND_DIR = _root("LDP_CRYPTO_FUNDING", "crypto_funding")
SPOT_DIR = _cfg.CRYPTO_SPOT_1H
OF_DIR = _root("LDP_CRYPTO_ORDERFLOW", "crypto_orderflow")
OI_DIR = _root("LDP_CRYPTO_OI", "crypto_oi")
DELISTED_DIR = _cfg.CRYPTO_DELISTED_1H
DELISTED_FUND_DIR = _root("LDP_CRYPTO_DELISTED_FUNDING", "crypto_delisted_funding")
LISTING_CSV = _cfg.CRYPTO_LISTING_DATES

CACHE_DIR = os.environ.get("XS_CACHE", os.path.join(HERE, "_cache"))
OUT_ROOT = os.environ.get("XS_OUT_ROOT", HERE)
LEADER = "BTCUSDT"

# ---- universe filter (keep RAM/compute sane; pair-switching still spans the set) ----
# min bars of history to be eligible; min median dollar-vol percentile to be eligible.
MIN_HISTORY_BARS = int(os.environ.get("XS_MIN_HIST", 2000))   # ~83 days at 1h
MIN_DOLLAR_VOL = float(os.environ.get("XS_MIN_DVOL", 0.0))    # 0 = no liquidity filter
UNIVERSE_LIMIT = int(os.environ.get("XS_UNIV", 0))            # 0 = all eligible; else top-N by dvol
SMOKE_PAIRS = os.environ.get("XS_SMOKE_PAIRS", "")            # csv to force a tiny universe

# ---- WFO geometry (1h bars; env-overridable). Cross-sectional → larger windows ok. ----
IS = int(os.environ.get("XS_IS", 12000))      # ~500 days at 1h
WALK = int(os.environ.get("XS_WALK", 6000))   # ~250 days OOS
IS_SEL = int(os.environ.get("XS_IS_SEL", 3000))

NS_MODEL = int(os.environ.get("XS_NS_MODEL", 1))
NS_KNOB = int(os.environ.get("XS_NS_KNOB", 48))
MIN_OOS_REB = int(os.environ.get("XS_MIN_OOS_REB", 5))  # min OOS rebalances to report a combo

# ---- device ----
def gpu_enabled():
    if os.environ.get("XS_GPU", "1") == "0":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def torch_device():
    import torch
    return torch.device("cuda" if gpu_enabled() else "cpu")


def combo_seed(name, combo_tuple):
    return abs(hash((name, combo_tuple))) % (2**31 - 1)


def wfo_windows(T, IS_=None, WALK_=None):
    IS_ = IS if IS_ is None else IS_
    WALK_ = WALK if WALK_ is None else WALK_
    ws = 0; wid = 0
    while ws + IS_ + WALK_ <= T:
        yield wid, ws, ws + IS_, ws + IS_ + WALK_
        ws += WALK_
        wid += 1


def sharpe(pnl):
    pnl = np.asarray(pnl, dtype=np.float64)
    if pnl.size < 2:
        return 0.0
    sd = pnl.std()
    return float(pnl.mean() / sd * np.sqrt(252 * 24 / BAR_HOURS)) if sd > 1e-12 else 0.0


def profit_factor(pnl):
    pnl = np.asarray(pnl, dtype=np.float64)
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    return float(pos / neg) if neg > 1e-12 else (np.inf if pos > 0 else 0.0)
