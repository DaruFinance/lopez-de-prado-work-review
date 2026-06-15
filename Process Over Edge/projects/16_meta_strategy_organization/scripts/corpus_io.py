"""
corpus_io.py: shared data layer for the capstone experiments.

Builds, per asset, a DENSE (days x strategies) matrix of *net* daily PnL from the
real per-strategy daily corpus. Daily ``pnl_sum`` was verified to equal the
trade-level ``pnl_net`` exactly (SPY strategy 0: -161.2232719... both sides), so
``pnl_sum`` is realized, after-cost PnL. We use it directly as realized
performance. No look-ahead: each cell is the PnL booked on that calendar day.

The hot loop is the scatter-add that lays sparse (day, strategy, pnl) triples
into the dense matrix. It is implemented twice -- a NumPy ``np.add.at`` reference
and a Numba ``@njit`` kernel -- and verified BIT-IDENTICAL (max|delta| == 0) in
``_verify_scatter``. The kernel drives the large crypto assets.

RAM discipline: one asset at a time; the largest crypto matrix is ~50k strats x
~2.9k days x 8 bytes ~= 1.2 GB, plus a transient ~1-2 GB for the raw triples,
well under the box budget. Correlations are tiled by the callers, never the full
50k x 50k matrix.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from numba import njit

# Resolve the repo root (nearest ancestor holding config.py) so this module
# reads its data root from the central, env-var parameterized config rather than
# a hard-coded path. Mirrors the pattern used by project 15's run_uniqueness.py.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import config as cfg  # noqa: E402

# Per-strategy daily PnL corpus. Override with the LDP_PNL_DAILY env var; the
# at-scale corpora are produced by a separate data pipeline (see the study
# README) and are not bundled in this repository.
PNL_BASE = cfg.PNL_DAILY

# Coarse market tags from the asset directory names.
EQUITY = {"SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLV", "UVXY", "VXX"}
FX_TOKENS = ("fx", "forex")


def list_assets() -> list[str]:
    return sorted(d[len("asset="):] for d in os.listdir(PNL_BASE)
                  if d.startswith("asset="))


def market_of(asset: str) -> str:
    base = asset.split("_")[0]
    if base in EQUITY:
        return "equity"
    if any(tok in asset for tok in FX_TOKENS):
        return "fx"
    return "crypto"


@njit(cache=True, fastmath=False)
def _scatter_kernel(dcodes, scodes, pnl, nd, ns):
    """Numba scatter-add: M[d, s] += pnl, summing duplicate (d,s). Same
    arithmetic and accumulation order as np.add.at on the same arrays."""
    M = np.zeros((nd, ns), dtype=np.float64)
    for i in range(pnl.shape[0]):
        M[dcodes[i], scodes[i]] += pnl[i]
    return M


def _scatter_ref(dcodes, scodes, pnl, nd, ns):
    M = np.zeros((nd, ns), dtype=np.float64)
    np.add.at(M, (dcodes, scodes), pnl)
    return M


def _verify_scatter(seed: int = 7) -> float:
    """Bit-identical check on random triples; returns max|delta| (want 0.0)."""
    rng = np.random.default_rng(seed)
    nd, ns, n = 40, 25, 5000
    d = rng.integers(0, nd, n).astype(np.int64)
    s = rng.integers(0, ns, n).astype(np.int64)
    p = rng.standard_normal(n)
    a = _scatter_ref(d, s, p, nd, ns)
    b = _scatter_kernel(d, s, p, nd, ns)
    return float(np.max(np.abs(a - b)))


def build_matrix(asset: str, use_numba: bool = True, verbose: bool = False):
    """Return (M, dates, strat_names) for one asset.

    M: (n_days, n_strats) float64 dense net-daily-PnL on the asset's own trading
    calendar (only days that actually appear). dates: sorted unique dates.
    strat_names: sorted unique strategy names (column order).
    """
    t0 = time.time()
    d = ds.dataset(os.path.join(PNL_BASE, f"asset={asset}"), format="parquet")
    tbl = d.to_table(columns=["strategy_name", "date", "pnl_sum"])
    sn = tbl.column("strategy_name").to_numpy(zero_copy_only=False)
    dt = tbl.column("date").to_numpy(zero_copy_only=False)
    pnl = np.ascontiguousarray(tbl.column("pnl_sum").to_numpy())
    del tbl
    scodes, suniq = pd.factorize(sn, sort=True)
    dcodes, duniq = pd.factorize(dt, sort=True)
    scodes = scodes.astype(np.int64); dcodes = dcodes.astype(np.int64)
    nd, ns = len(duniq), len(suniq)
    if use_numba:
        M = _scatter_kernel(dcodes, scodes, pnl, nd, ns)
    else:
        M = _scatter_ref(dcodes, scodes, pnl, nd, ns)
    if verbose:
        print(f"    [{asset}] M={M.shape} built in {time.time()-t0:.1f}s "
              f"({M.nbytes/1e6:.0f} MB)", flush=True)
    dates = pd.to_datetime(pd.Index(duniq)).values.astype("datetime64[D]")
    return M, dates, np.asarray(suniq)


if __name__ == "__main__":
    print("scatter bit-identical max|delta| =", _verify_scatter())
    M, dates, names = build_matrix("SPY_equity", verbose=True)
    print("SPY:", M.shape, dates[0], dates[-1], names[:2])
