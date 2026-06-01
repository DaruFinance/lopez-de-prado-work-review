#!/usr/bin/env bash
# =============================================================================
# Project 08, Microstructural Features (Lopez de Prado, AFML Ch.19)
# FULL multi-market run. Heavy job; avoid launching alongside
# other heavy jobs (RandomForest uses all cores via n_jobs=-1).
#
# WHAT IT DOES
#   24 instruments across 3 markets (10 crypto perps, 7 US-equity ETFs, 7 forex
#   majors). Per instrument: build Ch.19 microstructure features on ~8 bars/day
#   information-driven bars (crypto/equity dollar bars, forex tick bars), then
#   evaluate next-bar DIRECTION + VOLATILITY AUC and a COSTED long/short Sharpe
#   under purged k-fold CV. Headline = Deflated Sharpe Ratio + PBO (lib/overfit).
#
# RESOURCES (measured)
#   Cores : RandomForest fit is ~96% of wall time -> n_jobs=-1 (all 32 cores).
#   RAM   : peak ~3-4 GB. The heaviest single object is one instrument's base
#           1m frame (equity SPY ~1.9M rows ~ 0.5 GB) loaded one-at-a-time, plus
#           the bar-level design matrix (<= ~42k bars x 9 features = trivial) and
#           one RandomForest (200 trees, depth 5). Instruments are processed
#           sequentially, so peak RAM ~= one base frame + one forest, NOT the sum.
#           Safe on this 46 GB box; no tiling needed.
#   Disk  : tables (~5 CSVs, < 1 MB) + 4 PNG figures. Negligible.
#   Time  : ~25-40 min wall on 32 cores. Bar build is Numba O(n) (seconds total);
#           cost is 24 instruments x ~12 RF fits (dir+vol over 6 purged folds).
#           Equity ETFs (~42k bars, full RTH history back to 2007) are the
#           heaviest; crypto/forex (~8.7k bars) are light. The CSCV PBO on the
#           stacked OOS-return matrix is cheap (10 splits, C(10,5)=252 combos).
#
# CAUSAL / COSTED / PURGED , every feature at bar t uses only bars <= t; the
#   long/short test charges a per-market round-trip cost on each position change;
#   all scoring is purged k-fold (embargo 1%). DSR is the overfitting gate.
#
# DATA AVAILABILITY PER MARKET (honest tiers)
#   crypto : taker_buy_volume -> TRUE signed flow. ALL estimators (Kyle,
#            Hasbrouck, VPIN, OFI true).
#   equity : volume + trade count, NO side -> Bulk-Volume Classification proxy
#            for Kyle/Hasbrouck/VPIN/OFI; price-only estimators exact.
#   forex  : tick count only, NO volume -> price-only estimators + tick-rule
#            Kyle (impact per signed tick) only; VPIN/Hasbrouck/OFI NOT computed.
#
# RERUN / IDEMPOTENT, overwrites tables/ and figures/ in place. Safe to re-run.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/scripts"

# Determinism for the numeric libs; RandomForest uses n_jobs=-1 (all cores).
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1          # BLAS single-thread; parallelism is via RF n_jobs

# Bit-identical Numba-kernel check first (fails fast if a kernel drifted).
python3 run_microstructure.py --verify

# Full multi-market run (n_jobs=-1 => all cores for RandomForest).
python3 run_microstructure.py --jobs -1
