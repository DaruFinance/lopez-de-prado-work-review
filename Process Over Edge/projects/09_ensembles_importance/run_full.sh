#!/usr/bin/env bash
# =============================================================================
# Project 09, Ensembles (bagging vs boosting + hyper-tuning) & Feature Importance
# (López de Prado: AFML Ch.6 ensembles, Ch.8 importance, Ch.9 hyper-tuning;
#  ML4AM Ch.6 clustered feature importance)
#
# FULL multi-market run (crypto + US equities + forex), causal-only, costed,
# PURGED-CV / CPCV, headline metric = Deflated Sharpe Ratio (lib/overfit.py).
#
# -----------------------------------------------------------------------------
# WHAT IT DOES
#   For 42 instruments (27 crypto dollar-bars + 7 ETF dollar-bars + 8 FX tick-bars):
#     (1) Bagging (RandomForest: low max_features, min_weight_fraction_leaf>0,
#         uniqueness sample-weights, max_samples = avg label uniqueness) vs
#         Boosting (HistGradientBoosting). Hyper-tuned over a purged grid by
#         NEG-LOG-LOSS (headline) and by ACCURACY (control). Reports OOS DSR and
#         the IS-OOS log-loss overfit gap for each.
#     (2) Feature importance: MDI vs MDA vs clustered-MDA (permutation, log-loss
#         scored, OOS); MDI substitution-bias signal; top-set stability across
#         CPCV paths (Jaccard).
#
# SIZING (set after a 1-core profile; sklearn fits = ~89% of runtime, so the
# levers are grid size, n_estimators=200, instrument count, NOT numba):
#   RF grid  = 3 max_features x 2 min_weight_fraction_leaf x 1 n_estimators = 6 cfgs
#   HGB grid = 2 max_iter x 2 learning_rate x 2 max_leaf_nodes              = 8 cfgs
#   Per instrument: ~877 sklearn fits, ~70 s single-core (profiled on BTCUSDT).
#
# EST RUNTIME : ~49 min single-core; with --jobs 16 -> ~8-15 min wall.
# EST PEAK RAM: < ~0.4 GB / worker (20k-bar series + 200-tree RF + ~350x10 feats)
#               => < ~7 GB at --jobs 16 (machine has 46 GB). Safe.
#
# NUMBA (verified bit-identical vs pure-Python references, --verify):
#   - sequential-bootstrap draw          (AFML 4.5.2)
#   - co-event count + avg label uniqueness (AFML 4.4)  max|Δ|: u=5.6e-16, idx=0
#   sklearn model fits are deliberately NOT numba'd.
#
# DETERMINISM: random_state=0 throughout; per-worker BLAS/OMP pinned to 1 thread
# so results do not depend on --jobs.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Pin every numerical backend to one thread PER PROCESS; parallelism is at the
# instrument level via --jobs (avoids BLAS oversubscription / nondeterminism).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

LOG="full_run.log"
echo "[$(date -u +%FT%TZ)] starting Project 09 full run" | tee "$LOG"

python3 scripts/run_ensembles_importance.py --jobs 16 2>&1 | tee -a "$LOG"

echo "[$(date -u +%FT%TZ)] done. tables/ figures/ written." | tee -a "$LOG"
