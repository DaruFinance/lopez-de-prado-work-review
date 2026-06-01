#!/usr/bin/env bash
# Project 0 at scale, backtest overfitting / DSR / PBO / effective-N, ALL 3 MARKETS.
# Requires the per-strategy daily PnL corpora for each market to be present under
# the LDP_PNL_DAILY root (produced by a separate data pipeline; see DATA.md).
# Est: crypto ~3 min + equity ~3 min + fx ~3 min, about 10 min; RAM < 3 GB (tiled).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for M in crypto equity fx; do
  echo "=== market: $M ==="
  LDP_MARKET=$M python3 scripts/run_corpus_overfit.py || echo "[warn] market $M failed (corpora missing?)"
done
