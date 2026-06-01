#!/usr/bin/env bash
# Project 0 at scale — backtest overfitting / DSR / PBO / effective-N, ALL 3 MARKETS.
# Requires the equity/forex corpora to exist (the queue runs
# run_full_equity_forex_corpora.sh BEFORE this). Crypto corpora already on disk.
# Est: crypto ~3 min + equity ~3 min + fx ~3 min ≈ 10 min; RAM < 3 GB (tiled).
set -u
cd /home/daru/ldp_review/projects/00_backtest_overfitting
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for M in crypto equity fx; do
  echo "=== market: $M ==="
  LDP_MARKET=$M python3 scripts/run_corpus_overfit.py || echo "[warn] market $M failed (corpora missing?)"
done
