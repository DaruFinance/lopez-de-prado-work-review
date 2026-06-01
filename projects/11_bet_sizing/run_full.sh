#!/usr/bin/env bash
# =============================================================================
# Project 11 — Bet Sizing (López de Prado, AFML Ch.10)  — FULL RUN
# =============================================================================
# WHAT IT DOES
#   Runs the full bet-sizing study across all 42 real-data instruments
#   (27 crypto perps + 7 US-equity ETFs + 8 FX pairs), 32 IS-tunable trials
#   each, comparing fixed-size vs probability-sized vs discretized-probability-
#   sized books. Headline metric = Deflated Sharpe Ratio (net of costs), with
#   PBO + effective-N as deflation diagnostics. Writes tables/ + figures/.
#
# ESTIMATED RUNTIME (single process, n_jobs=1)
#   Profiled 1 crypto instrument (32 trials) = ~15.8 s wall on 1 core.
#   Sklearn bagged-tree fitting dominates (~75%); the Numba active-bet kernel
#   and sizing math are negligible (<1%). Equities are heavier per instrument
#   (~more events; ~13 s in smoke with only 2 trials -> ~60-90 s at 32 trials);
#   FX is lighter. Mixed-universe estimate:
#       crypto  27 x ~16 s  ~= 430 s
#       equity   7 x ~75 s  ~= 525 s
#       forex    8 x ~12 s  ~=  95 s
#   TOTAL  ~= 1050 s  ->  budget ~15-25 min single-process, 1 core.
#   (The whole job is embarrassingly parallel over instruments; this script is
#    deliberately SINGLE-PROCESS / n_jobs=1 for a clean overnight run. To go
#    faster, fan out instruments across cores — the per-instrument code is
#    already 1-core/n_jobs=1 internally and thread-safe.)
#
# ESTIMATED PEAK RAM
#   One instrument at a time. The largest base series (BTCUSDT 1m ~1.58M rows
#   x ~10 float/int cols) loads at ~150-250 MB, aggregates down to 20k bars
#   (<5 MB). Bagged trees on a few-thousand-event matrix add ~100-300 MB
#   transiently. The per-bar difference arrays are O(n_bars)=20k floats (<1 MB).
#   PEAK ~= 0.8-1.5 GB resident. Safe on the ~46 GB box; no N x N materialisation.
#
# OUTPUTS (idempotent; overwritten each run)
#   tables/raw_results.parquet     per-instrument result rows (machine-readable)
#   tables/per_instrument.csv      per-instrument: PF/SR/DSR/turnover x 3 schemes
#   tables/by_market_summary.csv   by-market medians + DSR>0.95 counts + PBO
#   tables/results.md              markdown summary (by-market + per-instrument head)
#   figures/fig1_dsr_by_scheme.png      DSR by market, fixed/prob/disc
#   figures/fig2_turnover_by_scheme.png turnover (overtrading) by scheme
#   figures/fig3_equity_by_scheme.png   representative book equity curves
#   figures/fig4_betsize_dist.png       prob vs discretized bet-size histograms
#
# DATA (real, read-only)
#   crypto : /mnt/c/Users/USUARIO/Desktop/ldp_cache_1m/*_1m.parquet
#   equity : /mnt/d/algoseek_data/etf_1min/{SPY,QQQ,IWM,XLK,XLF,XLE,XLV}.csv.gz
#   forex  : /mnt/c/Users/USUARIO/Desktop/ldp_cache_fx/*_fx1m.parquet
#
# VERIFICATION (run anytime; not part of the full job)
#   python3 scripts/run_bet_sizing.py --verify-kernel
#     -> avg_active Numba kernel vs pure-Python reference: max|delta| = 0.0e+00
#        (BIT-IDENTICAL across 200 random cases). Numba kernel ~102x faster than
#        the reference at full scale (0.033 ms vs 3.37 ms / call).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Pin BLAS/OpenMP to 1 thread so the run is single-core and reproducible.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 scripts/run_bet_sizing.py 2>&1 | tee full_run.log
