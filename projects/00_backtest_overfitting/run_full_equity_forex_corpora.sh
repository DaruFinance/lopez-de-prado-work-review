#!/usr/bin/env bash
# =============================================================================
# run_full_equity_forex_corpora.sh
# HEAVY generation of the US-EQUITY and FOREX TA-strategy corpora (per-strategy
# daily PnL) for the at-scale backtest-overfitting harness. QUEUED OVERNIGHT.
#
# Engine : lib/ta_grid.py + lib/realism.py  (Numba njit indicators + sim kernel
#          with REALISTIC frictions; verified bit-identical vs NumPy reference,
#          max|Δpnl|=0.0, max|Δn_trades|=0 on both equity & forex friction models)
# Driver : scripts/gen_equity_forex_corpora.py  (--wide, idempotent _DONE)
#
# REALISTIC COSTS (lib/realism.py — calibrated, causal, NO CLAMPING)
#   FOREX  : per-fill = time-of-day HALF-SPREAD in pips x pip_size, charged each
#            entry/exit fill. Base half-spread/pair (overlap-tight): EURUSD 0.10,
#            GBPUSD 0.25, USDJPY 0.15, USDCHF/USDCAD/EURGBP 0.30, AUDUSD 0.20,
#            NZDUSD 0.35; UTC time-of-day mult 1.0 (12-16 overlap) .. 3.0 (21-23
#            rollover) .. 2.2 (thin Asian). Overnight SWAP ~0.30 pip/night (x3 Wed)
#            on held positions at the 21:00 UTC rollover bar. Positions force-flat
#            at the last open bar before the Fri 22:00 UTC weekend close; the real
#            Fri->Sun price gap is KEPT (not clamped). JPY pip = 0.01.
#   EQUITY : per-fill = time-of-day HALF-SPREAD (bp) + min-ticket commission
#            ($0.0035/sh, $0.35 floor) as a fraction of fill price. Base half-spread:
#            SPY/QQQ 0.5, IWM 1.0, XLK/XLF 1.5, XLE/XLV 2.0, VXX 8.0, UVXY 12.0 bp;
#            intraday mult 2.5x open (09:30-10:00), 1.8x close (15:45-16:00), 1.0x
#            mid. Short BORROW accrued per ET calendar day on shorts (sector/broad
#            ~0.5-0.7%/yr, UVXY/VXX 5%/yr); cash longs no financing. Force-flat at
#            RTH close (intraday). Modeled spreads (Algoseek etf_1min lacks NBBO).
#
# CORPUS SIZE (--wide grid = 6,296 structural strategies / instrument)
#   Equities : 9 instruments  (SPY QQQ IWM XLK XLF XLE XLV UVXY VXX)  -> ~56,664
#   Forex    : 8 instruments  (EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURGBP) -> ~50,368
#   TOTAL    : 17 instruments x 6,296  ~= 107,000 strategies
#   (well past the harness's ~50k/market diversity target; >50k = diminishing returns)
#
# DATA / COVERAGE
#   Equities : Algoseek ETF 1-min, RTH only, within-session DOLLAR bars
#              (~78 bars/RTH day); full history 2007-2026 (VXX 2009+, UVXY 2011+)
#   Forex    : HistData 1-min, count-clock TICK bars (~96/day); 2022-2024 (3 yr;
#              spot FX has no volume -> volume=quote_volume=count proxy)
#
# ESTIMATES (measured on 1 core; SPY full-history wide = 24.4M rows / 54s / 77MB)
#   Runtime  : ~10 min single-core sequential; this script parallelizes across
#              instruments (NPROC procs, each pinned to 1 BLAS/Numba thread) ->
#              ~3-4 min wall on a multi-core box. Numba JIT warmup ~2s once/proc.
#   Peak RAM : one instrument in memory at a time; ~1.5 GB peak for SPY (longest
#              history) during output-buffer assembly. With NPROC=4 budget ~6 GB.
#              Lower NPROC if RAM-constrained. Bars themselves are ~25 MB.
#   Disk     : ~0.8 GB total (zstd parquet; daily-aggregated PnL is small).
#              D: has ~29 GB free -> ample headroom.
#
# OUTPUT (exact harness layout; auto-discovered by run_corpus_overfit.py)
#   /mnt/d/strategies_parquet/pnl_daily/asset=<TICKER>_equity/part-00000.parquet
#   /mnt/d/strategies_parquet/pnl_daily/asset=<PAIR>_fx/part-00000.parquet
#   columns: asset, family, strategy_name, date(date32), pnl_sum(f64), n_trades(i32)
#   Costed: REALISTIC per-bar time-of-day half-spread + commission per fill, plus
#   FX swap (x3 Wed) + weekend force-flat and equity short-borrow (see header).
#   Causal (signal at t -> fill at t+1 open). Intrabar OHLC SL/TP.
#   Idempotent: per-asset _DONE sentinel; re-running skips completed assets.
#
# USAGE
#   bash run_full_equity_forex_corpora.sh            # all 17, NPROC=4
#   NPROC=8 bash run_full_equity_forex_corpora.sh    # more parallelism
#   then: python3 scripts/run_corpus_overfit.py  (after adapting it to add the
#         equity/fx markets, or run the per-market variant)
# =============================================================================
set -euo pipefail

REPO=/home/daru/ldp_review
GEN="$REPO/scripts/gen_equity_forex_corpora.py"
NPROC="${NPROC:-4}"

EQ=(SPY QQQ IWM XLK XLF XLE XLV UVXY VXX)
FX=(EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURGBP)

echo "=== FULL equity+forex corpus generation (wide grid ~6,296/instrument) ==="
echo "NPROC=$NPROC  instruments: ${#EQ[@]} equity + ${#FX[@]} forex"
date

# one instrument per worker; pin numerical libs to 1 thread so NPROC workers
# don't oversubscribe cores. Each invocation is idempotent (skips _DONE).
run_eq() {
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    python3 "$GEN" --wide --market equity --tickers "$1"
}
run_fx() {
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    python3 "$GEN" --wide --market forex --tickers "$1"
}
export GEN
export -f run_eq run_fx

printf '%s\n' "${EQ[@]}" | xargs -P "$NPROC" -I{} bash -c 'run_eq "$@"' _ {}
printf '%s\n' "${FX[@]}" | xargs -P "$NPROC" -I{} bash -c 'run_fx "$@"' _ {}

echo "=== DONE ==="
date
echo "Output: /mnt/d/strategies_parquet/pnl_daily/asset={SPY..VXX}_equity, {EURUSD..EURGBP}_fx"
du -sh /mnt/d/strategies_parquet/pnl_daily/asset=*_equity /mnt/d/strategies_parquet/pnl_daily/asset=*_fx 2>/dev/null | tail -20 || true
