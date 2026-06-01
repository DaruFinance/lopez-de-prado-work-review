# Bet Sizing from Predicted Probabilities (López de Prado, AFML Ch.10)

Reproduces López de Prado's **bet-sizing** recipe and tests, across **Crypto +
US Equities + Forex** (42 instruments, ≥10 per market), whether sizing a bet by
the **meta-model probability**, and **averaging concurrent bets** /
**discretizing** the size to curb overtrading, beats a fixed-size book. The
**headline metric is the Deflated Sharpe Ratio (DSR)**, net of realistic costs,
with PBO and effective-N as deflation diagnostics.

Everything runs on **real 1-minute data** (no synthetic series), every change in
book position is **costed** (per-turnover bp by market), all signals are
**causal**, and the meta probabilities are **out-of-fold from purged k-fold CV**
(reusing `projects/03_meta_labeling/scripts/tbm.py`) so overlapping
triple-barrier labels cannot leak into the sizing input.

## The Ch.10 recipe implemented

1. **Probability → size.** From the meta probability `p = P(primary's bet is
   profitable)`, against `p0 = 0.5`:
   `z = (p − 0.5) / sqrt(p(1−p))`, `m = 2·Φ(z) − 1 ∈ (−1, 1)` (LdP 10.1-10.2).
   The signed bet is `s · m` (the primary fixes the side `s`; the meta only
   sizes/vetoes).
2. **Average concurrent / overlapping active bets** (the HOT LOOP). A bet is
   active over `[entry, entry+hold)`. At each bar the book's net position is the
   **mean** signed size of all bets active there (LdP `avgActiveSignals`), this
   nets opposing bets and damps turnover. Implemented as a Numba difference-array
   kernel `_avg_active_kernel` (O(n_ev + n_bars)), verified **bit-identical**
   (`max|Δ| = 0`) against an independent pure-Python reference and ~**102×**
   faster at full scale.
3. **Discretize** the size to a grid step `d` (`discretizeSignal`) to avoid
   over-trading on tiny size changes.
4. **Costed book P&L.** `pnl_t = pos_t · r_{t→t+1} − cost · |pos_t − pos_{t−1}|`
   (cost on the change in book position = realised turnover).

Three schemes share the same events, OOF probabilities, costs, and the same
active-bet averaging + costed-book engine; only the per-event target size
differs: **fixed** (unit), **prob** (continuous `m`), **prob_disc**
(discretized `m`). The grid of `(fast/slow, pt/sl, max_hold, meta_threshold,
disc_step)` are the IS-tunable knobs = the **32 DSR trials**.

## Run

```bash
./run_full.sh                                  # full 42-instrument run (~15-25 min, 1 core)
python3 scripts/run_bet_sizing.py --smoke       # one instrument/market, tiny subset
python3 scripts/run_bet_sizing.py --profile     # cProfile a single instrument
python3 scripts/run_bet_sizing.py --verify-kernel  # numba vs python bit-identical check
```

See `run_full.sh` header for runtime/RAM estimates and the full output list.

## Costs (per turnover, bp of notional)
crypto 7.0 · equities 2.0 · forex 1.0, charged on the change in book position.

## Status
COMPLETE. Full 42-instrument run finished (~673 s, ~0.85 GB peak). Deepening
(`scripts/deepen.py`) + complete writeup (`writeup/bet_sizing.md`) done.

**Headline.** Turnover collapses **−80 to −87%** across all 42 instruments and all
three markets (median 254 → 49 prob / 46 disc), but the **Deflated Sharpe does not
move** (median ΔDSR ≈ 0, no sign-test significance; **0/42 instruments cross
DSR > 0.95** under any scheme; PBO median 0.49). Of the two sizing variants,
**discretized ≥ continuous** (disc_dsr ≥ prob_dsr in 33/42; disc PF beats fixed in
26/42 vs 15/42 for continuous). Verdict: **bet sizing is a precision / cost-control
layer, not an alpha source**, it expresses a fixed edge more cheaply, it does not
create one.

**Realism fix.** `lib.realism.equity_commission_rate_bp` models a literal
1-share position, so the \$0.35 min-ticket floor became 8.75-70 bp/fill and crushed
every equity book (5/7 ETFs NaN in the initial run). Fixed project-locally
(`equity_per_side_cost_realistic`, \$10k notional, lib untouched): equity cost now
0.85-3 bp/side and all 7 ETFs are informative. No clamping.
