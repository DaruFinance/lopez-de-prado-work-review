# Project 05 — Trend-Scanning Labels (López de Prado, ML4AM Ch.5)

**Status:** COMPLETE. Full 42-instrument run + deepening (trend-scan vs
fixed-horizon vs triple-barrier targets, plus look-forward-band robustness) done.
See `writeup/README.md` for the 8-section writeup and the honest DSR verdict.

## The method (reproduced)
Trend-scanning is LdP's label for *how trending* a point is. For each observation
`t`, regress log-price on a time index over every forward look-ahead window
`L ∈ [Lmin, Lmax]`:

    y_j = a + b·j ,   j = 0..L-1 ,   y_j = logprice[t+j]

compute the slope `b` and its **t-value** `t_b = b / se(b)`. The label is the
**sign of the t-value at the horizon `L*` that maximises `|t_b|`** (the most
statistically significant local trend); `|t_b(L*)|` is the **confidence / meta
weight**, and `L*` is the data-chosen horizon. We also build a **fixed-horizon**
control labeller (OLS t-value at a single `L`) — the labeller LdP contrasts
trend-scanning against.

These are **labels** (forward-looking supervised targets), not features.

## The experiment (leakage-free, costed, multi-market)
You cannot trade a label directly (that leaks the future into P&L). Instead:

1. Build trend-scan and fixed-horizon labels at each event (events = bars whose
   `|t_b|` clears an IS-tuned confidence quantile — LdP samples on significant
   trends).
2. A bagged-tree **secondary model predicts the label sign** from **causal
   features only**, trained & scored under **purged k-fold CV** with embargo.
3. The strategy trades the **out-of-fold predicted side**, entering at the event
   close and exiting after a **causal forward hold**, net of per-turnover costs
   (crypto 7 bp / equities 2 bp / forex 1 bp per side).
4. Score with the **Deflated Sharpe Ratio** (9-trial IS-tunable grid = the trial
   count fed to the False-Strategy benchmark), plus **PBO (CSCV)** and
   **effective-N**. Compare a model trained on trend-scan labels vs one trained on
   fixed-horizon labels: *does trend-scanning make a better target?*

**Markets:** Crypto (27 perps, dollar bars) + US Equities (7 ETFs RTH, dollar
bars) + Forex (8 majors, tick bars) = 42 instruments, ≥10 per market.

## Hot loop & performance
The per-observation multi-horizon OLS-t scan is the hot loop, written as a Numba
`@njit` kernel with **incremental** running sums (O(1) per extra horizon bar). An
independent NumPy reference (`trend_scan_reference`, full refit per horizon)
verifies it:

- **label & `L*` bit-identical**; realised window return bit-identical (`max|Δ|=0`).
- `t_val` differs by `~1e-6` (float **summation-order** only — the kernel
  accumulates sums incrementally, the reference refits from scratch). The signs,
  argmax horizon, and realised returns that drive selection and P&L are exact, so
  the experiment is unaffected.
- **Numba speedup ≈ 1176×** vs the pure-NumPy reference (n=3000, band (20,120):
  4502 ms → 3.8 ms). Profiling shows the full-run cost is dominated by the
  sklearn meta-model CV fits, not the kernel.

## Files
- `scripts/trendscan.py` — Numba kernels (`trend_scan`, `fixed_horizon`,
  `causal_hold_ret`), NumPy reference, loaders, causal features.
- `scripts/run_trend_scanning.py` — idempotent driver; `--smoke` / `--profile` /
  `--verify` flags.
- `run_full.sh` — exact full-scale command + header (runtime, RAM, outputs).
- `tables/`, `figures/` — outputs (overwritten per run).

## Run
```bash
python3 scripts/run_trend_scanning.py --verify    # kernel vs reference
python3 scripts/run_trend_scanning.py --smoke      # 1 inst/market, tiny (1 core)
bash run_full.sh                                   # full 42-instrument run (~20-25 min)
```

## Data risks / caveats
- **Smoke windows show degenerate edges** (PF in the thousands) because a short
  sample over-selects a single persistent trend; this vanishes at full scale where
  thousands of regime-switching events balance the side target. Judge only the
  full run.
- Equities use **RTH dollar bars with session-aware returns** (overnight gaps
  excluded); FX has no volume so it uses the **tick clock** (count-threshold bars).
- Costs are per-turnover only; no funding leg modelled here (statistical-label
  study with a costed side-prediction overlay, not a production carry strategy).
- The forward look-ahead lives **only in the label**; features and the acted-on
  OOF side use no window-`t`-or-later information.
