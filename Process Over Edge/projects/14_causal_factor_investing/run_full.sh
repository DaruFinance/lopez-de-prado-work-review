#!/usr/bin/env bash
# =============================================================================
# Project 14, Causal Factor Investing (López de Prado, "Causal Factor Investing"
# 2023; "Where Are the Factors?" association-vs-causation critique).
#
# FULL multi-market run. Reproduces, on REAL daily data across crypto + US equity
# ETFs + forex:
#   (a) Monte Carlo of the three elementary causal structures (fork/confounder,
#       chain/mediator, collider), showing a naive OLS "factor" is significant
#       under mis-conditioning and that correct backdoor adjustment fixes it;
#   (b) real cross-asset factors (momentum, realized vol, short-reversal), naive
#       associational panel regression vs a confounder-adjusted (backdoor) one,
#       reporting where the significance verdict FLIPS, with the headline metric
#       = Deflated Sharpe Ratio (lib/overfit.py) on the naive long-short sleeves;
#   (c) a hierarchy-of-evidence / falsification checklist applied to the survivors.
#
# This is a METHODOLOGICAL / simulation study (LdP's critique made testable), NOT
# a deployable edge, see README. All evidence is observational; no costless or
# synthetic price data; factors are causal (lagged, no look-ahead).
#
# ---- Resource envelope (measured on this box) -------------------------------
#   RUNTIME : ~80 s first run (one-time ETF 1-min -> daily cache build ~52 s),
#             ~25 s on subsequent runs (panel + ETF daily series are cached to
#             the repo-local data_cache/). MC (20k sims x 2k obs, NumPy
#             per-sim path) ~12 s; figures ~3 s.
#   PEAK RAM: < 2 GB. Daily panels are tiny (5k days x ~27 cols). The only
#             sizeable transient is ONE ETF-year 1-min csv at a time (~50 MB),
#             freed immediately. The MC per-sim path never materializes the full
#             n_sims x n_obs block (< 200 MB working set).
#   CORES   : single-process. NumPy/BLAS may use threads for reductions; pin with
#             OMP_NUM_THREADS if you want a strict 1-core run (smoke does this).
#
# ---- Profiling / Numba finding (honest) -------------------------------------
#   The MC OLS reductions over n_obs=2000 are large vectorized sums that
#   NumPy/BLAS handles better than scalar @njit loops: Numba/NumPy ratio ~1.8x
#   (NumPy WINS). The production engine is therefore engine='numpy'; the Numba
#   kernels are retained as a VERIFIED BIT-IDENTICAL cross-check (max|Δ| ~1e-13,
#   float-reorder only) and would only win for many tiny regressions. A fully
#   batched-vectorized MC would be fastest but its n_sims x n_obs arrays blow RAM
#   (~2.4 GB/array) at full scale, so the RAM-safe per-sim path is used.
#   Reproduce the profile with:  python3 scripts/run_causal.py --profile
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

python3 scripts/run_causal.py --n-sims 20000 --n-obs 2000

# Outputs:
#   tables/mc_structures.csv                  per-structure MC bias + rejection rates (Type-I / power)
#   tables/mc_sweep.csv                       dose-response: decision error vs bias strength
#   tables/real_factor_naive_vs_backdoor.csv  per market x factor: naive vs TWO backdoors, flip + kind
#   tables/real_factor_dsr.csv                DSR of the best naive long-short factor sleeve
#   tables/hierarchy_checklist.csv            falsification scorecard (levels 1-7)
#   tables/summary.json                       MC summary + cross-engine verify + headline counts
#   figures/fig1_mc_structures.png            MC coefficient distributions (3 structures)
#   figures/fig2_real_factor_flip.png         naive vs backdoor t-stat flip map
#   figures/fig3_mc_dose_response.png         decision error vs structural-bias strength
