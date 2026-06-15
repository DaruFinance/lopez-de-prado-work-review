#!/usr/bin/env python3
"""SURVIVOR SKELETON — a reusable, costed, leak-free chassis from the two review survivors.

  - HRP (S8 survivor): allocates capital across assets to MINIMIZE out-of-sample portfolio
    variance — its documented strength (it beats the unstable raw Markowitz optimizer OOS).
    Imported verbatim from s08_v2.
  - Causal trailing realized covariance (S5a survivor's principle: next-period (co)variance
    is best forecast by recent realized (co)variance): used to VOL-TARGET the book to a
    chosen risk level.

Chassis: signal -> direction; HRP -> risk-balanced weights; trailing-cov -> vol-target;
costs on turnover. Decision at end of bar t (uses data <= t) earns bar t+1 (leak-free).

HONEST NOTE: HRP is a RISK-ALLOCATION / min-variance tool, NOT an alpha-sizer. For a
uniform cross-sectional alpha, equal-NOTIONAL weighting can capture more raw edge; HRP's
value is a lower, more STABLE out-of-sample portfolio variance. So we validate HRP on its
real job (OOS variance vs Markowitz and equal-weight), and validate the chassis plumbing
(leak-free, edge capture, vol-target) separately.

Validations (synthetic clustered, heterogeneous-vol universe; light, no market panel):
  A leak-free: poison the FUTURE -> pre-poison weights bit-identical.
  B captures a real edge: planted causal edge -> net-positive; noise -> not.
  C vol-target: realized vol near target (trailing-cov forecast).
  D HRP on its real job: OOS portfolio vol(HRP) <= vol(raw Markowitz) [S8 stability] AND
    HRP risk-competitive with equal-weight.
"""
import os
os.sched_setaffinity(0, range(16, 32))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

S08 = os.path.join(os.path.dirname(HERE), "S08_shrink_optimize")
S12 = os.path.join(os.path.dirname(HERE), "S12_clean_audit")
sys.path.insert(0, S08); sys.path.insert(0, S12)
from s08_v2 import alloc_hrp, alloc_equal_weight, alloc_raw_markowitz  # S8 survivor + baselines
from s12_battery import sharpe

ANN = np.sqrt(252 * 24)


def make_panel(beta, seed, T=20000, N=80, rho=0.92, base_vol=0.01, K=8):
    """Clustered, heterogeneous-vol returns (market + K cluster factors + idio, lognormal
    per-asset vols) so HRP's hierarchical risk-balancing has real structure to exploit, plus
    a causal planted edge of strength beta (signal s[t] forecasts rets[t+1])."""
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, K, N)
    vols = base_vol * np.exp(rng.normal(0.0, 0.8, N))
    m = rng.standard_normal((T, 1)); cf = rng.standard_normal((T, K)); idio = rng.standard_normal((T, N))
    wm, wc = 0.35, 0.55
    wi = np.sqrt(max(1e-9, 1 - wm ** 2 - wc ** 2))
    eps = wm * m + wc * cf[:, cluster] + wi * idio
    shock = rng.standard_normal((T, N)); s = np.empty((T, N)); s[0] = shock[0]; a = np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        s[t] = rho * s[t - 1] + a * shock[t]
    fut = beta * s + np.sqrt(np.clip(1 - beta ** 2, 0, 1)) * eps
    rets = np.zeros((T, N)); rets[1:] = fut[:-1] * vols[None, :]
    return rets, s


def survivor_backtest(rets, signal, hrp_win=240, target_vol=0.0015, cost=0.0007,
                      rebal=24, allocator=alloc_hrp, lev_cap=5.0, vol_lookback=240):
    """Two passes: (1) unlevered HRP-weighted, signal-directed book; (2) FEEDBACK
    vol-target — scale leverage by the book's own TRAILING REALIZED vol (the S5a survivor's
    principle: trailing realized vol forecasts next vol). Leak-free: weights at t use only
    info <= t and earn bar t+1."""
    T, N = rets.shape
    Wr = np.zeros((T, N)); wprev = np.zeros(N)
    for t in range(hrp_win, T - 1):                           # pass 1: unlevered weights
        if (t % rebal) != 0:
            Wr[t] = wprev; continue
        win = np.nan_to_num(rets[t - hrp_win:t])              # causal (<= t-1)
        try:
            h = np.nan_to_num(np.asarray(allocator(win), float))
        except Exception:
            h = np.ones(N) / N
        d = np.sign(np.nan_to_num(signal[t - 1]))             # direction, causal
        Wr[t] = np.nan_to_num(h * d); wprev = Wr[t]
    raw_gross = np.zeros(T)
    raw_gross[:-1] = (Wr[:-1] * np.nan_to_num(rets[1:])).sum(1)
    # pass 2: feedback vol-target on the book's trailing realized vol (causal: shift(1))
    rv = pd.Series(raw_gross).rolling(vol_lookback, min_periods=vol_lookback // 2).std().shift(1).to_numpy()
    lev = np.clip(target_vol / np.where((rv > 1e-9) & np.isfinite(rv), rv, np.inf), 0.0, lev_cap)
    W = Wr * lev[:, None]
    gross = np.zeros(T)
    gross[:-1] = (W[:-1] * np.nan_to_num(rets[1:])).sum(1)    # earn the NEXT bar (leak-free)
    turn = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, N)))).sum(1)
    return gross - turn * cost, W


def oos_portfolio_vol(rets, allocator, win=120, step=120):
    """HRP's REAL job: rolling min-variance allocation. Estimate weights on the trailing
    `win` bars, measure the REALIZED portfolio vol on the next `step` (out-of-sample) bars.
    Returns annualized OOS vol. Lower = better risk control; Markowitz tends to blow up."""
    T, N = rets.shape
    segs = []
    for t in range(win, T - step, step):
        w = np.nan_to_num(np.asarray(allocator(np.nan_to_num(rets[t - win:t])), float))
        segs.append(np.nan_to_num(rets[t:t + step]) @ w)
    r = np.concatenate(segs) if segs else np.zeros(1)
    return float(np.std(r) * ANN)


def main():
    out = {}
    rets, sig = make_panel(beta=0.08, seed=1)
    pnl_hrp, W = survivor_backtest(rets, sig, allocator=alloc_hrp)
    rets0, sig0 = make_panel(beta=0.0, seed=2)
    pnl_noise, _ = survivor_backtest(rets0, sig0, allocator=alloc_hrp)
    tgt = 0.0015   # ~12% annualized (sqrt(252*24)*0.0015); a realistic risk target
    out["B_real_edge_net_sharpe"] = round(sharpe(pnl_hrp), 3)
    out["B_noise_net_sharpe"] = round(sharpe(pnl_noise), 3)
    out["C_target_vol_per_bar"] = tgt
    out["C_realized_vol_per_bar"] = round(float(np.std(pnl_hrp[pnl_hrp != 0])), 5)

    # ---- D: HRP on its REAL job — out-of-sample portfolio variance ----
    v_hrp = oos_portfolio_vol(rets, alloc_hrp)
    v_mkv = oos_portfolio_vol(rets, alloc_raw_markowitz)
    v_eq = oos_portfolio_vol(rets, alloc_equal_weight)
    out["D_oos_vol_hrp"] = round(v_hrp, 4)
    out["D_oos_vol_raw_markowitz"] = round(v_mkv, 4)
    out["D_oos_vol_equalweight"] = round(v_eq, 4)
    out["D_hrp_beats_markowitz_on_oos_vol"] = bool(v_hrp <= v_mkv)        # the S8 finding
    out["D_hrp_competitive_with_equalweight"] = bool(v_hrp <= 1.10 * v_eq)

    # ---- A: leak-free (poison the future; pre-poison weights identical) ----
    k = len(rets) // 2
    rets_p = rets.copy()
    rng = np.random.default_rng(123)
    rets_p[k:] = rng.standard_normal(rets_p[k:].shape) * 0.05
    _, W_p = survivor_backtest(rets_p, sig, allocator=alloc_hrp)
    pre = max(0, k - 2)
    max_chg = float(np.abs(W[:pre] - W_p[:pre]).max())
    out["A_leakfree_max_weight_change_prepoison"] = round(max_chg, 12)
    out["A_leakfree_pass"] = bool(max_chg < 1e-12)

    out["VERDICT"] = dict(
        plumbing_leakfree=out["A_leakfree_pass"],
        captures_real_edge=bool(out["B_real_edge_net_sharpe"] > 0.3 and out["B_noise_net_sharpe"] < out["B_real_edge_net_sharpe"]),
        vol_target_reasonable=bool(0.3 * tgt < out["C_realized_vol_per_bar"] < 3 * tgt),
        hrp_does_its_job=bool(out["D_hrp_beats_markowitz_on_oos_vol"] and out["D_hrp_competitive_with_equalweight"]),
    )
    out["VERDICT"]["ALL_PASS"] = bool(all(out["VERDICT"].values()))
    out["NOTE"] = ("HRP is the risk layer (lower/stable OOS variance vs ILL-CONDITIONED Markowitz; "
                   "regime-dependent — HRP wins at q=win/N near 1, loses when Markowitz has ample data), "
                   "not the alpha; signal supplies direction; feedback trailing-vol vol-targets. Validated "
                   "on synthetic only; ready for a real-data WFO. COST-STRESS CAVEAT: at the default rebal "
                   "the book turns over ~580x/yr (~40% notional/yr at 7bp/side) and survives here only "
                   "because the synthetic Sharpe is large; a real lower-Sharpe book needs turnover control "
                   "(slower signal / turnover penalty / wider rebal) — validate cost-survivability before use.")
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, "survivor_skeleton_validation.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
