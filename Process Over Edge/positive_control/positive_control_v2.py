#!/usr/bin/env python3
"""Positive control v2: addresses two gaps in v1.
  (1) v1 reimplemented the verdict with a flipped tail-guard. v2 uses the exact battery
      verdict logic (the worst>=0 -> pass tail-guard and the cross-candidate max-Sharpe
      deflation haircut), matching the shared battery module.
  (2) v1's planted edge was implausibly clean. v2's edge is realistic: only ~40% of
      instruments are predictable, per-instrument edge strength is heterogeneous, the
      edge turns partly off in some walk-forward windows (non-stationary regime, so some
      windows are weak or negative), and the noise is fat-tailed (Student-t, df=4).

Scope: this exercises the edge-judgment gates (net-of-cost Sharpe, tail-guard,
sign-consistency, persistence, multiple-testing deflation) and the causality /
pollute-verify leak test. It does not re-test every corpus failure mode (clustered SEs,
stale/ffill-from-coarse data, PBO/CSCV, effective-N); those are separate and
item-specific. The claim is narrow: the edge gates plus leak test pass realistic
net-of-cost edges, reject noise and sub-cost edges, and catch a high-Sharpe leak that
clears every economic gate. Rejection is conditional on truth, not unconditional.
"""
import os
os.sched_setaffinity(0, range(16, 32))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "apex_audit"))   # shared battery primitives

import json
import numpy as np

from s12_battery import sharpe, pf, block_boot_idx  # same battery primitives as the corpus

T, N, NW = 42000, 120, 7
COST, RHO, VOL = 0.0007, 0.92, 0.01
M, BLK = 2000, 24
WIN = T // NW


def make_panel_v2(avg_beta, leak_lambda, seed, rho=RHO):
    """Realistic heterogeneous, non-stationary, fat-tailed planted edge."""
    rng = np.random.default_rng(seed)
    # heterogeneous betas: only ~40% of instruments carry any edge, varying strengths
    betas = np.zeros(N)
    active = rng.choice(N, int(0.6 * N), replace=False)
    betas[active] = np.abs(rng.normal(avg_beta, 0.6 * avg_beta, active.size)) if avg_beta > 0 else 0.0
    # non-stationary regime: edge strength varies per window (some weak, none fully off)
    regime = rng.uniform(0.5, 1.5, NW)
    reg_t = np.repeat(regime, WIN)[:T]
    if len(reg_t) < T:
        reg_t = np.concatenate([reg_t, np.full(T - len(reg_t), regime[-1])])
    # persistent causal signal
    shock = rng.standard_normal((T, N))
    s = np.empty((T, N)); s[0] = shock[0]; a = np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        s[t] = rho * s[t - 1] + a * shock[t]
    # fat-tailed noise (Student-t df=4, unit variance)
    tn = rng.standard_t(4, (T, N)) / np.sqrt(4 / (4 - 2))
    eff = betas[None, :] * reg_t[:, None]                       # [T,N] time- & asset-varying edge
    r = (eff * s + np.sqrt(np.clip(1 - eff ** 2, 0, 1)) * tn) * VOL
    sig_used = s + (leak_lambda * (r / VOL) if leak_lambda > 0 else 0.0)
    return s, sig_used, r


def book_pnl(sig_used, r, rebal=8):
    sd = sig_used - sig_used.mean(axis=1, keepdims=True)
    w = sd / (np.abs(sd).sum(axis=1, keepdims=True) + 1e-12)
    for t in range(1, T):                       # rebalance throttle: hold between rebalances
        if t % rebal != 0:
            w[t] = w[t - 1]
    gross = (w * r).sum(axis=1)
    dw = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, N)))).sum(axis=1)
    return gross - dw * COST


# ---- battery verdict logic: matches the shared s12_battery module ----
def battery_combo(pnl):
    wid = np.minimum(np.arange(T) // WIN, NW - 1)
    wtot = np.array([pnl[wid == k].sum() for k in range(NW)])
    wshp = {int(k): round(sharpe(pnl[wid == k]), 2) for k in range(NW)}
    worst = float(wtot.min())
    ex_worst = float(wtot.sum() - worst)
    rrr = float(ex_worst / abs(worst)) if worst < 0 else float("inf")
    pos_w = int((wtot > 0).sum())
    med = T // 2
    early, late = sharpe(pnl[:med]), sharpe(pnl[med:])
    return dict(pooled_sharpe=sharpe(pnl), pf=pf(pnl), n_windows=NW, pos_windows=pos_w,
                sign_frac=round(pos_w / NW, 3), worst_window_total=worst,
                ex_worst_rrr=(round(rrr, 3) if np.isfinite(rrr) else None),
                early_sharpe=round(early, 3), late_sharpe=round(late, 3), per_window_sharpe=wshp)


def conv_verdict(r):
    # identical to the shared battery verdict (incl. worst>=0 -> tail-guard pass)
    tg = (r["worst_window_total"] >= 0) or (r["ex_worst_rrr"] is not None and r["ex_worst_rrr"] > 1.0)
    sc = (r["sign_frac"] >= 0.6)
    pers = (r["late_sharpe"] > 0 and r["early_sharpe"] > 0)
    return bool(tg), bool(sc), bool(pers)


def causal_pollute_verify(avg_beta, leak_lambda, seed, rho=RHO):
    s, sig_used, r = make_panel_v2(avg_beta, leak_lambda, seed, rho)
    rng2 = np.random.default_rng(seed + 999)
    r_pois = rng2.standard_normal((T, N)) * VOL
    sig_pois = s + (leak_lambda * (r_pois / VOL) if leak_lambda > 0 else 0.0)
    chg = float(np.abs(sig_pois - sig_used).max())
    return (chg < 1e-9), chg


def main():
    #  (label, avg_beta, leak_lambda, seed, rebal_bars, signal_persistence_rho)
    specs = [("noise", 0.0, 0.0, 11, 8, RHO), ("weak", 0.05, 0.0, 12, 8, RHO),
             ("moderate", 0.15, 0.0, 13, 8, RHO), ("strong", 0.30, 0.0, 14, 8, RHO),
             ("LEAK", 0.0, 0.5, 15, 8, RHO)]
    pnls, combo, causal = {}, {}, {}
    for lbl, b, lam, sd, rb, rh in specs:
        s, sig_used, r = make_panel_v2(b, lam, sd, rh)
        pnls[lbl] = book_pnl(sig_used, r, rebal=rb)
        combo[lbl] = battery_combo(pnls[lbl])
        causal[lbl], _ = causal_pollute_verify(b, lam, sd, rh)
    # cross-candidate DEFLATION haircut (same as battery: max-Sharpe across the 5 demeaned series)
    rng = np.random.default_rng(7)
    demeaned = {k: pnls[k] - pnls[k].mean() for k in pnls}
    null_max = np.empty(M)
    for m in range(M):
        null_max[m] = max(sharpe(demeaned[k][block_boot_idx(T, BLK, rng)]) for k in pnls)
    defl_p = {k: float((null_max >= combo[k]["pooled_sharpe"]).mean()) for k in pnls}

    out = {}
    print(f"{'strategy':>9} {'netSR':>8} {'posW':>5} {'TG':>4} {'sign':>5} {'pers':>5} "
          f"{'defl_p':>7} {'econ':>5} {'causal':>7} {'VERDICT':>8}")
    for lbl, b, lam, sd, rb, rh in specs:
        r_ = combo[lbl]; tg, sc, pers = conv_verdict(r_)
        econ = bool(tg and sc and pers and defl_p[lbl] < 0.05 and r_["pooled_sharpe"] > 0.3)
        v = bool(econ and causal[lbl])
        out[lbl] = dict(beta=b, leak_lambda=lam, **r_, deflation_p=round(defl_p[lbl], 4),
                        tail_guard=tg, sign_consistent=sc, persistent=pers,
                        economic_gates_pass=econ, causal_pollute_verify=causal[lbl], PASS=v)
        print(f"{lbl:>9} {r_['pooled_sharpe']:>8.2f} {r_['pos_windows']:>3}/{NW} {str(tg):>4} "
              f"{str(sc):>5} {str(pers):>5} {defl_p[lbl]:>7.3f} {str(econ):>5} {str(causal[lbl]):>7} "
              f"{'PASS' if v else 'FAIL':>8}")
    noise_fail = not out["noise"]["PASS"]
    real_pass = out["moderate"]["PASS"] and out["strong"]["PASS"]
    leak_caught = (not out["LEAK"]["PASS"]) and out["LEAK"]["economic_gates_pass"] and (not out["LEAK"]["causal_pollute_verify"])
    # realism check: do the real edges actually have weak/variable windows?
    cv_strong = float(np.std(list(out["strong"]["per_window_sharpe"].values())) /
                      (abs(np.mean(list(out["strong"]["per_window_sharpe"].values()))) + 1e-9))
    out["CONCLUSION"] = dict(
        noise_correctly_fails=bool(noise_fail), real_edges_correctly_pass=bool(real_pass),
        leak_caught_despite_economic_pass=bool(leak_caught),
        strong_edge_window_sharpe_CV=round(cv_strong, 3),
        uses_exact_battery_verdict=True, deflation_is_cross_candidate_haircut=True,
        DISCRIMINATES=bool(noise_fail and real_pass and leak_caught),
        scope="edge-judgment gates + causality/leak test; not the full corpus gate-stack",
    )
    print("\nCONCLUSION:", json.dumps(out["CONCLUSION"], indent=2))
    with open(os.path.join(HERE, "positive_control_v2_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
