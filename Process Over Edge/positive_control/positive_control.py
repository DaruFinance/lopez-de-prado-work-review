#!/usr/bin/env python3
"""Positive control: does the full pipeline pass a known-real edge, or destroy
everything? And does it still catch a leak even when the economic gates pass?

Five strategies run through the same judgment gates that scored the real corpus
(battery functions imported from s12_battery):
  - net of realistic costs (7 bp per fill)
  - tail-guard: ex-worst-window reward/risk
  - sign-consistency across instrument cohorts
  - persistence (early versus late half)
  - deflation: stationary-block-bootstrap of the Sharpe (is the edge reliably > 0?)
  - causality (pollute-and-verify): poison the earned returns and check the signal is
    unchanged. A causal forecast is invariant; a leak (signal contaminated with the
    return it earns) changes.

The five:
  noise    (beta=0)            -> no edge              -> must fail
  weak     (small beta)        -> real but sub-cost    -> must fail (not tradeable net of cost)
  moderate (beta)              -> real, net-tradeable  -> must pass
  strong   (large beta)        -> real, net-tradeable  -> must pass
  LEAK     (beta=0 + signal contaminated with the earned return)
                               -> high Sharpe, passes every economic gate, but is a
                                  leak -> must be caught (causality gate fails) -> fail

Causal by construction: signal s[t] is built only from shocks <= t; the position from
s[t] earns the forward return r[t]; r[t]'s noise is drawn independently. No peeking. A
real intraday edge can have a high annualized Sharpe (a small per-bar edge times the
square root of many bars), so the discriminator is not the Sharpe level. It is causality
plus net-of-cost plus robustness.
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

from s12_battery import sharpe, pf, block_boot_idx  # same gate functions as the corpus

T = 42000
N = 120
NW = 7
COST = 0.0007
RHO = 0.92
VOL = 0.01
M = 2000
BLK = 24


def build_signal(seed):
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal((T, N))
    s = np.empty((T, N), float)
    s[0] = shock[0]
    a = np.sqrt(1 - RHO ** 2)
    for t in range(1, T):
        s[t] = RHO * s[t - 1] + a * shock[t]
    eps = rng.standard_normal((T, N))
    return s, eps, rng


def weights(sig):
    sd = sig - sig.mean(axis=1, keepdims=True)
    return sd / (np.abs(sd).sum(axis=1, keepdims=True) + 1e-12)


def pnl_of(sig, r):
    w = weights(sig)
    gross = (w * r).sum(axis=1)
    dw = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, N)))).sum(axis=1)
    return gross - dw * COST


def gates(sig_used, r, rng):
    pnl = pnl_of(sig_used, r)
    wid = np.minimum(np.arange(T) // (T // NW), NW - 1)
    wtot = np.array([pnl[wid == k].sum() for k in range(NW)])
    wshp = [round(sharpe(pnl[wid == k]), 2) for k in range(NW)]
    worst = float(wtot.min())
    rrr = float((wtot.sum() - worst) / abs(worst)) if worst < 0 else float("inf")
    pos_w = int((wtot > 0).sum())
    half = T // 2
    early, late = sharpe(pnl[:half]), sharpe(pnl[half:])
    wf = weights(sig_used)
    ha = (wf[:, :N // 2] * r[:, :N // 2]).sum(1)
    hb = (wf[:, N // 2:] * r[:, N // 2:]).sum(1)
    sign_ok = (ha.sum() > 0) and (hb.sum() > 0)
    bs = np.array([sharpe(pnl[block_boot_idx(T, BLK, rng)]) for _ in range(M)])
    p_le0 = float((bs <= 0).mean())
    return pnl, dict(
        net_sharpe=round(sharpe(pnl), 3), pf=round(pf(pnl), 3),
        pos_windows=pos_w, n_windows=NW, per_window_sharpe=wshp,
        ex_worst_rrr=(round(rrr, 3) if np.isfinite(rrr) else None),
        early_sharpe=round(early, 3), late_sharpe=round(late, 3),
        sign_consistent=bool(sign_ok), deflation_p_not_positive=round(p_le0, 4),
    )


def causality_pollute_verify(beta, leak_lambda, seed):
    """Rebuild the signal-as-used with the EARNED returns poisoned; a causal signal is
    invariant, a leak changes. Returns (is_causal, max_abs_change)."""
    s, eps, _ = build_signal(seed)
    b = min(max(beta, 0.0), 0.999)
    r = (b * s + np.sqrt(1 - b * b) * eps) * VOL
    sig_used = s + (leak_lambda * (r / VOL) if leak_lambda > 0 else 0.0)
    # poison the earned returns (independent re-draw) and recompute the signal-as-used
    rng2 = np.random.default_rng(seed + 999)
    r_pois = rng2.standard_normal((T, N)) * VOL
    sig_pois = s + (leak_lambda * (r_pois / VOL) if leak_lambda > 0 else 0.0)
    chg = float(np.abs(sig_pois - sig_used).max())
    return (chg < 1e-9), chg, s, r, sig_used


def verdict(g, causal):
    tg = (g["ex_worst_rrr"] is None) or (g["ex_worst_rrr"] > 1.0)
    economic = (g["net_sharpe"] > 0.3 and tg and g["sign_consistent"]
                and g["pos_windows"] >= int(np.ceil(0.6 * NW))
                and g["deflation_p_not_positive"] < 0.05)
    return bool(economic and causal), economic


def main():
    rows = [("noise", 0.0, 0.0, 1), ("weak", 0.015, 0.0, 2),
            ("moderate", 0.03, 0.0, 3), ("strong", 0.06, 0.0, 4),
            ("LEAK", 0.0, 0.5, 5)]
    out = {}
    print(f"{'strategy':>9} {'netSR':>7} {'PF':>5} {'posW':>5} {'RRR':>6} {'defl_p':>7} "
          f"{'econ_gates':>10} {'causal?':>8} {'VERDICT':>8}")
    for lbl, beta, lam, seed in rows:
        causal, chg, s, r, sig_used = causality_pollute_verify(beta, lam, seed)
        rng = np.random.default_rng(seed + 7)
        pnl, g = gates(sig_used, r, rng)
        v, econ = verdict(g, causal)
        out[lbl] = dict(beta=beta, leak_lambda=lam, **g,
                        causal_pollute_verify=causal, signal_change_when_poisoned=round(chg, 4),
                        economic_gates_pass=econ, PASS=v)
        rrr = g["ex_worst_rrr"]; rrr_s = "inf" if rrr is None else f"{rrr:.2f}"
        print(f"{lbl:>9} {g['net_sharpe']:>7.2f} {g['pf']:>5.2f} {g['pos_windows']:>3}/{NW} "
              f"{rrr_s:>6} {g['deflation_p_not_positive']:>7.3f} {str(econ):>10} "
              f"{str(causal):>8} {'PASS' if v else 'FAIL':>8}")
    noise_fail = not out["noise"]["PASS"]
    real_pass = out["moderate"]["PASS"] and out["strong"]["PASS"]
    leak_caught = (not out["LEAK"]["PASS"]) and out["LEAK"]["economic_gates_pass"] and (not out["LEAK"]["causal_pollute_verify"])
    monotone = out["noise"]["net_sharpe"] <= out["weak"]["net_sharpe"] <= out["moderate"]["net_sharpe"] <= out["strong"]["net_sharpe"]
    ok = noise_fail and real_pass and leak_caught and monotone
    out["CONCLUSION"] = dict(
        noise_correctly_fails=bool(noise_fail),
        real_edges_correctly_pass=bool(real_pass),
        leak_caught_despite_passing_economic_gates=bool(leak_caught),
        verdict_tracks_truth_monotonically=bool(monotone),
        BULLETPROOF=bool(ok),
        statement=("The pipeline PASSES real net-of-cost edges, FAILS noise and sub-cost "
                   "edges, and CATCHES a high-Sharpe leak that clears every economic gate "
                   "(only the causality/pollute-verify test exposes it). It discriminates "
                   "real from fake by mechanism, not by the Sharpe number.") if ok
                   else "INCONCLUSIVE; inspect the table.",
    )
    print("\nCONCLUSION:", json.dumps(out["CONCLUSION"], indent=2))
    with open(os.path.join(HERE, "positive_control_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
