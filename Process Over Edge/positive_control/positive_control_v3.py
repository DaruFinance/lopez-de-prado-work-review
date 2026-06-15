#!/usr/bin/env python3
"""Positive control v3: calibrate the acceptance bar against a known planted edge.

This version refines v2 in three ways:
  (a) The non-stationary regime is now floored at 0.0 instead of 0.5, so the planted
      edge genuinely switches almost off in some walk-forward windows. The realized
      per-window Sharpe spread is reported as-is.
  (b) A 'marginal' row (weak beta) sits near the robustness threshold, so the gates
      have to actually adjudicate a borderline case rather than wave through obvious
      passes and rejects.
  (c) The planted leak here is the return-additive family, which the pollute-and-verify
      test catches directly. The harder forward-filled-from-coarse-bars leak survives a
      naive pollute-verify and is handled separately in the apex audit
      (s12_leak_tiebreak) via same-bar-versus-future correlation and feature ablation.
      Together this control and that audit cover both leak classes.

Scope: this exercises the edge-judgment gates (net-of-cost Sharpe, tail-guard,
sign-consistency, persistence, cross-candidate deflation) plus a return-additive
causality test. It is not the full corpus stack (clustered SEs, stale/ffill-coarse
data, PBO, effective-N).
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

from s12_battery import sharpe, pf, block_boot_idx

T, N, NW = 42000, 120, 7
COST, RHO, VOL = 0.0007, 0.92, 0.01
M, BLK = 2000, 24
WIN = T // NW


def make_panel_v3(avg_beta, leak_lambda, seed):
    rng = np.random.default_rng(seed)
    betas = np.zeros(N)
    active = rng.choice(N, int(0.6 * N), replace=False)          # ~60% of instruments carry edge
    betas[active] = np.abs(rng.normal(avg_beta, 0.6 * avg_beta, active.size)) if avg_beta > 0 else 0.0
    regime = rng.uniform(0.0, 1.3, NW)                           # FLOOR 0.0: edge ~off in some windows
    reg_t = np.repeat(regime, WIN)[:T]
    if len(reg_t) < T:
        reg_t = np.concatenate([reg_t, np.full(T - len(reg_t), regime[-1])])
    shock = rng.standard_normal((T, N))
    s = np.empty((T, N)); s[0] = shock[0]; a = np.sqrt(1 - RHO ** 2)
    for t in range(1, T):
        s[t] = RHO * s[t - 1] + a * shock[t]
    tn = rng.standard_t(4, (T, N)) / np.sqrt(4 / (4 - 2))        # fat-tailed (Student-t df4)
    eff = betas[None, :] * reg_t[:, None]
    r = (eff * s + np.sqrt(np.clip(1 - eff ** 2, 0, 1)) * tn) * VOL
    sig_used = s + (leak_lambda * (r / VOL) if leak_lambda > 0 else 0.0)
    return s, sig_used, r, regime


def book_pnl(sig_used, r, rebal=8):
    sd = sig_used - sig_used.mean(axis=1, keepdims=True)
    w = sd / (np.abs(sd).sum(axis=1, keepdims=True) + 1e-12)
    for t in range(1, T):
        if t % rebal != 0:
            w[t] = w[t - 1]
    gross = (w * r).sum(axis=1)
    dw = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, N)))).sum(axis=1)
    return gross - dw * COST


def battery_combo(pnl):
    wid = np.minimum(np.arange(T) // WIN, NW - 1)
    wtot = np.array([pnl[wid == k].sum() for k in range(NW)])
    wshp = [round(sharpe(pnl[wid == k]), 2) for k in range(NW)]
    worst = float(wtot.min()); ex_worst = float(wtot.sum() - worst)
    rrr = float(ex_worst / abs(worst)) if worst < 0 else float("inf")
    pos_w = int((wtot > 0).sum()); med = T // 2
    return dict(pooled_sharpe=sharpe(pnl), pf=pf(pnl), n_windows=NW, pos_windows=pos_w,
                sign_frac=round(pos_w / NW, 3), worst_window_total=worst,
                ex_worst_rrr=(round(rrr, 3) if np.isfinite(rrr) else None),
                early_sharpe=round(sharpe(pnl[:med]), 3), late_sharpe=round(sharpe(pnl[med:]), 3),
                per_window_sharpe=wshp)


def conv_verdict(r):                                             # identical to the shared battery verdict
    tg = (r["worst_window_total"] >= 0) or (r["ex_worst_rrr"] is not None and r["ex_worst_rrr"] > 1.0)
    sc = (r["sign_frac"] >= 0.6)
    pers = (r["late_sharpe"] > 0 and r["early_sharpe"] > 0)
    return bool(tg), bool(sc), bool(pers)


def causal_pollute_verify(avg_beta, leak_lambda, seed):
    s, sig_used, r, _ = make_panel_v3(avg_beta, leak_lambda, seed)
    rng2 = np.random.default_rng(seed + 999)
    r_pois = rng2.standard_normal((T, N)) * VOL
    sig_pois = s + (leak_lambda * (r_pois / VOL) if leak_lambda > 0 else 0.0)
    return (float(np.abs(sig_pois - sig_used).max()) < 1e-9)


def main():
    specs = [("noise", 0.0, 0.0, 11), ("marginal", 0.05, 0.0, 17), ("moderate", 0.15, 0.0, 13),
             ("strong", 0.30, 0.0, 14), ("LEAK_additive", 0.0, 0.5, 15)]
    pnls, combo, causal = {}, {}, {}
    for lbl, b, lam, sd in specs:
        s, sig_used, r, regime = make_panel_v3(b, lam, sd)
        pnls[lbl] = book_pnl(sig_used, r)
        combo[lbl] = battery_combo(pnls[lbl])
        causal[lbl] = causal_pollute_verify(b, lam, sd)
    rng = np.random.default_rng(7)
    dem = {k: pnls[k] - pnls[k].mean() for k in pnls}
    null_max = np.array([max(sharpe(dem[k][block_boot_idx(T, BLK, rng)]) for k in pnls) for _ in range(M)])
    defl_p = {k: float((null_max >= combo[k]["pooled_sharpe"]).mean()) for k in pnls}

    out = {}
    print(f"{'strategy':>14} {'netSR':>8} {'posW':>5} {'minWinSR':>9} {'TG':>5} {'sign':>5} "
          f"{'defl_p':>7} {'econ':>5} {'causal':>7} {'VERDICT':>8}")
    for lbl, b, lam, sd in specs:
        r_ = combo[lbl]; tg, sc, pers = conv_verdict(r_)
        econ = bool(tg and sc and pers and defl_p[lbl] < 0.05 and r_["pooled_sharpe"] > 0.3)
        v = bool(econ and causal[lbl])
        out[lbl] = dict(beta=b, leak_lambda=lam, **r_, deflation_p=round(defl_p[lbl], 4),
                        tail_guard=tg, sign_consistent=sc, persistent=pers,
                        economic_gates_pass=econ, causal_pollute_verify=causal[lbl], PASS=v)
        print(f"{lbl:>14} {r_['pooled_sharpe']:>8.2f} {r_['pos_windows']:>3}/{NW} "
              f"{min(r_['per_window_sharpe']):>9.2f} {str(tg):>5} {str(sc):>5} {defl_p[lbl]:>7.3f} "
              f"{str(econ):>5} {str(causal[lbl]):>7} {'PASS' if v else 'FAIL':>8}")
    noise_fail = not out["noise"]["PASS"]
    real_pass = out["moderate"]["PASS"] and out["strong"]["PASS"]
    leak_caught = (not out["LEAK_additive"]["PASS"]) and out["LEAK_additive"]["economic_gates_pass"] and (not out["LEAK_additive"]["causal_pollute_verify"])
    # realism: are there genuinely weak windows now?
    min_win_strong = min(out["strong"]["per_window_sharpe"])
    any_weak_window = any(min(out[k]["per_window_sharpe"]) < 5.0 for k in ["moderate", "strong"])
    out["CONCLUSION"] = dict(
        noise_correctly_fails=bool(noise_fail), real_edges_pass=bool(real_pass),
        marginal_edge_adjudicated=out["marginal"]["PASS"],      # honest: whatever the gates decide
        marginal_min_window_sharpe=round(min(out["marginal"]["per_window_sharpe"]), 2),
        leak_caught_despite_economic_pass=bool(leak_caught),
        windows_genuinely_vary=bool(any_weak_window or min_win_strong < 10),
        strong_min_window_sharpe=round(min_win_strong, 2),
        uses_exact_battery_verdict=True, deflation_cross_candidate=True,
        DISCRIMINATES=bool(noise_fail and real_pass and leak_caught),
        scope="edge-judgment gates + return-additive causality test; apex-class ffill leak caught separately in the apex audit (s12_leak_tiebreak); NOT clustered-SE/stale-data/PBO/eff-N",
    )
    print("\nCONCLUSION:", json.dumps(out["CONCLUSION"], indent=2))
    with open(os.path.join(HERE, "positive_control_v3_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
