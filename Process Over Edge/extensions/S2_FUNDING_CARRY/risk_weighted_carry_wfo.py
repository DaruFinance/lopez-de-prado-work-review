#!/usr/bin/env python3
"""Funding carry — targeted fix attempt: inverse-volatility leg weighting.

The carry's failure has a clear diagnosis: the funding premium is real, stable,
and persistent; the book is already dollar/beta-neutral; what kills the net is the
IDIOSYNCRATIC PRICE-RESIDUAL VARIANCE (high-vol names dominate the equal-weighted legs and
their idiosyncratic moves swamp the thin carry). Prior tests covered turnover (rebal/q) but
not leg risk-weighting. This script attacks the diagnosed problem directly: within each leg,
weight names INVERSELY to their trailing (causal) volatility, so volatile names contribute
less variance. If the price-residual shrinks enough, the stable carry may survive.

Honest IS-tuned walk-forward (rebal,q IS-selected per window), same battery + deflation,
leak-free (signal funding[t]<=t-1; per-name vol trailing+shift1; price earns t+1).
Comparison baseline: the equal-weighted carry's OOS Sharpe was 0.335 (FAIL).
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "S12_clean_audit"))
from funding_carry import load_panel, backtest, VOL_LB
from s12_battery import sharpe, pf, block_boot_idx

IS, WALK, M, BLK, NVOL = 12000, 6000, 2000, 24, 168
GRID = [(rb, q) for rb in (24, 72, 168, 336) for q in (0.1, 0.2, 0.3)]


def carry_weights_rw(funding, valid, namevol, q, rebal):
    """Long lowest-funding quantile, short highest; INVERSE-VOL weighted within each leg."""
    T, N = funding.shape
    csig = -funding
    W = np.zeros((T, N)); wprev = np.zeros(N)
    for t in range(max(VOL_LB, NVOL), T - 1):
        if t % rebal != 0:
            W[t] = wprev; continue
        m = valid[t] & np.isfinite(csig[t]) & (funding[t] != 0) & np.isfinite(namevol[t]) & (namevol[t] > 1e-9)
        idx = np.where(m)[0]
        if idx.size < 20:
            W[t] = wprev; continue
        sv = csig[t][idx]; k = max(1, int(q * idx.size)); order = np.argsort(sv)
        longs = idx[order[-k:]]; shorts = idx[order[:k]]
        w = np.zeros(N)
        wl = 1.0 / namevol[t][longs]; wl = wl / wl.sum(); w[longs] = wl       # inverse-vol within leg
        ws = 1.0 / namevol[t][shorts]; ws = ws / ws.sum(); w[shorts] = -ws
        W[t] = w; wprev = w
    return W


def main():
    ret, funding, valid = load_panel()
    T, N = ret.shape
    namevol = pd.DataFrame(np.nan_to_num(ret)).rolling(NVOL, min_periods=NVOL // 2).std().shift(1).to_numpy()
    pnl_cfg = {}
    for (rb, q) in GRID:
        W = carry_weights_rw(funding, valid, namevol, q=q, rebal=rb)
        pnl, _ = backtest(ret, funding, W)
        pnl_cfg[(rb, q)] = pnl

    oos_pnl = np.full(T, np.nan); chosen = []; wins = []
    w = max(VOL_LB, NVOL)
    while w + IS + WALK <= T:
        ie, oe = w + IS, w + IS + WALK
        best, bs = None, -1e18
        for cfg in GRID:
            s = sharpe(pnl_cfg[cfg][w:ie])
            if np.isfinite(s) and s > bs:
                bs, best = s, cfg
        oos_pnl[ie:oe] = pnl_cfg[best][ie:oe]
        chosen.append(best); wins.append((w, ie, oe))
        w += WALK

    m = np.isfinite(oos_pnl); oos = oos_pnl[m]
    net = sharpe(oos)
    wtot = np.array([np.nansum(oos_pnl[ie:oe]) for (_, ie, oe) in wins])
    wsh = [round(sharpe(oos_pnl[ie:oe][np.isfinite(oos_pnl[ie:oe])]), 2) for (_, ie, oe) in wins]
    worst = float(wtot.min()); rrr = float((wtot.sum() - worst) / abs(worst)) if worst < 0 else float("inf")
    pos = int((wtot > 0).sum()); nw = len(wins); med = len(oos) // 2
    tg = (worst >= 0) or (np.isfinite(rrr) and rrr > 1.0)
    sc = (pos / nw) >= 0.6
    pers = (sharpe(oos[:med]) > 0 and sharpe(oos[med:]) > 0)
    rng = np.random.default_rng(7)
    bsd = np.array([sharpe(oos[block_boot_idx(len(oos), BLK, rng)]) for _ in range(M)])
    p_le0 = float((bsd <= 0).mean())
    verdict = bool(net > 0.3 and tg and sc and pers and p_le0 < 0.05)
    out = dict(
        strategy="funding carry, INVERSE-VOL leg weighting, IS-tuned WFO",
        baseline_equalweight_oos_sharpe=0.335, n_windows=nw,
        oos_net_sharpe=round(net, 3), oos_pf=round(pf(oos), 3),
        per_window_oos_sharpe=wsh, pos_windows=pos,
        ex_worst_window_rrr=(round(rrr, 3) if np.isfinite(rrr) else None),
        early_half_sharpe=round(sharpe(oos[:med]), 3), late_half_sharpe=round(sharpe(oos[med:]), 3),
        tail_guard=bool(tg), sign_consistent=bool(sc), persistent=bool(pers),
        deflation_block_boot_p_not_positive=round(p_le0, 4),
        VERDICT="PASS" if verdict else "FAIL",
        read="Inverse-vol leg weighting targets the diagnosed price-residual variance. PASS only if it now clears net>0.3 + tail-guard + sign + persistence + deflation; else the carry is genuinely not tradeable as a simple cross-sectional book.",
    )
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, "risk_weighted_carry_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
