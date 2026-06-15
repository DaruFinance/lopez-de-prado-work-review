#!/usr/bin/env python3
"""Funding carry — rigorous IS-tuned walk-forward (no full-sample peeking).

The sweep showed a net-positive region (rebal~72, q~0.3) but picking it on the full sample
is multiple-testing bias. Here (rebal, q) are IS-TUNABLE knobs chosen on each window's
in-sample slice by best IS net Sharpe, then applied OUT-OF-SAMPLE. The concatenated OOS
path is judged by the same battery (tail-guard / sign-consistency / persistence) + a
block-bootstrap Sharpe-significance deflation. Carry book + costs + timing identical to
funding_carry.py (leak-free, verified there). Each config's pnl[t] is causal (carry signal
funding[t]<=t-1, price earns t+1, feedback vol-target uses trailing realized), so selecting
configs by an IS slice and reading their OOS slice introduces no leakage.
"""
import os
os.sched_setaffinity(0, range(16, 32))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import json
import numpy as np

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
from funding_carry import load_panel, carry_weights, backtest, VOL_LB
from s12_battery import sharpe, pf, block_boot_idx

IS, WALK, M, BLK = 12000, 6000, 2000, 24
GRID = [(rb, q) for rb in (24, 72, 168, 336) for q in (0.1, 0.2, 0.3)]


def main():
    ret, funding, valid = load_panel()
    T, N = ret.shape
    pnl_cfg = {}
    for (rb, q) in GRID:                                  # precompute each config's causal pnl
        W = carry_weights(funding, valid, q=q, rebal=rb)
        pnl, _ = backtest(ret, funding, W)
        pnl_cfg[(rb, q)] = pnl

    oos_pnl = np.full(T, np.nan); chosen = []; wins = []
    w = VOL_LB
    while w + IS + WALK <= T:
        ie, oe = w + IS, w + IS + WALK
        best, bs = None, -1e18
        for cfg in GRID:                                  # IS-select best config (in-sample only)
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
        strategy="funding carry, IS-tuned (rebal,q) walk-forward", grid=str(GRID),
        n_windows=nw, IS=IS, WALK=WALK,
        chosen_configs_per_window=[f"rebal{c[0]}_q{c[1]}" for c in chosen],
        oos_net_sharpe=round(net, 3), oos_pf=round(pf(oos), 3),
        per_window_oos_sharpe=wsh, pos_windows=pos,
        ex_worst_window_rrr=(round(rrr, 3) if np.isfinite(rrr) else None),
        early_half_sharpe=round(sharpe(oos[:med]), 3), late_half_sharpe=round(sharpe(oos[med:]), 3),
        tail_guard=bool(tg), sign_consistent=bool(sc), persistent=bool(pers),
        deflation_block_boot_p_not_positive=round(p_le0, 4),
        VERDICT="PASS" if verdict else "FAIL",
        read=("IS-tuned OOS net Sharpe is the honest figure (full-sample sweep best was ~0.44, biased). "
              "PASS only if it clears net>0.3 + tail-guard + sign-consistency + persistence + deflation."),
    )
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, "funding_carry_wfo_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
