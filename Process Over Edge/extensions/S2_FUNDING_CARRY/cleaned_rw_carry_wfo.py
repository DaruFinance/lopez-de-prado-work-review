#!/usr/bin/env python3
"""Funding carry: the recommended risk-weighted fix, tested honestly: risk-weighted
carry on a CLEAN universe (exclude pegged/stablecoin perps + bottom-30% illiquid names),
IS-tuned walk-forward, with a funding-vs-price DECOMPOSITION and GRID-AWARE deflation.

Why: full-sample probe showed excluding pegged/illiquid (a) collapses the narrow-quantile
configs (the gain came from the excluded names) but (b) the wider rb72_q0.3
config HOLDS with MORE funding (~62% funding-driven). So this is the decisive test: on a
clean tradeable universe, IS-tuned and honestly deflated, is the risk-weighted carry a real
funding edge or still the fragile low-vol price tilt?

Decision rule (anti-goalpost): PASS only if OOS net Sharpe > 0.3 AND tail-guard AND
sign-consistency AND persistence AND grid-aware deflation p<0.05 AND the OOS gain is
majority FUNDING (not the price tilt). Else honest negative.
"""
import os
os.sched_setaffinity(0, range(16, 32))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import glob
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
from funding_carry import backtest, VOL_LB
from risk_weighted_carry_wfo import carry_weights_rw
from s12_battery import sharpe, pf, block_boot_idx

IS, WALK, M, BLK = 12000, 6000, 2000, 24
GRID = [(rb, q) for rb in (24, 72, 168, 336) for q in (0.1, 0.2, 0.3)]
STABLE = {x.upper() for x in ("USDC", "DAI", "TUSD", "USDP", "FDUSD", "BUSD", "USDD", "USTC", "EUR", "GBP", "AEUR", "USDE")}


def main():
    npz = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "S12_clean_audit", "_cache", "panel_*.npz")))[-1]
    z = np.load(npz, allow_pickle=True)
    ret, funding, valid, dvol = z["ret"], z["funding"], z["valid"], z["dvol"]
    symbols = [str(s) for s in z["symbols"]]
    T, N = ret.shape
    base = lambda s: s.upper()[:-4] if s.upper().endswith("USDT") else s.upper()
    is_peg = np.array([base(s) in STABLE for s in symbols])
    namedvol = np.nanmedian(np.where(valid, dvol, np.nan), axis=0)
    liq_ok = namedvol >= np.nanpercentile(namedvol[np.isfinite(namedvol)], 30)
    keep = (~is_peg) & liq_ok
    valid_c = valid & keep[None, :]
    namevol = pd.DataFrame(np.nan_to_num(ret)).rolling(168, min_periods=84).std().shift(1).to_numpy()

    pnl_cfg, fund_cfg, price_cfg = {}, {}, {}
    for (rb, q) in GRID:
        W = carry_weights_rw(funding, valid_c, namevol, q=q, rebal=rb)
        pnl, comp = backtest(ret, funding, W)
        pnl_cfg[(rb, q)] = pnl; fund_cfg[(rb, q)] = comp["funding"]; price_cfg[(rb, q)] = comp["price"]

    oos_pnl = np.full(T, np.nan); oos_f = np.full(T, np.nan); oos_p = np.full(T, np.nan)
    wins = []; w = max(VOL_LB, 168)
    while w + IS + WALK <= T:
        ie, oe = w + IS, w + IS + WALK
        best, bs = None, -1e18
        for cfg in GRID:
            s = sharpe(pnl_cfg[cfg][w:ie])
            if np.isfinite(s) and s > bs:
                bs, best = s, cfg
        oos_pnl[ie:oe] = pnl_cfg[best][ie:oe]; oos_f[ie:oe] = fund_cfg[best][ie:oe]; oos_p[ie:oe] = price_cfg[best][ie:oe]
        wins.append((w, ie, oe)); w += WALK

    m = np.isfinite(oos_pnl); oos = oos_pnl[m]
    net = sharpe(oos)
    fund_bp = float(np.nansum(oos_f) * 1e4); price_bp = float(np.nansum(oos_p) * 1e4)
    gross_bp = fund_bp + price_bp
    funding_share = fund_bp / gross_bp if gross_bp != 0 else float("nan")
    wtot = np.array([np.nansum(oos_pnl[ie:oe]) for (_, ie, oe) in wins])
    wsh = [round(sharpe(oos_pnl[ie:oe][np.isfinite(oos_pnl[ie:oe])]), 2) for (_, ie, oe) in wins]
    worst = float(wtot.min()); rrr = float((wtot.sum() - worst) / abs(worst)) if worst < 0 else float("inf")
    pos = int((wtot > 0).sum()); nw = len(wins); med = len(oos) // 2
    tg = (worst >= 0) or (np.isfinite(rrr) and rrr > 1.0)
    sc = (pos / nw) >= 0.6
    pers = (sharpe(oos[:med]) > 0 and sharpe(oos[med:]) > 0)

    # GRID-AWARE deflation: cross-config max-Sharpe block-bootstrap null over all 12 demeaned configs
    rng = np.random.default_rng(7)
    cfg_series = {cfg: pnl_cfg[cfg][np.isfinite(oos_pnl)] for cfg in GRID}  # aligned to OOS region
    dem = {cfg: cfg_series[cfg] - cfg_series[cfg].mean() for cfg in GRID}
    null_max = np.array([max(sharpe(dem[c][block_boot_idx(len(oos), BLK, rng)]) for c in GRID) for _ in range(M)])
    p_grid = float((null_max >= net).mean())

    funding_majority = funding_share > 0.5
    verdict = bool(net > 0.3 and tg and sc and pers and p_grid < 0.05 and funding_majority)
    out = dict(
        strategy="risk-weighted funding carry, CLEAN universe (no pegged/illiquid), IS-tuned WFO, grid-aware deflation",
        universe_total=N, pegged_excluded=int(is_peg.sum()), illiquid_excluded=int((~liq_ok).sum()), kept=int(keep.sum()),
        oos_net_sharpe=round(net, 3), oos_pf=round(pf(oos), 3),
        oos_funding_bp=round(fund_bp, 0), oos_price_bp=round(price_bp, 0), funding_share_of_gross=round(funding_share, 3),
        per_window_oos_sharpe=wsh, pos_windows=pos,
        ex_worst_window_rrr=(round(rrr, 3) if np.isfinite(rrr) else None),
        early_half_sharpe=round(sharpe(oos[:med]), 3), late_half_sharpe=round(sharpe(oos[med:]), 3),
        tail_guard=bool(tg), sign_consistent=bool(sc), persistent=bool(pers),
        grid_aware_deflation_p=round(p_grid, 4), funding_majority=bool(funding_majority),
        VERDICT="PASS" if verdict else "FAIL",
        read="PASS requires net>0.3 + tail-guard + sign + persistence + grid-deflation<0.05 + funding-majority. If it passes AND is funding-driven, the clean-universe RW carry is a real edge (revises the negative). If it fails or is price-tilt-driven, it's confirmed not tradeable.",
    )
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, "cleaned_rw_carry_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
