#!/usr/bin/env python3
"""New idea through the same machine: cross-sectional FUNDING-RATE CARRY on Binance
perps. Demonstrates the rigorous pipeline is reusable on a fresh strategy.

Idea: collect the funding spread. Perps with HIGH positive funding pay longs->shorts, so
go SHORT them; perps with NEGATIVE funding pay shorts->longs, so go LONG them. Market-
neutral cross-sectional book (long the lowest-funding quantile, short the highest).

Leak-free timing (this whole project is about leakage, so be exact):
  - signal at decision time t = funding[t]  (panel funding is ALREADY shift(1) => <= t-1).
  - price earned over the hold = ret[t+1]    (forward; leak-free).
  - funding collected over the hold = -(W[t]·funding[t])·(1/8)  (funding[t] <= t-1; funding
    is an 8h rate ffilled hourly + sticky, so funding[t] ~ the rate over [t,t+1]; using the
    <= t-1 value is conservative and unambiguously causal). 1/8 scales 8h->1h (= S12 conv).
  - pollute-verify: poison the FUTURE (ret & funding) -> pre-poison weights bit-identical.

Evaluation = same machine: net of 7bp/fill costs, feedback vol-target, per-window tail-guard
/ sign-consistency / persistence (fixed s12_battery verdict), block-bootstrap Sharpe
significance, and a price-vs-funding DECOMPOSITION (is any edge genuine carry or incidental
price?). Honest verdict.
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
from s12_battery import sharpe, pf, block_boot_idx

COST, FUND_SCALE, NW = 0.0007, 1.0 / 8.0, 7
M, BLK, VOL_LB = 2000, 24, 240
TARGET_VOL, LEV_CAP, Q, REBAL = 0.0015, 5.0, 0.2, 8
ANN = np.sqrt(252 * 24)


def load_panel():
    npz = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "S12_clean_audit", "_cache", "panel_*.npz")))[-1]
    z = np.load(npz, allow_pickle=True)
    return z["ret"], z["funding"], z["valid"]


def carry_weights(funding, valid, q=Q, rebal=REBAL):
    """Long lowest-funding quantile, short highest. Market-neutral, gross=2, throttled."""
    T, N = funding.shape
    csig = -funding                                   # high csig = low funding -> long
    W = np.zeros((T, N)); wprev = np.zeros(N)
    for t in range(VOL_LB, T - 1):
        if t % rebal != 0:
            W[t] = wprev; continue
        m = valid[t] & np.isfinite(csig[t]) & (funding[t] != 0)
        idx = np.where(m)[0]
        if idx.size < 20:
            W[t] = wprev; continue
        sv = csig[t][idx]; k = max(1, int(q * idx.size))
        order = np.argsort(sv)
        w = np.zeros(N)
        w[idx[order[-k:]]] = 1.0 / k                  # longs: lowest funding
        w[idx[order[:k]]] = -1.0 / k                  # shorts: highest funding
        W[t] = w; wprev = w
    return W


def backtest(ret, funding, W, target_vol=TARGET_VOL):
    T, N = ret.shape
    fnd = np.nan_to_num(funding); rt = np.nan_to_num(ret)
    price_u = np.zeros(T); price_u[:-1] = (W[:-1] * rt[1:]).sum(1)        # forward price (leak-free)
    fund_u = -(W * fnd).sum(1) * FUND_SCALE                              # funding collected (causal)
    raw = price_u + fund_u
    rv = pd.Series(raw).rolling(VOL_LB, min_periods=VOL_LB // 2).std().shift(1).to_numpy()
    lev = np.clip(target_vol / np.where((rv > 1e-9) & np.isfinite(rv), rv, np.inf), 0.0, LEV_CAP)
    Wl = W * lev[:, None]
    price = np.zeros(T); price[:-1] = (Wl[:-1] * rt[1:]).sum(1)
    fund = -(Wl * fnd).sum(1) * FUND_SCALE
    turn = np.abs(np.diff(Wl, axis=0, prepend=np.zeros((1, N)))).sum(1)
    cost = turn * COST
    pnl = price + fund - cost
    return pnl, dict(price=price, funding=fund, cost=cost, lev=lev)


def battery(pnl, t0):
    """per-window tail-guard/sign/persistence on the active OOS region [t0:]."""
    p = pnl[t0:]; T = len(p); wl = T // NW
    wid = np.minimum(np.arange(T) // wl, NW - 1)
    wtot = np.array([p[wid == k].sum() for k in range(NW)])
    wsh = [round(sharpe(p[wid == k]), 2) for k in range(NW)]
    worst = float(wtot.min()); rrr = float((wtot.sum() - worst) / abs(worst)) if worst < 0 else float("inf")
    pos = int((wtot > 0).sum()); med = T // 2
    tg = (worst >= 0) or (np.isfinite(rrr) and rrr > 1.0)
    sc = (pos / NW) >= 0.6
    pers = (sharpe(p[:med]) > 0 and sharpe(p[med:]) > 0)
    return dict(pooled_sharpe=round(sharpe(p), 3), pf=round(pf(p), 3), pos_windows=pos,
                ex_worst_rrr=(round(rrr, 3) if np.isfinite(rrr) else None), per_window_sharpe=wsh,
                tail_guard=bool(tg), sign_consistent=bool(sc), persistent=bool(pers))


def main():
    ret, funding, valid = load_panel()
    T, N = ret.shape
    W = carry_weights(funding, valid)
    pnl, comp = backtest(ret, funding, W)
    t0 = VOL_LB + 1
    active = pnl[t0:]

    # decomposition: where does pnl come from?
    price_sh = sharpe(comp["price"][t0:]); fund_sh = sharpe(comp["funding"][t0:])
    gross = comp["price"][t0:] + comp["funding"][t0:]
    decomp = dict(
        net_sharpe=round(sharpe(active), 3), gross_sharpe=round(sharpe(gross), 3),
        price_only_sharpe=round(price_sh, 3), funding_only_sharpe=round(fund_sh, 3),
        total_price_bp=round(float(comp["price"][t0:].sum() * 1e4), 1),
        total_funding_bp=round(float(comp["funding"][t0:].sum() * 1e4), 1),
        total_cost_bp=round(float(comp["cost"][t0:].sum() * 1e4), 1),
        realized_vol_per_bar=round(float(np.std(active)), 6),
    )

    bat = battery(pnl, t0)

    # block-bootstrap Sharpe significance (is the net edge reliably > 0?)
    rng = np.random.default_rng(7)
    bs = np.array([sharpe(active[block_boot_idx(len(active), BLK, rng)]) for _ in range(M)])
    p_le0 = float((bs <= 0).mean())

    # pollute-verify: poison the FUTURE -> pre-poison weights identical
    k = T // 2
    fp = funding.copy(); rng2 = np.random.default_rng(123)
    fp[k:] = rng2.standard_normal(fp[k:].shape) * 1e-3
    Wp = carry_weights(fp, valid)
    pre = max(0, k - REBAL - 1)
    leak_chg = float(np.abs(W[:pre] - Wp[:pre]).max())

    verdict = bool(decomp["net_sharpe"] > 0.3 and bat["tail_guard"] and bat["sign_consistent"]
                   and bat["persistent"] and p_le0 < 0.05 and leak_chg < 1e-12)
    out = dict(
        strategy="cross-sectional funding-rate carry (long low-funding, short high-funding)",
        universe=f"{N} Binance perps, 1h bars", q=Q, rebal=REBAL, cost_bp_per_fill=COST * 1e4,
        decomposition=decomp, battery=bat,
        deflation_block_boot_p_not_positive=round(p_le0, 4),
        leakfree_max_weight_change_prepoison=round(leak_chg, 12), leakfree_pass=bool(leak_chg < 1e-12),
        VERDICT="PASS" if verdict else "FAIL",
        read=("Edge is genuine carry iff funding_only_sharpe>0 dominates and net survives costs+price. "
              "If net<=0 or funding contribution is swamped by price/cost, it is an honest negative."),
    )
    print(json.dumps(out, indent=2))
    with open(os.path.join(HERE, "funding_carry_result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
