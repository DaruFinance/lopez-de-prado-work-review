#!/usr/bin/env python3
"""Robustness battery on the clean-audit OOS ledger: apply the EDGE bar.

The clean engine is causal (every feature shift1, verified in xs_data), net-of-cost
(7bp/fill + funding) and survivorship-fixed (vol>0). It still shows pooled OOS Sharpe ~20
at short horizons. That is the same high-breadth-Sharpe class that earlier flagship runs
fell into, so it must clear the same battery those runs failed:
  1. Tail-guard: ex-worst-window reward/risk RRR > 1 (earns across windows, not one).
  2. Sign-consistency: majority of disjoint OOS windows positive.
  3. Persistence: early-half vs late-half OOS Sharpe (not a decayed pre-2022 effect).
  4. Deflation: block-bootstrap max-Sharpe null across the 10 combos (M>=2000), p<0.05.
All on the OOS phase only, per-combo. Verdict per combo + overall.
"""
import os
import sys
os.sched_setaffinity(0, range(16, 32))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

import json
import numpy as np
import pandas as pd

# Run outputs from run_clean_audit.py, which writes the clean WFO ledger under HERE/runs.
LED = os.path.join(HERE, "runs", "XS_clean_lgbm", "trades.parquet")
OUT = os.path.join(HERE, "s12_battery_result.json")
ANN = np.sqrt(252 * 24)
rng = np.random.default_rng(12)
M = 2000          # bootstrap iterations
BLK = 24          # stationary-bootstrap mean block (hourly autocorr ~ 1 day)


def sharpe(x):
    x = np.asarray(x, float)
    if x.size < 2 or x.std() < 1e-12:
        return 0.0
    return float(x.mean() / x.std() * ANN)


def pf(x):
    x = np.asarray(x, float)
    p = x[x > 0].sum(); n = -x[x < 0].sum()
    return float(p / n) if n > 1e-12 else (np.inf if p > 0 else 0.0)


def block_boot_idx(T, blk, rng):
    """Circular block bootstrap index of length T (vectorized) — preserves
    autocorrelation up to ~blk bars, which is what deflates a serially-correlated
    pnl series' max-Sharpe null."""
    nblk = int(np.ceil(T / blk))
    starts = rng.integers(0, T, nblk)
    idx = (starts[:, None] + np.arange(blk)[None, :]).ravel()[:T] % T
    return idx


def main():
    df = pd.read_parquet(LED)
    cols = {c.lower(): c for c in df.columns}
    print("ledger columns:", list(df.columns), "rows:", len(df))
    cc = cols.get("combo_id", "combo_id"); wc = cols.get("window_id", cols.get("wid", "window_id"))
    pc = cols.get("phase", "phase"); bc = cols.get("bar_idx", cols.get("bar", "bar_idx"))
    pn = cols.get("pnl_bp", cols.get("pnl", "pnl_bp"))  # ledger stores pnl in bp; Sharpe/PF are scale-invariant
    oos = df[df[pc] == 1].copy()
    print("OOS rows:", len(oos), "combos:", sorted(oos[cc].unique()))

    # per-combo pnl series indexed by bar_idx (sum across any dup bars within combo)
    combos = sorted(oos[cc].unique())
    results = {}
    series = {}
    for c in combos:
        d = oos[oos[cc] == c]
        s = d.groupby(bc)[pn].sum().sort_index()
        series[c] = s.values.astype(float)
        wins = sorted(d[wc].unique())
        wtot = np.array([float(d[d[wc] == w][pn].sum()) for w in wins])
        wshp = {int(w): sharpe(d[d[wc] == w][pn].values) for w in wins}
        worst = float(wtot.min())
        ex_worst = float(wtot.sum() - worst)
        rrr = float(ex_worst / abs(worst)) if worst < 0 else float("inf")
        pos_w = int((wtot > 0).sum()); nw = len(wins)
        # persistence by bar_idx median
        bidx = d[bc].values; med = np.median(bidx)
        early = sharpe(d[d[bc] <= med][pn].values); late = sharpe(d[d[bc] > med][pn].values)
        results[int(c)] = dict(
            pooled_sharpe=sharpe(series[c]), pf=pf(series[c]),
            n_windows=nw, pos_windows=pos_w, sign_frac=round(pos_w / max(nw, 1), 3),
            worst_window_total=worst, ex_worst_rrr=round(rrr, 3) if np.isfinite(rrr) else None,
            early_sharpe=round(early, 3), late_sharpe=round(late, 3),
            per_window_sharpe={int(w): round(v, 2) for w, v in wshp.items()},
            per_window_total=[round(float(x), 6) for x in wtot],
        )

    # ---- deflation: block-bootstrap max-Sharpe null across the `len(combos)` trials ----
    obs_best = max(results[c]["pooled_sharpe"] for c in results)
    obs_best_combo = max(results, key=lambda c: results[c]["pooled_sharpe"])
    # demean each series under H0 (no edge), bootstrap, recompute sharpe, take max across combos
    demeaned = {c: series[c] - series[c].mean() for c in combos}
    null_max = np.empty(M)
    for m in range(M):
        best = -1e18
        for c in combos:
            x = demeaned[c]
            idx = block_boot_idx(len(x), BLK, rng)
            best = max(best, sharpe(x[idx]))
        null_max[m] = best
    p_defl = float((null_max >= obs_best).mean())

    verdicts = {}
    for c in results:
        r = results[c]
        # Tail-guard fix: a book with NO losing window (worst >= 0) is the strongest
        # possible tail-guard outcome and must PASS. The earlier test treated the
        # resulting null/inf RRR as a FAIL, which wrongly failed this book on tail-guard.
        tg = (r["worst_window_total"] >= 0) or (r["ex_worst_rrr"] is not None and r["ex_worst_rrr"] > 1.0)
        sc = (r["sign_frac"] >= 0.6)
        pers = (r["late_sharpe"] > 0 and r["early_sharpe"] > 0)
        verdicts[c] = dict(tail_guard_pass=bool(tg), sign_consistent=bool(sc), persistent=bool(pers))

    best_v = verdicts[obs_best_combo]
    overall_pass = bool(best_v["tail_guard_pass"] and best_v["sign_consistent"]
                        and best_v["persistent"] and p_defl < 0.05)
    out = dict(
        ledger_rows=len(df), oos_rows=len(oos), n_combos=len(combos),
        ANN_factor=round(float(ANN), 2),
        best_combo=int(obs_best_combo), best_pooled_sharpe=round(obs_best, 3),
        deflation=dict(M=M, block=BLK, null_max_p50=round(float(np.percentile(null_max, 50)), 3),
                       null_max_p95=round(float(np.percentile(null_max, 95)), 3),
                       null_max_max=round(float(null_max.max()), 3), p_deflated=p_defl),
        per_combo=results, verdicts=verdicts,
        EDGE_BAR=dict(best_combo_verdict=best_v, p_deflated=p_defl, OVERALL_PASS=overall_pass),
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None if o == float("inf") else o)
    print(json.dumps(out["EDGE_BAR"], indent=2))
    print("best combo", obs_best_combo, "pooled SR", round(obs_best, 3),
          "| deflated p=", p_defl, "| tail-guard RRR=", results[obs_best_combo]["ex_worst_rrr"],
          "| sign_frac=", results[obs_best_combo]["sign_frac"],
          "| early/late SR=", results[obs_best_combo]["early_sharpe"], "/", results[obs_best_combo]["late_sharpe"])
    print("per-window SR (best):", results[obs_best_combo]["per_window_sharpe"])
    print("OVERALL_PASS:", overall_pass)


if __name__ == "__main__":
    main()
