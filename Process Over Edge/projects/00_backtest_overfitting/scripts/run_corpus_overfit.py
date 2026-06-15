#!/usr/bin/env python3
"""
Project 0 at SCALE, backtest-overfitting / DSR / PBO / effective-N over a real,
DIVERSE strategy corpus: ~5,000 strategies per pair x 10 pairs = ~50,000 per
market (diversity over concentration, per spec; >50k is diminishing returns).

Crypto uses the existing per-strategy daily PnL (strategy-pnl-daily-rs output);
strategies are already costed (fee 0.05 / slip 0.03 / funding 0.01 in the specs).

RAM-safe design (WSL ~46 GB; tile, never materialize N x N):
  - DSR / False Strategy Theorem: streamed per-strategy Sharpe distribution.
  - PBO via CSCV: per-strategy block sums (T blocks), combinatorial recombination.
  - effective-N: eigenvalue participation ratio computed from the T x T Gram
    matrix of standardized returns (the N x N correlation has rank <= T, so its
    nonzero eigenvalues equal those of the T x T matrix -> exact, cheap).
"""
import sys, glob, os, warnings
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
from scipy import stats as ss

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.exists(_os.path.join(_d, "config.py")):
    _d = _os.path.dirname(_d)
REPO_ROOT = _d
_sys.path.insert(0, REPO_ROOT)
import config as cfg
from config import LIB as _LIB
_sys.path.insert(0, _LIB)
import overfit as OF
import style as ST
warnings.filterwarnings("ignore"); ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = cfg.PNL_DAILY
PER_PAIR = 2500               # diverse: ~2500 x ~20 pairs = ~50k (>50k = diminishing returns)
MAX_PAIRS = 20
SEED = 7
MARKET = os.environ.get("LDP_MARKET", "crypto")  # crypto | equity | fx
ANN = np.sqrt(252)            # daily PnL -> annualised Sharpe for display


def discover_assets(market):
    allassets = [os.path.basename(p).replace("asset=", "")
                 for p in sorted(glob.glob(f"{BASE}/asset=*"))]
    def cls(a):
        al = a.lower()
        if al.endswith("_equity"): return "equity"
        if al.endswith("_fx"): return "fx"
        if "forex" in al: return "legacy"      # broken-axis legacy bundles -> skip
        return "crypto"
    return [a for a in allassets if cls(a) == market][:MAX_PAIRS]


CRYPTO = discover_assets(MARKET)   # assets for the selected market


def load_pair_matrix(asset, n=PER_PAIR, seed=SEED):
    """Return (dates, returns matrix T x n, strategy ids) for a sampled subset."""
    fs = glob.glob(f"{BASE}/asset={asset}/*.parquet")
    tbl = ds.dataset(fs).to_table(columns=["strategy_name", "date", "pnl_sum"])
    df = tbl.to_pandas()
    uniq = df["strategy_name"].unique()
    rng = np.random.default_rng(seed)
    pick = uniq if len(uniq) <= n else rng.choice(uniq, n, replace=False)
    df = df[df["strategy_name"].isin(set(pick))]
    # pivot to date x strategy, fill non-trading days with 0 (flat)
    m = df.pivot_table(index="date", columns="strategy_name", values="pnl_sum",
                       aggfunc="sum", fill_value=0.0).sort_index()
    return m


def main():
    print(f"Loading {len(CRYPTO)} {MARKET} pairs x {PER_PAIR} strategies ...")
    sharpes = []          # per-strategy Sharpe (full series) across the corpus
    per_pair_rows = []
    common = None
    mats = {}
    for a in CRYPTO:
        m = load_pair_matrix(a)
        ret = m.to_numpy(np.float64)
        sd = ret.std(0, ddof=1); mu = ret.mean(0)
        sr = np.where(sd > 0, mu / sd, 0.0)
        sharpes.append(sr)
        # per-pair PBO + effective-N
        pbo = OF.pbo_cscv(ret, 12)["pbo"]
        effN = effective_n_gram(ret)
        per_pair_rows.append(dict(pair=a.split("_")[0], n=ret.shape[1], T=ret.shape[0],
                                  best_sr_ann=sr.max()*ANN, pbo=pbo, eff_n=effN))
        mats[a] = m
        common = set(m.index) if common is None else (common & set(m.index))
        print(f"  {a.split('_')[0]}: {ret.shape[1]} strat x {ret.shape[0]} days  "
              f"bestSR(ann)={sr.max()*ANN:.2f}  PBO={pbo:.2f}  effN={effN}")
        del ret

    sr_all = np.concatenate(sharpes)
    N = sr_all.size
    var_sr = sr_all.var(ddof=1)
    # use a representative T for the corpus (median pair length) for the per-period basis
    Tmed = int(np.median([r["T"] for r in per_pair_rows]))
    sr0 = OF.expected_max_sharpe(N, var_sr)                 # False Strategy Theorem null
    best = sr_all.max()
    # DSR of the global best strategy across the whole corpus
    dsr = OF.prob_sharpe_ratio(best, Tmed, 0.0, 3.0, sr_benchmark=sr0)

    # pooled effective-N on the common-date window across all 10 pairs
    common = sorted(common)
    pooled = np.hstack([mats[a].reindex(common).to_numpy(np.float64) for a in CRYPTO])
    effN_pool = effective_n_gram(pooled)
    print(f"\n=== {MARKET.upper()} CORPUS (N={N: } strategies, {len(CRYPTO)} pairs) ===")
    print(f"  best Sharpe (ann)         : {best*ANN:.3f}")
    print(f"  E[max] null (ann, N={N: }) : {sr0*ANN:.3f}   <- False Strategy Theorem")
    print(f"  DSR of corpus best        : {dsr:.3f}   (>0.95 = significant)")
    print(f"  median per-pair PBO       : {np.median([r['pbo'] for r in per_pair_rows]):.3f}")
    print(f"  effective-N (pooled, common {len(common)}d): {effN_pool}  of {N: } nominal")

    dfp = pd.DataFrame(per_pair_rows)
    dfp.to_csv(f"{PROJ}/tables/corpus_per_pair_{MARKET}.csv", index=False)
    summ = dict(market=MARKET, n_strategies=int(N), n_pairs=len(CRYPTO),
                best_sr_ann=float(best*ANN), sr0_ann=float(sr0*ANN), dsr=float(dsr),
                median_pbo=float(np.median([r['pbo'] for r in per_pair_rows])),
                eff_n_pooled=int(effN_pool))
    pd.DataFrame([summ]).to_csv(f"{PROJ}/tables/corpus_summary_{MARKET}.csv", index=False)
    with open(f"{PROJ}/tables/corpus_summary_{MARKET}.md", "w") as fh:
        fh.write(f"# Backtest overfitting on a real {N: }-strategy {MARKET} corpus "
                 f"({len(CRYPTO)} pairs x {PER_PAIR})\n\n")
        fh.write(pd.DataFrame([summ]).round(3).to_markdown(index=False))
        fh.write("\n\n## Per pair\n\n"+dfp.round(3).to_markdown(index=False))
    make_figs(sr_all, sr0, best, dfp, N)
    print("\nCorpus overfit study done.")


def effective_n_gram(ret):
    """Participation ratio of the strategy-return correlation matrix via the
    T x T Gram trick (corr has rank <= T). Exact, no N x N materialization."""
    T, N = ret.shape
    Z = (ret - ret.mean(0)) / np.where(ret.std(0, ddof=1) > 0, ret.std(0, ddof=1), 1.0)
    Z = np.nan_to_num(Z)
    G = (Z @ Z.T) / (N)                 # T x T ; shares nonzero spectrum with corr
    lam = np.clip(np.linalg.eigvalsh(G), 0, None)
    s = lam.sum()
    return int(round(s*s/np.square(lam).sum())) if (s > 0 and np.square(lam).sum() > 0) else 1


def make_figs(sr_all, sr0, best, dfp, N):
    d = f"{PROJ}/figures"
    ann = ANN
    # Fig: corpus Sharpe distribution vs the False-Strategy-Theorem null bar
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(sr_all*ann, bins=120, color=ST.PALETTE["dollar"], alpha=0.8)
    ax.axvline(sr0*ann, color=ST.PALETTE["accent"], lw=2, ls="--",
               label=f"E[max] under null (N={N: }) = {sr0*ann:.2f}")
    ax.axvline(best*ann, color="black", lw=2, label=f"corpus best = {best*ann:.2f}")
    ax.set_xlabel("Annualised Sharpe"); ax.set_ylabel("strategies")
    ax.set_title(f"Real {N: }-strategy {MARKET} corpus: best Sharpe vs the multiple-testing null")
    ax.legend()
    fig.savefig(f"{d}/fig5_corpus_sharpe_vs_null_{MARKET}.png"); plt.close(fig)
    # Fig: per-pair PBO + effective-N
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(dfp["pair"], dfp["pbo"], color=ST.PALETTE["dollar"]); axes[0].axhline(0.5, ls="--", color=ST.PALETTE["accent"])
    axes[0].set_title("Probability of Backtest Overfitting, per pair"); axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(dfp["pair"], dfp["eff_n"], color=ST.PALETTE["volume"])
    axes[1].set_title(f"Effective independent trials per pair (of {PER_PAIR})"); axes[1].tick_params(axis="x", rotation=45)
    fig.savefig(f"{d}/fig6_corpus_pbo_effn_{MARKET}.png"); plt.close(fig)
    print("corpus figures written")


if __name__ == "__main__":
    main()
