"""
E3 -- Program-level Probability of Backtest Overfitting (PBO) via CSCV.

The Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu
2017) asks: when you pick the best in-sample strategy, how often does it land in
the bottom half out of sample? PBO ~ 0.5 means in-sample ranking carries no
out-of-sample information (pure overfit); PBO ~ 0 means the best in-sample
strategy reliably stays good.

We run Combinatorially-Symmetric Cross-Validation (CSCV, lib/overfit.pbo_cscv)
on the REAL net-daily-PnL matrix of each asset, and pool across the program.
Because the daily corpus has up to ~50k strategies per asset and CSCV's cost
scales with N, we evaluate a representative random sample of the activity-
filtered strategies per asset (fixed seed for reproducibility) and report the
PBO and the full logit distribution. We compare against study 00's PBO on its
2,500-strategy moving-average grid (a different, smaller corpus slice) to
reconcile the per-corpus figures honestly.

Parallelism capped at 4. The CSCV block sums are already vectorised in NumPy;
the dominant cost is the C(S, S/2) combinatorial loop, which is small (S=12 ->
924 combos) and not the bottleneck (matrix slicing is), so no extra kernel is
needed here -- but we sample strategies to bound both RAM and runtime.
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import corpus_io as cio                                      # noqa: E402
from lib import overfit                                      # noqa: E402

TAB = PROJ / "tables"; FIG = PROJ / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

SAMPLE_N = 1500       # representative strategy sample per asset (activity-filtered)
MIN_ACTIVE_DAYS = 120 # require enough non-zero days for a stable Sharpe
N_SPLITS = 12
SEED = 20260601


def run_asset(asset: str) -> dict | None:
    M, dates, names = cio.build_matrix(asset, use_numba=True)
    nd, ns = M.shape
    if nd < 200:
        return None
    nact = (M != 0).sum(0)
    elig = np.where(nact >= MIN_ACTIVE_DAYS)[0]
    if len(elig) < 50:
        return None
    rng = np.random.default_rng(SEED)
    sel = elig if len(elig) <= SAMPLE_N else rng.choice(elig, SAMPLE_N, replace=False)
    R = np.ascontiguousarray(M[:, sel])           # (T_days x N_sample) net PnL
    res = overfit.pbo_cscv(R, n_splits=N_SPLITS)
    logits = res["logits"]
    return dict(
        asset=asset, market=cio.market_of(asset),
        n_strats=ns, n_sampled=int(R.shape[1]), T=int(R.shape[0]),
        pbo=float(res["pbo"]), n_combos=int(res["n_combos"]),
        logit_mean=float(np.mean(logits)), logit_median=float(np.median(logits)),
        logits=logits,
    )


def main():
    print("=== E3: program-level PBO via CSCV ===", flush=True)
    assets = cio.list_assets()
    t0 = time.time()
    results = Parallel(n_jobs=4, backend="loky", verbose=5)(
        delayed(run_asset)(a) for a in assets)
    results = [r for r in results if r is not None]
    print(f"  ran {len(results)} assets in {time.time()-t0:.0f}s", flush=True)

    rows, all_logits, logit_by_mkt = [], [], {}
    for r in results:
        rows.append({k: v for k, v in r.items() if k != "logits"})
        all_logits.append(r["logits"])
        logit_by_mkt.setdefault(r["market"], []).append(r["logits"])
    df = pd.DataFrame(rows).sort_values("asset")
    df.to_csv(TAB / "e3_per_asset.csv", index=False)
    print(f"wrote {TAB/'e3_per_asset.csv'}", flush=True)

    pooled = np.concatenate(all_logits)
    by_mkt = {m: np.concatenate(v) for m, v in logit_by_mkt.items()}
    headline = dict(
        n_assets=len(results),
        sample_n_per_asset=SAMPLE_N,
        program_pbo_pooled_logit=float((pooled <= 0).mean()),
        program_median_pbo_per_asset=float(df["pbo"].median()),
        program_mean_pbo_per_asset=float(df["pbo"].mean()),
        pbo_by_market={m: float((v <= 0).mean()) for m, v in by_mkt.items()},
        median_pbo_by_market={m: float(df.loc[df.market == m, "pbo"].median())
                              for m in df.market.unique()},
        study00_reference=dict(crypto=0.2776, equity=0.0, fx=0.0054,
                               note="study 00 PBO on its 2,500-strategy MA grid; "
                                    "E3 here uses up to 1,500 sampled strategies "
                                    "from the full daily corpus per asset"),
    )
    (TAB / "e3_headline.json").write_text(json.dumps(headline, indent=2, default=float))
    np.savez_compressed(TAB / "e3_logits.npz",
                        pooled=pooled, **{f"mkt_{m}": v for m, v in by_mkt.items()})
    print(f"wrote {TAB/'e3_headline.json'}", flush=True)
    print("\n--- E3 HEADLINE ---")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
