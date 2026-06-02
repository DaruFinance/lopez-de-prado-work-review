"""
E4 -- Program-scale expected-max-Sharpe vs the observed best, with REAL var_sr
and REAL effective-N from the corpus.

The False Strategy Theorem says the expected MAXIMUM Sharpe of N skill-less
trials is E[max] ~ sqrt(var_sr) * Z(N). To use it on the real program we need
the program's own (1) trial count N, (2) dispersion of trial Sharpes var_sr, and
(3) effective number of independent trials once strategies are correlated. We
measure all three from the real net-daily-PnL corpus and ask: is the observed
best Sharpe inside the skill-less expectation once N and correlation are
accounted for?

Steps (all real data):
  1. Stream every asset; for each strategy compute its full-history annualized
     Sharpe (activity-filtered). Collect the program-wide trial-Sharpe
     distribution -> empirical var_sr (per-observation) and the OBSERVED BEST.
  2. Effective-N: per asset, eigenvalue participation ratio of a sampled
     strategy-correlation matrix (lib.overfit.effective_n_trials), tiled to a
     bounded sample to cap RAM; sum gives a program effective-N.
  3. E[max Sharpe] at the program N under (a) independent N and (b) effective N,
     using the empirical var_sr; overlay the observed best.

The expected-max kernel is the verified Numba kernel from exp_max_sharpe.py
(reused). Parallelism capped at 4.
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

ANN = np.sqrt(252.0)
MIN_ACTIVE_DAYS = 120     # activity filter for a stable Sharpe estimate
CORR_SAMPLE = 1200        # strategies sampled per asset for the corr / eff-N
SEED = 20260601


def asset_stats(asset: str) -> dict | None:
    M, dates, names = cio.build_matrix(asset, use_numba=True)
    nd, ns = M.shape
    nact = (M != 0).sum(0)
    elig = np.where(nact >= MIN_ACTIVE_DAYS)[0]
    if len(elig) < 50:
        return None
    sub = M[:, elig]
    mu = sub.mean(0); sd = sub.std(0, ddof=1)
    sr = np.where(sd > 0, mu / sd, np.nan)        # per-observation Sharpe
    sr = sr[np.isfinite(sr)]
    # effective-N on a bounded sample (eigenvalue participation ratio)
    rng = np.random.default_rng(SEED)
    samp = elig if len(elig) <= CORR_SAMPLE else rng.choice(elig, CORR_SAMPLE, replace=False)
    Rs = M[:, samp]
    eff = overfit.effective_n_trials(Rs, k_max=2)   # k_max small: skip silhouette
    return dict(
        asset=asset, market=cio.market_of(asset),
        n_eligible=int(len(elig)), n_total=int(ns),
        best_sr_perobs=float(np.nanmax(sr)),
        best_sr_ann=float(np.nanmax(sr) * ANN),
        var_sr_perobs=float(np.nanvar(sr, ddof=1)),
        median_abscorr=float(eff["corr_med"]),
        eff_n_sample=float(eff["pr"]),
        eff_n_frac=float(eff["pr"] / len(samp)),     # PR / sampled-N ratio
        n_sampled=int(len(samp)),
        sr_perobs=sr,
    )


def main():
    print("=== E4: program-scale E[max Sharpe] with REAL var_sr and eff-N ===", flush=True)
    assets = cio.list_assets()
    t0 = time.time()
    results = Parallel(n_jobs=4, backend="loky", verbose=5)(
        delayed(asset_stats)(a) for a in assets)
    results = [r for r in results if r is not None]
    print(f"  ran {len(results)} assets in {time.time()-t0:.0f}s", flush=True)

    rows, all_sr = [], []
    for r in results:
        rows.append({k: v for k, v in r.items() if k != "sr_perobs"})
        all_sr.append(r["sr_perobs"])
    df = pd.DataFrame(rows).sort_values("best_sr_ann", ascending=False)
    df.to_csv(TAB / "e4_per_asset.csv", index=False)
    print(f"wrote {TAB/'e4_per_asset.csv'}", flush=True)

    sr_all = np.concatenate(all_sr)
    N_program = int(df["n_eligible"].sum())          # real program trial count
    var_sr = float(np.var(sr_all, ddof=1))           # empirical dispersion (per-obs)
    observed_best_perobs = float(np.max(sr_all))
    observed_best_ann = observed_best_perobs * ANN
    # effective-N: scale the sampled eff-N ratio up to the full eligible pool per
    # asset, then sum (independent across assets is conservative-low for the program)
    eff_n_program = float((df["eff_n_frac"] * df["n_eligible"]).sum())

    emax_indep = overfit.expected_max_sharpe(N_program, var_sr) * ANN
    emax_eff = overfit.expected_max_sharpe(max(2, int(round(eff_n_program))), var_sr) * ANN

    headline = dict(
        n_assets=len(results),
        program_N_eligible=N_program,
        program_N_total=int(df["n_total"].sum()),
        empirical_var_sr_perobs=var_sr,
        observed_best_sharpe_ann=observed_best_ann,
        observed_best_asset=str(df.iloc[0]["asset"]),
        program_eff_n=eff_n_program,
        program_eff_n_ratio=float(eff_n_program / N_program),
        median_abscorr=float(df["median_abscorr"].median()),
        e_max_sharpe_ann_independent_N=float(emax_indep),
        e_max_sharpe_ann_effective_N=float(emax_eff),
        observed_best_within_skilless_independent=bool(observed_best_ann <= emax_indep),
        observed_best_within_skilless_effective=bool(observed_best_ann <= emax_eff),
        anchor_program_best_3p21_note="prior aggregation cited best ann Sharpe ~3.21 "
            "(equity, large MA-grid corpus); E4 here recomputes the observed best on "
            "the full daily corpus (activity-filtered, full history).",
    )
    (TAB / "e4_headline.json").write_text(json.dumps(headline, indent=2, default=float))
    print(f"wrote {TAB/'e4_headline.json'}", flush=True)
    # curve table: E[max] vs N at empirical var_sr, with observed-best overlay
    Ns = np.unique(np.round(np.logspace(0, np.log10(max(10, N_program)) + 0.5, 60)).astype(int))
    Ns = Ns[Ns >= 2]
    curve = pd.DataFrame(dict(
        n_trials=Ns,
        e_max_sharpe_ann=[overfit.expected_max_sharpe(int(n), var_sr) * ANN for n in Ns]))
    curve.to_csv(TAB / "e4_emax_curve.csv", index=False)
    print(f"wrote {TAB/'e4_emax_curve.csv'}", flush=True)
    print("\n--- E4 HEADLINE ---")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
