"""
E2 -- The meta-strategy portfolio (LdP's actual thesis: many weakly-correlated
bets, allocated, deflated at the PORTFOLIO level).

E1 showed that no single in-sample winner clears the False Strategy Theorem null:
the best-strategy Deflated Sharpe is ~0 in every window, so a disciplined desk
that insists on one deflated survivor deploys nothing. LdP's resolution is not to
find one great strategy; it is to COMBINE many weakly-correlated bets so the
diversified sleeve carries a higher information ratio than any of its parts, and
to deflate the SLEEVE against the number of portfolio-construction trials.

Per asset, in each rolling walk-forward window we:
  1. Build a diversified shortlist from the IS data: take the top IS-Sharpe
     strategies, then greedily drop near-duplicates (|corr| > 0.7) to keep a set
     of weakly-correlated bets (this is the "many small uncorrelated bets" idea).
  2. Allocate over the shortlist with several constructors: equal weight, inverse
     variance, HRP, and NCO (denoised), all on IS covariance (no look-ahead).
  3. Deploy each sleeve in OOS and book realized net PnL (unit-risk scaled, IS
     divisor applied to OOS).
  4. Gate the BEST OOS-realized sleeve's POOLED track against a portfolio-level
     False Strategy Theorem null whose trial count is the number of construction
     choices searched (allocators x shortlist sizes), with var_sr from those
     trials' Sharpes. Report whether the diversified sleeve clears DSR > 0.95.

Also reports the cross-strategy correlation structure (median |corr| inside the
shortlist) and the effective number of independent bets.

Reuses study 13's HRP / NCO / denoise machinery. Parallelism capped at 4.
"""
from __future__ import annotations
import os, sys, time, json, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as ss
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import corpus_io as cio                                      # noqa: E402
from lib import overfit                                      # noqa: E402
_p13 = ROOT / "projects/13_portfolio_construction/scripts/run_portfolio.py"
_spec = importlib.util.spec_from_file_location("p13_portfolio", _p13)
p13 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p13)

TAB = PROJ / "tables"; FIG = PROJ / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

ANN = np.sqrt(252.0)
IS_DAYS, OOS_DAYS, STEP = 252, 63, 63
MIN_IS_OBS = 60
TOPK = 600           # IS-Sharpe shortlist to draw weakly-correlated bets from
CORR_DROP = 0.70     # drop near-duplicate bets above this |corr|
SHORT_SIZES = (8, 16, 32)   # shortlist sizes tried (part of the trial count)
DSR_GATE = 0.95


def _diversified_shortlist(is_block, sr_is, k_top, max_size):
    """Top-IS-Sharpe, then greedily keep weakly-correlated bets up to max_size."""
    order = np.argsort(sr_is)[::-1][:k_top]
    sub = is_block[:, order]
    # correlation among the top-k
    C = np.corrcoef(sub.T)
    C = np.nan_to_num(C, nan=0.0)
    keep = [0]
    for i in range(1, len(order)):
        if all(abs(C[i, j]) <= CORR_DROP for j in keep):
            keep.append(i)
        if len(keep) >= max_size:
            break
    return order[keep]


def _allocators(cov):
    out = {}
    n = cov.shape[0]
    out["equal"] = np.full(n, 1.0 / n)
    iv = 1.0 / np.clip(np.diag(cov), 1e-12, None); out["invvar"] = iv / iv.sum()
    try:
        w = p13.w_hrp(cov, use_numba=True)
        out["hrp"] = w / w.sum() if (np.all(np.isfinite(w)) and w.sum() > 0) else out["equal"]
    except Exception:
        out["hrp"] = out["equal"]
    try:
        corr, std = p13._corr_from_cov(cov)
        q = cov.shape[0] / max(1, IS_DAYS)
        dn = p13.denoise_corr(corr, q)
        covd = p13._cov_from_corr(dn, std)
        w = p13.w_nco(covd)
        out["nco"] = w / w.sum() if (np.all(np.isfinite(w)) and w.sum() > 0) else out["equal"]
    except Exception:
        out["nco"] = out["equal"]
    return out


def run_asset(asset: str) -> dict | None:
    M, dates, names = cio.build_matrix(asset, use_numba=True)
    nd, ns = M.shape
    if nd < IS_DAYS + OOS_DAYS + STEP:
        return None
    starts = list(range(0, nd - IS_DAYS - OOS_DAYS + 1, STEP))
    # realized OOS daily per (allocator, shortlist-size) construction trial
    streams = {}            # key -> list of OOS daily arrays
    med_abscorr, eff_n = [], []
    for st in starts:
        is_sl = M[st:st + IS_DAYS]; oos_sl = M[st + IS_DAYS:st + IS_DAYS + OOS_DAYS]
        nact = (is_sl != 0).sum(0); pool = nact >= MIN_IS_OBS
        if pool.sum() < TOPK:
            continue
        isa = np.ascontiguousarray(is_sl[:, pool]); oosa = np.ascontiguousarray(oos_sl[:, pool])
        sd = isa.std(0, ddof=1); sd = np.where(sd > 0, sd, 1.0)
        is_sc = isa / sd; oos_sc = oosa / sd
        sr_is = np.nan_to_num(isa.mean(0) / isa.std(0, ddof=1))
        for sz in SHORT_SIZES:
            sel = _diversified_shortlist(is_sc, sr_is, TOPK, sz)
            if len(sel) < 3:
                continue
            sub_is = is_sc[:, sel]; sub_oos = oos_sc[:, sel]
            cov = np.atleast_2d(np.cov(sub_is.T))
            # correlation diagnostics (largest shortlist only, to avoid double count)
            if sz == SHORT_SIZES[-1]:
                Csel = np.nan_to_num(np.corrcoef(sub_is.T))
                iu = np.triu_indices(len(sel), 1)
                med_abscorr.append(float(np.median(np.abs(Csel[iu]))) if len(iu[0]) else np.nan)
                lam = np.clip(np.linalg.eigvalsh(Csel), 0, None)
                eff_n.append(float(lam.sum() ** 2 / np.square(lam).sum()) if np.square(lam).sum() > 0 else 1.0)
            for name, w in _allocators(cov).items():
                streams.setdefault(f"{name}_{sz}", []).append(sub_oos @ w)
    if not streams:
        return None
    # pooled realized OOS track per construction trial
    trial_tracks = {k: np.concatenate(v) for k, v in streams.items()}
    trial_sr = {k: overfit.sharpe(t) for k, t in trial_tracks.items()}
    best_key = max(trial_sr, key=trial_sr.get)
    best_track = trial_tracks[best_key]
    # portfolio-level deflation: N = number of construction trials searched
    sr_vec = np.array(list(trial_sr.values()))
    N_trials = len(sr_vec)
    var_sr = float(np.var(sr_vec, ddof=1)) if N_trials > 1 else 0.0
    sr0 = overfit.expected_max_sharpe(N_trials, var_sr)
    bt = best_track[np.isfinite(best_track)]
    dsr = overfit.prob_sharpe_ratio(overfit.sharpe(bt), len(bt),
                                    float(ss.skew(bt)), float(ss.kurtosis(bt, fisher=False)),
                                    sr_benchmark=sr0)
    # also a fixed-policy HRP@32 sleeve (no per-asset trial selection) for an
    # honest, selection-free realized number
    hrp_key = f"hrp_{SHORT_SIZES[-1]}"
    hrp_track = trial_tracks.get(hrp_key, best_track)
    return dict(
        asset=asset, market=cio.market_of(asset), n_strats=ns,
        n_windows=len(med_abscorr),
        median_abscorr_shortlist=float(np.nanmedian(med_abscorr)) if med_abscorr else np.nan,
        eff_n_bets=float(np.nanmedian(eff_n)) if eff_n else np.nan,
        best_trial=best_key,
        best_sleeve_oos_sharpe_ann=float(overfit.sharpe(best_track) * ANN),
        hrp32_oos_sharpe_ann=float(overfit.sharpe(hrp_track) * ANN),
        n_construction_trials=N_trials,
        portfolio_sr0_ann=float(sr0 * ANN),
        portfolio_dsr=float(dsr) if dsr is not None and np.isfinite(dsr) else np.nan,
        clears_dsr95=bool(dsr is not None and np.isfinite(dsr) and dsr >= DSR_GATE),
        best_track=best_track, hrp_track=hrp_track,
    )


def main():
    print("=== E2: the meta-strategy portfolio (diversified, deflated) ===", flush=True)
    assets = cio.list_assets()
    t0 = time.time()
    results = Parallel(n_jobs=4, backend="loky", verbose=5)(
        delayed(run_asset)(a) for a in assets)
    results = [r for r in results if r is not None]
    print(f"  ran {len(results)} assets in {time.time()-t0:.0f}s", flush=True)

    rows = []
    pooled_best, pooled_hrp = [], []
    for r in results:
        rows.append({k: v for k, v in r.items() if k not in ("best_track", "hrp_track")})
        pooled_best.append(r["best_track"]); pooled_hrp.append(r["hrp_track"])
    df = pd.DataFrame(rows).sort_values("asset")
    df.to_csv(TAB / "e2_per_asset.csv", index=False)
    print(f"wrote {TAB/'e2_per_asset.csv'}", flush=True)

    best_all = np.concatenate(pooled_best); hrp_all = np.concatenate(pooled_hrp)
    # program-level meta-portfolio deflation: pool the per-asset HRP sleeves
    # (selection-free policy) and deflate against the per-asset trial search count
    n_program_trials = int(df["n_construction_trials"].sum())
    sr_program = overfit.sharpe(hrp_all)
    var_sr_prog = float(df["best_sleeve_oos_sharpe_ann"].var() / (ANN ** 2))
    sr0_prog = overfit.expected_max_sharpe(max(2, n_program_trials), var_sr_prog)
    ha = hrp_all[np.isfinite(hrp_all)]
    dsr_prog = overfit.prob_sharpe_ratio(sr_program, len(ha), float(ss.skew(ha)),
                                          float(ss.kurtosis(ha, fisher=False)),
                                          sr_benchmark=sr0_prog)
    headline = dict(
        n_assets=len(results),
        median_abscorr_shortlist=float(df["median_abscorr_shortlist"].median()),
        median_eff_n_bets=float(df["eff_n_bets"].median()),
        n_assets_clearing_dsr95=int(df["clears_dsr95"].sum()),
        median_portfolio_dsr=float(df["portfolio_dsr"].median()),
        pooled_best_sleeve_oos_sharpe_ann=float(overfit.sharpe(best_all) * ANN),
        pooled_hrp32_oos_sharpe_ann=float(overfit.sharpe(hrp_all) * ANN),
        program_meta_portfolio_oos_sharpe_ann=float(sr_program * ANN),
        program_meta_sr0_ann=float(sr0_prog * ANN),
        program_meta_dsr=float(dsr_prog) if dsr_prog is not None and np.isfinite(dsr_prog) else np.nan,
        program_meta_clears_dsr95=bool(dsr_prog is not None and np.isfinite(dsr_prog) and dsr_prog >= DSR_GATE),
        n_program_construction_trials=n_program_trials,
    )
    (TAB / "e2_headline.json").write_text(json.dumps(headline, indent=2, default=float))
    print(f"wrote {TAB/'e2_headline.json'}", flush=True)
    np.savez_compressed(TAB / "e2_pooled_oos.npz", best=best_all, hrp=hrp_all)
    print("\n--- E2 HEADLINE ---")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
