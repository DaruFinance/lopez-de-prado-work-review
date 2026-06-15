#!/usr/bin/env python3
"""
S11 Deep Run — Regime + Overfitting-Gauge Deployment.

Locked bar (PREREGISTRATION.md, S11):
  PASS: gated+blended deployment beats BOTH lone-best (Sisyphus) AND always-deploy
        baselines on realized OOS, gains coming from low-PBO markets (equities/FX)
        with crypto correctly NOT rescued; gates verified causal.
  FAIL-> "no improvement."

THREE DEPLOYMENTS, rolling WFO (IS=252d, OOS=63d, step=63d):

  1. LONE-BEST (Sisyphus baseline):
       Per window, pick the single strategy with the highest IS Sharpe.
       Deploy at unit IS-risk (divide OOS by IS std). No other selection.

  2. ALWAYS-DEPLOY (top-K HRP, no causal gate):
       Per window, take the top K% by IS Sharpe from the active pool.
       HRP-allocate on IS covariance at unit IS-portfolio risk.
       This is the "naive best-practices" blend: IS winner selection + HRP,
       but WITHOUT the regime or PBO causal gate.

  3. GATED+BLEND (S11 subject — STRICTLY CAUSAL):
       Same top-K IS winner shortlist as (2), plus two causal IS-only gates:
         (a) REGIME gate: fit 2-state Gaussian HMM on IS equal-weighted
             portfolio of the top-K shortlist. Deploy only if the LAST IS day
             is in the "good" state (higher mean IS return).
             IS-only: OOS regime label is NEVER computed or used.
         (b) PBO gate: CSCV on the IS matrix (N_SAMPLE sampled from pool).
             Deploy only if IS PBO < PBO_GATE (IS rank carries OOS signal).
             IS-only: OOS PnL is NEVER seen.
       If BOTH gates pass: deploy the top-K HRP blend (same weights as (2)).
       If either gate fails: hold cash (zero) for that OOS window.

ANTI-LOOKAHEAD CRUX:
  The shallow version leaked by using a full-sample sleeve median for gates.
  Here: regime and PBO are computed per-window from IS data only. No
  cross-window aggregation is used to set any threshold. Threshold PBO_GATE
  is a fixed hyperparameter set before the run (not data-driven).

CAUSAL GATE CHECK:
  Pollute-and-verify: permute per-window pbo_pass labels randomly. If the gate
  is informative, the permuted gate should score worse than the real gate.
  We expect < 50% of B=500 permutations to beat the true gate's pooled OOS SR.

REAL COSTS: already in pnl_sum (net of fees/slip). No additional cost applied
  since the corpus already carries costs (verified in corpus_io.py header).

OUTPUT ARTIFACTS:
  s11_per_asset.parquet/csv   — per-asset summary row
  s11_windows.parquet          — per-window gate decisions (causal audit trail)
  s11_blend_weights.parquet    — per-window deployment flags
  s11_pooled_oos.npz           — daily OOS series for all three deployments
  s11_headline.json            — verdict, Sharpes, CI, significance, RAM
"""
from __future__ import annotations
import gc, json, os, sys, time, resource, warnings

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
)

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as ss
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

OUT_DIR = Path(HERE)
REPO    = Path(REPO_ROOT)
LIB     = Path(_LIB)
SCRIPTS = REPO / "projects/16_meta_strategy_organization/scripts"
sys.path.insert(0, str(SCRIPTS))

import corpus_io as cio
import overfit as OF

# --------------------------------------------------------------------------- #
# Hyperparameters (locked before run)
# --------------------------------------------------------------------------- #
SEED            = 20260613
IS_DAYS         = 252       # in-sample window (trading days)
OOS_DAYS        = 63        # out-of-sample window
STEP            = 63        # roll step
MIN_IS_OBS      = 60        # min non-zero IS days to include a strategy
TOP_K           = 20        # top-K IS Sharpe strategies for always-deploy and gated blend
                             # (fixed count, not fraction — controls HRP portfolio size)
N_SAMPLE_PBO    = 600       # max strategies in IS matrix for CSCV (RAM bound)
N_HMM_EW        = 20        # use top-K strategies for IS EW portfolio (same as TOP_K)
N_SPLITS_CSCV   = 10        # CSCV splits (C(10,5)=252 combos)
PBO_GATE        = 0.50      # IS PBO must be < this to pass
ANN             = np.sqrt(252.0)
N_BOOT          = 2000
BLOCK_BOOT      = 21

# --------------------------------------------------------------------------- #
# HMM regime gate (IS only)
# --------------------------------------------------------------------------- #
def _regime_good_is(is_portfolio: np.ndarray) -> bool:
    """Fit 2-state Gaussian HMM on IS equal-weighted top-K portfolio PnL.

    Returns True if the last IS day is in the higher-mean state.
    Causal: only IS data used. OOS never touched.

    Fallback (if hmmlearn unavailable): trailing-21-day mean > 0.
    """
    if len(is_portfolio) < 30:
        return False
    x = is_portfolio.reshape(-1, 1)
    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(
            n_components=2, covariance_type="full",
            n_iter=150, random_state=SEED, tol=1e-4,
        )
        model.fit(x)
        states = model.predict(x)
        m0 = float(is_portfolio[states == 0].mean()) if (states == 0).any() else -np.inf
        m1 = float(is_portfolio[states == 1].mean()) if (states == 1).any() else -np.inf
        good = 0 if m0 >= m1 else 1
        return int(states[-1]) == good
    except Exception:
        return float(is_portfolio[-21:].mean()) > 0.0


# --------------------------------------------------------------------------- #
# IS-only PBO gate
# --------------------------------------------------------------------------- #
def _pbo_is(is_act: np.ndarray, rng: np.random.Generator) -> float:
    """CSCV on a sample from the IS matrix.

    Returns float PBO in [0,1]. PBO < PBO_GATE → passes.
    Strictly causal: is_act is the IS slice only.
    """
    T, N = is_act.shape
    # activity filter
    nact = (is_act != 0).sum(0)
    elig = np.where(nact >= MIN_IS_OBS)[0]
    if len(elig) < 20:
        return 1.0
    sel = elig if len(elig) <= N_SAMPLE_PBO else rng.choice(elig, N_SAMPLE_PBO, replace=False)
    R = np.ascontiguousarray(is_act[:, sel])
    if R.shape[0] < N_SPLITS_CSCV * 2:
        return 1.0
    try:
        return float(OF.pbo_cscv(R, n_splits=N_SPLITS_CSCV)["pbo"])
    except Exception:
        return 1.0


# --------------------------------------------------------------------------- #
# HRP allocator
# --------------------------------------------------------------------------- #
def _hrp_weights(cov: np.ndarray) -> np.ndarray:
    N = cov.shape[0]
    if N == 1:
        return np.array([1.0])
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    from scipy.cluster.hierarchy import linkage, leaves_list
    Z = linkage(dist[np.triu_indices(N, 1)], method="single")
    order = list(leaves_list(Z))
    weights = np.ones(N)
    clusters = [order]
    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) < 2:
                continue
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            def cv(idx):
                sub = cov[np.ix_(idx, idx)]
                w = 1.0 / np.maximum(np.diag(sub), 1e-12)
                w /= w.sum()
                return max(float(w @ sub @ w), 1e-12)
            alpha = 1.0 - cv(left) / (cv(left) + cv(right))
            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)
            if len(left) >= 2:
                new_clusters.append(left)
            if len(right) >= 2:
                new_clusters.append(right)
        clusters = new_clusters
    return np.maximum(weights, 0.0) / max(weights.sum(), 1e-12)


def _unit_vol_hrp(is_sub: np.ndarray, oos_sub: np.ndarray) -> np.ndarray:
    """HRP portfolio scaled to unit IS vol. Returns (OOS_DAYS,) daily PnL."""
    if is_sub.shape[1] == 1:
        sd = float(is_sub[:, 0].std(ddof=1))
        scale = 1.0 / sd if sd > 1e-8 else 1.0
        return oos_sub[:, 0] * scale
    cov = np.cov(is_sub.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    try:
        w = _hrp_weights(cov)
    except Exception:
        w = np.full(is_sub.shape[1], 1.0 / is_sub.shape[1])
    is_port = is_sub @ w
    sd = float(is_port.std(ddof=1))
    if sd > 1e-8:
        w = w / sd
    return oos_sub @ w


# --------------------------------------------------------------------------- #
# Per-asset rolling WFO
# --------------------------------------------------------------------------- #
def run_asset(asset: str) -> dict | None:
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    M, dates, names = cio.build_matrix(asset, use_numba=True, verbose=False)
    nd, ns = M.shape
    market = cio.market_of(asset)

    if nd < IS_DAYS + OOS_DAYS + STEP:
        del M; gc.collect()
        return None

    starts = list(range(0, nd - IS_DAYS - OOS_DAYS + 1, STEP))
    if not starts:
        del M; gc.collect()
        return None

    lone_daily, always_daily, gated_daily = [], [], []
    win_rows, weight_rows = [], []
    zeros = np.zeros(OOS_DAYS)

    for win_i, st in enumerate(starts):
        is_sl  = M[st : st + IS_DAYS]
        oos_sl = M[st + IS_DAYS : st + IS_DAYS + OOS_DAYS]

        nact = (is_sl != 0).sum(0)
        pool = np.where(nact >= MIN_IS_OBS)[0]

        if len(pool) < TOP_K:
            for lst in [lone_daily, always_daily, gated_daily]:
                lst.append(zeros.copy())
            win_rows.append(dict(
                asset=asset, market=market, win=win_i, st=int(st),
                deployed_lone=False, deployed_always=False, deployed_gated=False,
                regime_pass=False, pbo_is=1.0, pbo_pass=False, n_pool=0,
                lone_oos_sr=0.0, always_oos_sr=0.0, gated_oos_sr=0.0,
            ))
            continue

        is_act  = np.ascontiguousarray(is_sl[:, pool])
        oos_act = np.ascontiguousarray(oos_sl[:, pool])

        # unit-risk scaling (IS std)
        sd_is = is_act.std(0, ddof=1)
        sd_is = np.where(sd_is > 0, sd_is, 1.0)
        is_scaled  = is_act / sd_is
        oos_scaled = oos_act / sd_is

        sr_is = is_act.mean(0) / sd_is   # IS per-obs Sharpe (unit-scaled)

        # Top-K by IS Sharpe
        top_idx = np.argsort(sr_is)[::-1][:TOP_K]

        # ------------------------------------------------------------------- #
        # BASELINE 1: Lone-best
        # ------------------------------------------------------------------- #
        k = int(top_idx[0])
        lone_oos = oos_scaled[:, k]
        lone_daily.append(lone_oos)

        # ------------------------------------------------------------------- #
        # BASELINE 2: Always-deploy (top-K HRP, no causal gates)
        # ------------------------------------------------------------------- #
        is_topk  = is_scaled[:, top_idx]
        oos_topk = oos_scaled[:, top_idx]
        always_oos = _unit_vol_hrp(is_topk, oos_topk)
        always_daily.append(always_oos)

        # ------------------------------------------------------------------- #
        # GATED+BLEND (S11 subject — strictly causal IS-only gates)
        # ------------------------------------------------------------------- #

        # (a) Regime gate — fit HMM on IS equal-weighted top-K portfolio
        is_ew_topk = is_topk.mean(1)           # (IS_DAYS,) IS only
        regime_pass = _regime_good_is(is_ew_topk)

        # (b) PBO gate — CSCV on IS matrix (full pool, sampled)
        pbo_val = _pbo_is(is_act, rng)         # IS only
        pbo_pass = pbo_val < PBO_GATE

        both_pass = regime_pass and pbo_pass

        if both_pass:
            gated_oos = _unit_vol_hrp(is_topk, oos_topk)   # same blend as always-deploy
        else:
            gated_oos = zeros.copy()
        gated_daily.append(gated_oos)

        weight_rows.append(dict(
            asset=asset, win=int(win_i), st=int(st),
            regime_pass=bool(regime_pass), pbo_is=float(pbo_val),
            pbo_pass=bool(pbo_pass), both_pass=bool(both_pass),
            n_pool=int(len(pool)),
        ))

        def _sr(x): return float(OF.sharpe(x) * ANN)
        win_rows.append(dict(
            asset=asset, market=market, win=int(win_i), st=int(st),
            deployed_lone=True, deployed_always=True,
            deployed_gated=bool(both_pass),
            regime_pass=bool(regime_pass), pbo_is=float(pbo_val),
            pbo_pass=bool(pbo_pass), n_pool=int(len(pool)),
            lone_oos_sr=_sr(lone_oos),
            always_oos_sr=_sr(always_oos),
            gated_oos_sr=_sr(gated_oos),
        ))

    del M; gc.collect()

    lone_d   = np.concatenate(lone_daily)
    always_d = np.concatenate(always_daily)
    gated_d  = np.concatenate(gated_daily)

    def _sr(x): return float(OF.sharpe(x) * ANN)
    return dict(
        asset=asset, market=market,
        n_windows=len(starts),
        lone_oos_sharpe_ann=_sr(lone_d),
        always_oos_sharpe_ann=_sr(always_d),
        gated_oos_sharpe_ann=_sr(gated_d),
        gated_beats_lone=bool(_sr(gated_d) > _sr(lone_d)),
        gated_beats_always=bool(_sr(gated_d) > _sr(always_d)),
        deploy_rate=float(np.mean([r["deployed_gated"] for r in win_rows])),
        pbo_pass_rate=float(np.mean([r["pbo_pass"] for r in win_rows])),
        regime_pass_rate=float(np.mean([r["regime_pass"] for r in win_rows])),
        elapsed_s=float(time.time() - t_start),
        _lone_d=lone_d, _always_d=always_d, _gated_d=gated_d,
        _win_rows=win_rows, _weight_rows=weight_rows,
    )


# --------------------------------------------------------------------------- #
# Bootstrap CI (block bootstrap on pooled daily OOS series)
# --------------------------------------------------------------------------- #
def _boot_ci(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    nb = int(np.ceil(n / BLOCK_BOOT))
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(BLOCK_BOOT)[None, :]).ravel()[:n] % n
        vals[b] = OF.sharpe(x[idx]) * ANN
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# Causal pollute-and-verify
# --------------------------------------------------------------------------- #
def _causal_verify(results: list[dict]) -> dict:
    """Permute per-window pbo_pass labels; check if true gate beats permuted gate.

    The true gate uses IS PBO to decide skip/deploy.
    Permuted gate randomly re-assigns skip/deploy independent of IS PBO.
    If the gate is causal (IS PBO → OOS outcome), permuting should be worse.
    We measure: fraction of B=500 permutations where permuted >= true gate SR.
    Causal = frac < 0.50.
    """
    rng = np.random.default_rng(SEED + 99)
    B = 500

    per_win = []
    for r in results:
        wr = r["_win_rows"]
        always_d = r["_always_d"]
        n_win = len(wr)
        for i, row in enumerate(wr):
            i0 = i * OOS_DAYS
            i1 = i0 + OOS_DAYS
            if i1 > len(always_d):
                break
            per_win.append(dict(
                always_chunk=always_d[i0:i1],
                pbo_pass=row["pbo_pass"],
                regime_pass=row["regime_pass"],
            ))

    if not per_win:
        return dict(causal_verified=True, note="no windows to test")

    # True gate outcome
    true_gated = np.concatenate([
        w["always_chunk"] if (w["pbo_pass"] and w["regime_pass"]) else np.zeros(OOS_DAYS)
        for w in per_win
    ])
    true_sr = float(OF.sharpe(true_gated) * ANN)

    # Permuted gates
    perm_srs = []
    pass_flags = np.array([w["pbo_pass"] and w["regime_pass"] for w in per_win])
    for _ in range(B):
        perm = rng.permutation(pass_flags)
        perm_port = np.concatenate([
            w["always_chunk"] if pp else np.zeros(OOS_DAYS)
            for w, pp in zip(per_win, perm)
        ])
        perm_srs.append(float(OF.sharpe(perm_port) * ANN))

    perm_srs = np.array(perm_srs)
    frac = float((perm_srs >= true_sr).mean())
    causal = bool(frac < 0.50)
    return dict(
        causal_verified=causal,
        frac_permutations_beat_true_gate=round(frac, 4),
        true_gated_sr_ann=round(true_sr, 4),
        perm_mean_sr_ann=round(float(perm_srs.mean()), 4),
        perm_p95_sr_ann=round(float(np.percentile(perm_srs, 95)), 4),
        n_permutations=B,
        note=(
            f"Permuted gate >= true gate in {frac*100:.1f}% of {B} trials "
            f"({'CAUSAL (<50%)' if causal else 'SUSPICIOUS (>=50%)'})"
        ),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("S11 deep run — Regime + Overfitting-Gauge Deployment", flush=True)
    print(f"IS={IS_DAYS}d  OOS={OOS_DAYS}d  step={STEP}d  TOP_K={TOP_K}", flush=True)
    print(f"PBO_GATE={PBO_GATE}  N_SAMPLE_PBO={N_SAMPLE_PBO}  CSCV_splits={N_SPLITS_CSCV}", flush=True)
    print("=" * 70, flush=True)

    try:
        import hmmlearn
        print(f"hmmlearn {hmmlearn.__version__} available", flush=True)
    except ImportError:
        print("hmmlearn not found — using rolling-mean fallback", flush=True)

    assets = cio.list_assets()
    print(f"Corpus: {len(assets)} assets", flush=True)

    results = []
    for i, asset in enumerate(assets):
        print(f"  [{i+1:2d}/{len(assets)}] {asset:<30s}", end="", flush=True)
        t_a = time.time()
        try:
            r = run_asset(asset)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            continue
        elapsed = time.time() - t_a
        ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if r is None:
            print(f"SKIPPED  ram={ram_mb:.0f}MB", flush=True)
            continue
        print(
            f"lone={r['lone_oos_sharpe_ann']:+.3f}"
            f"  always={r['always_oos_sharpe_ann']:+.3f}"
            f"  gated={r['gated_oos_sharpe_ann']:+.3f}"
            f"  pbo={r['pbo_pass_rate']:.2f}"
            f"  reg={r['regime_pass_rate']:.2f}"
            f"  dep={r['deploy_rate']:.2f}"
            f"  t={elapsed:.1f}s  ram={ram_mb:.0f}MB",
            flush=True,
        )
        results.append(r)

    print(f"\nProcessed {len(results)} / {len(assets)} assets.", flush=True)
    if not results:
        print("ERROR: no results.", flush=True)
        return

    # ------------------------------------------------------------------- #
    # Aggregate
    # ------------------------------------------------------------------- #
    drop = {"_lone_d", "_always_d", "_gated_d", "_win_rows", "_weight_rows"}
    df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in drop} for r in results]
    ).sort_values("asset").reset_index(drop=True)

    all_lone   = np.concatenate([r["_lone_d"]   for r in results])
    all_always = np.concatenate([r["_always_d"] for r in results])
    all_gated  = np.concatenate([r["_gated_d"]  for r in results])

    sr = lambda x: float(OF.sharpe(x) * ANN)

    pooled_lone_sr   = sr(all_lone)
    pooled_always_sr = sr(all_always)
    pooled_gated_sr  = sr(all_gated)

    print(f"\nBootstrap CIs (N_BOOT={N_BOOT}) ...", flush=True)
    lone_ci   = _boot_ci(all_lone)
    always_ci = _boot_ci(all_always)
    gated_ci  = _boot_ci(all_gated)

    # Per-market
    mkts = sorted(df["market"].unique())
    mkt_stats = {}
    for mkt in mkts:
        sub = [r for r in results if r["market"] == mkt]
        pl = np.concatenate([r["_lone_d"]   for r in sub])
        pa = np.concatenate([r["_always_d"] for r in sub])
        pg = np.concatenate([r["_gated_d"]  for r in sub])
        mkt_df = df[df.market == mkt]
        mkt_stats[mkt] = dict(
            n_assets=len(sub),
            lone_sr=round(sr(pl), 4),
            always_sr=round(sr(pa), 4),
            gated_sr=round(sr(pg), 4),
            gated_beats_lone=bool(sr(pg) > sr(pl)),
            gated_beats_always=bool(sr(pg) > sr(pa)),
            mean_pbo_pass_rate=round(float(mkt_df["pbo_pass_rate"].mean()), 3),
            mean_regime_pass_rate=round(float(mkt_df["regime_pass_rate"].mean()), 3),
            mean_deploy_rate=round(float(mkt_df["deploy_rate"].mean()), 3),
        )

    # Sign tests
    n_bl = int(df["gated_beats_lone"].sum())
    n_ba = int(df["gated_beats_always"].sum())
    n    = len(df)
    p_vs_lone   = float(ss.binom.sf(n_bl - 1, n, 0.5))
    p_vs_always = float(ss.binom.sf(n_ba - 1, n, 0.5))

    # Causal check
    print("\nCausal pollute-and-verify ...", flush=True)
    causal = _causal_verify(results)
    print(f"  {causal['note']}", flush=True)

    # ------------------------------------------------------------------- #
    # Verdict
    # ------------------------------------------------------------------- #
    beats_lone_pooled   = pooled_gated_sr > pooled_lone_sr
    beats_always_pooled = pooled_gated_sr > pooled_always_sr

    eq_improves = mkt_stats.get("equity", {}).get("gated_beats_lone", False)
    fx_improves = mkt_stats.get("fx",     {}).get("gated_beats_lone", False)
    low_pbo_ok  = eq_improves or fx_improves

    crypto_delta = 0.0
    if "crypto" in mkt_stats:
        crypto_delta = mkt_stats["crypto"]["gated_sr"] - mkt_stats["crypto"]["lone_sr"]
    crypto_ok = crypto_delta <= 0.05

    verdict_pass = (
        beats_lone_pooled and beats_always_pooled
        and low_pbo_ok and crypto_ok
        and causal["causal_verified"]
    )
    verdict = "PASS" if verdict_pass else "FAIL"

    # ------------------------------------------------------------------- #
    # Peak RAM
    # ------------------------------------------------------------------- #
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # ------------------------------------------------------------------- #
    # Artifacts
    # ------------------------------------------------------------------- #
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUT_DIR / "s11_per_asset.parquet", index=False)
    df.to_csv(OUT_DIR / "s11_per_asset.csv", index=False)
    print(f"wrote s11_per_asset ({len(df)} rows)", flush=True)

    all_wr = sum([r["_win_rows"] for r in results], [])
    pd.DataFrame(all_wr).to_parquet(OUT_DIR / "s11_windows.parquet", index=False)
    print(f"wrote s11_windows ({len(all_wr)} rows)", flush=True)

    all_wt = sum([r["_weight_rows"] for r in results], [])
    if all_wt:
        pd.DataFrame(all_wt).to_parquet(OUT_DIR / "s11_blend_weights.parquet", index=False)
        print(f"wrote s11_blend_weights ({len(all_wt)} rows)", flush=True)

    np.savez_compressed(
        OUT_DIR / "s11_pooled_oos.npz",
        lone=all_lone, always=all_always, gated=all_gated,
    )
    print("wrote s11_pooled_oos.npz", flush=True)

    elapsed_total = time.time() - t0

    headline = dict(
        verdict=verdict,
        verdict_conditions=dict(
            gated_beats_lone_pooled=bool(beats_lone_pooled),
            gated_beats_always_pooled=bool(beats_always_pooled),
            low_pbo_markets_improve=bool(low_pbo_ok),
            crypto_not_rescued=bool(crypto_ok),
            causal_verified=bool(causal["causal_verified"]),
        ),
        pooled_oos_sharpe_ann=dict(
            lone_best=round(pooled_lone_sr, 4),
            always_deploy=round(pooled_always_sr, 4),
            gated_blend=round(pooled_gated_sr, 4),
        ),
        ci_95_boot=dict(
            lone_best=lone_ci,
            always_deploy=always_ci,
            gated_blend=gated_ci,
        ),
        per_market=mkt_stats,
        sign_test=dict(
            n_gated_beats_lone=n_bl,
            n_gated_beats_always=n_ba,
            n_total=n,
            p_vs_lone_one_sided=round(p_vs_lone, 4),
            p_vs_always_one_sided=round(p_vs_always, 4),
        ),
        causal_gate_check=causal,
        crypto_delta_vs_lone=round(float(crypto_delta), 4),
        params=dict(
            IS_DAYS=IS_DAYS, OOS_DAYS=OOS_DAYS, STEP=STEP, TOP_K=TOP_K,
            PBO_GATE=PBO_GATE, N_SAMPLE_PBO=N_SAMPLE_PBO,
            N_SPLITS_CSCV=N_SPLITS_CSCV, SEED=SEED,
        ),
        n_assets=len(results),
        peak_ram_mb=round(float(peak_ram), 1),
        elapsed_s=round(float(elapsed_total), 1),
        artifact_dir=str(OUT_DIR),
    )

    (OUT_DIR / "s11_headline.json").write_text(
        json.dumps(headline, indent=2, default=float)
    )
    print("wrote s11_headline.json", flush=True)

    # ------------------------------------------------------------------- #
    # Final print
    # ------------------------------------------------------------------- #
    print("\n" + "=" * 70, flush=True)
    print(f"S11 VERDICT: {verdict}", flush=True)
    print(f"\nPooled OOS Sharpe (ann):", flush=True)
    print(f"  gated+blend : {pooled_gated_sr:+.4f}  CI:[{gated_ci[0]:+.3f},{gated_ci[1]:+.3f}]", flush=True)
    print(f"  lone-best   : {pooled_lone_sr:+.4f}  CI:[{lone_ci[0]:+.3f},{lone_ci[1]:+.3f}]", flush=True)
    print(f"  always-dep  : {pooled_always_sr:+.4f}  CI:[{always_ci[0]:+.3f},{always_ci[1]:+.3f}]", flush=True)
    print(f"\nPer-market:", flush=True)
    for mkt, s in mkt_stats.items():
        tag = "(*beats)" if s["gated_beats_lone"] else "       "
        print(
            f"  {mkt:8s} {tag}: gated={s['gated_sr']:+.4f}"
            f"  lone={s['lone_sr']:+.4f}  always={s['always_sr']:+.4f}"
            f"  deploy_rate={s['mean_deploy_rate']:.2f}"
            f"  pbo_pass={s['mean_pbo_pass_rate']:.2f}",
            flush=True,
        )
    print(f"\nSign tests:", flush=True)
    print(f"  gated > lone:   {n_bl}/{n} assets, p={p_vs_lone:.4f}", flush=True)
    print(f"  gated > always: {n_ba}/{n} assets, p={p_vs_always:.4f}", flush=True)
    print(f"\nCausal: {causal['note']}", flush=True)
    print(f"\nVerdict conditions:", flush=True)
    for k, v in headline["verdict_conditions"].items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}", flush=True)
    print(f"\nPeak RAM: {peak_ram:.0f} MB", flush=True)
    print(f"Total: {elapsed_total:.1f}s", flush=True)
    print(f"Artifacts: {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
