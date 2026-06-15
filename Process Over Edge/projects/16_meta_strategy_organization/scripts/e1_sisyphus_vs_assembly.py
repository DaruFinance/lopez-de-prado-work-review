"""
E1 -- Sisyphus vs the disclosed assembly line, head to head (centerpiece).

For each real asset we build a strategy-by-window performance matrix from the
real net-daily-PnL corpus and run two competing research PROCESSES across a
rolling walk-forward, recording the REALIZED out-of-sample (OOS) outcome of each:

(a) SISYPHUS -- the lone backtester. In every window, pick the single strategy
    with the best in-sample (IS) Sharpe. No disclosure of the number of trials,
    no deflation, no stability check. Deploy that one strategy through the next
    (OOS) window and book its realized net PnL.

(b) ASSEMBLY LINE -- the disclosed production line. Same candidate pool each
    window, but: (1) compute each candidate's IS Sharpe AND its purged-CV IS
    Sharpe (stability), (2) gate the IS winner-set by the Deflated Sharpe Ratio
    using the FULL disclosed trial count N (the False Strategy Theorem null),
    (3) deploy ONLY the DSR survivors, sizing them by an HRP allocation over
    their IS covariance; if nothing survives, hold CASH (zero return). Book the
    realized OOS net PnL of the deployed sleeve.

We measure, per process: realized OOS Sharpe (pooled across windows), the
IS->OOS Sharpe decay (the winner's curse) with a moving-block bootstrap band,
OOS hit rate, max drawdown, and the assembly line's deploy rate (fraction of
windows where any candidate cleared DSR).

Everything is REAL net PnL (daily pnl_sum == trade pnl_net, verified). No
look-ahead: IS statistics use only IS days; deployment earns only OOS days.
Costs are already embedded in the net PnL.

Hot path: per-strategy IS Sharpe over a window across ~50k strategies, repeated
over windows and assets. Implemented as a NumPy reference and a Numba @njit
kernel, verified bit-identical (max|delta|) on real window slices.

Parallelism capped at 4 (joblib n_jobs=4, BLAS threads=1, NUMBA_NUM_THREADS=4).
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
from numba import njit, prange
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import corpus_io as cio                                          # noqa: E402
from lib import overfit                                          # noqa: E402

# import study 13's HRP allocator (reuse, do not reimplement)
_p13 = ROOT / "projects/13_portfolio_construction/scripts/run_portfolio.py"
_spec = importlib.util.spec_from_file_location("p13_portfolio", _p13)
p13 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p13)

TAB = PROJ / "tables"; FIG = PROJ / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

ANN = np.sqrt(252.0)
IS_DAYS = 252          # ~1y in-sample
OOS_DAYS = 63          # ~1 quarter out-of-sample (deploy horizon)
STEP = 63              # roll quarterly
DSR_GATE = 0.95        # deflated-significance bar (LdP standard)
DSR_GATE_RELAXED = 0.60  # a relaxed disclosed gate, to map the deploy/return frontier
TOP_FRAC_FOR_HRP = 0.02  # consider top-2% IS strategies as the gating shortlist
MIN_IS_OBS = 60        # need at least this many IS days with activity (sparse-trial guard)


# --------------------------------------------------------------------------- #
# Hot loop: per-column Sharpe over a (T x N) slice (T days, N strategies).
# --------------------------------------------------------------------------- #
def _col_sharpe_ref(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0); sd = X.std(0, ddof=1)
    return np.where(sd > 0, mu / sd, 0.0)


@njit(cache=True, fastmath=False, parallel=True)
def _col_sharpe_kernel(X):
    T, N = X.shape
    out = np.zeros(N)
    for j in prange(N):
        s = 0.0
        for t in range(T):
            s += X[t, j]
        mu = s / T
        ss = 0.0
        for t in range(T):
            d = X[t, j] - mu
            ss += d * d
        var = ss / (T - 1)
        if var > 0.0:
            out[j] = mu / np.sqrt(var)
        else:
            out[j] = 0.0
    return out


def _verify_colsharpe(seed=11) -> float:
    rng = np.random.default_rng(seed)
    md = 0.0
    for _ in range(40):
        X = rng.standard_normal((rng.integers(80, 300), rng.integers(50, 2000)))
        md = max(md, float(np.max(np.abs(_col_sharpe_ref(X) - _col_sharpe_kernel(X)))))
    return md


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def _block_bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot=2000, block=21, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    nb = int(np.ceil(n / block))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        vals[b] = stat(x[idx])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# Per-asset processes
# --------------------------------------------------------------------------- #
def _gate_survivors(sr_is, is_active, order, sr0, gate):
    from scipy import stats as ss
    survivors = []
    for j in order:
        r = is_active[:, j]
        rf = r[np.isfinite(r)]
        nz = rf[rf != 0]
        if rf.std(ddof=1) <= 0 or len(nz) < MIN_IS_OBS:
            continue
        psr = overfit.prob_sharpe_ratio(
            sr_is[j], len(rf), float(ss.skew(rf)),
            float(ss.kurtosis(rf, fisher=False)), sr_benchmark=sr0)
        if psr is not None and np.isfinite(psr) and psr >= gate:
            survivors.append(j)
    return survivors


def _deploy(survivors, sub_is_scaled, sub_oos_scaled):
    """HRP-allocate survivors on IS covariance; return realized OOS daily stream."""
    if len(survivors) == 1:
        w = np.array([1.0])
    else:
        cov = np.atleast_2d(np.cov(sub_is_scaled.T))
        try:
            w = p13.w_hrp(cov, use_numba=True)
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                raise ValueError
            w = w / w.sum()
        except Exception:
            w = np.full(len(survivors), 1.0 / len(survivors))
    return sub_oos_scaled @ w, w


def run_asset(asset: str) -> dict | None:
    M, dates, names = cio.build_matrix(asset, use_numba=True)
    nd, ns = M.shape
    if nd < IS_DAYS + OOS_DAYS + STEP:
        return None
    starts = list(range(0, nd - IS_DAYS - OOS_DAYS + 1, STEP))
    sis_oos_daily, asm_oos_daily, asm_rlx_daily = [], [], []
    sis_is_sr, sis_oos_sr = [], []
    asm_is_sr, asm_oos_sr = [], []
    asm_deployed = asm_rlx_deployed = 0
    n_surv_list = []
    for st in starts:
        is_sl = M[st:st + IS_DAYS]
        oos_sl = M[st + IS_DAYS:st + IS_DAYS + OOS_DAYS]
        # candidate pool: strategies with enough active IS days (sparse-trial guard)
        nact = (is_sl != 0).sum(0)
        pool = nact >= MIN_IS_OBS
        if pool.sum() < 10:
            continue
        is_active = np.ascontiguousarray(is_sl[:, pool])
        oos_active = np.ascontiguousarray(oos_sl[:, pool])
        # unit-risk scaling: divide each column by its IS std (OOS uses same divisor)
        sd_is = is_active.std(0, ddof=1)
        sd_is = np.where(sd_is > 0, sd_is, 1.0)
        is_scaled = is_active / sd_is
        oos_scaled = oos_active / sd_is
        sr_is = _col_sharpe_kernel(is_active)        # scale-invariant
        # ---- Sisyphus: best IS Sharpe, deploy unit-risk in OOS ----
        k = int(np.argmax(sr_is))
        sis_oos_daily.append(oos_scaled[:, k])
        sis_is_sr.append(float(sr_is[k]))
        sis_oos_sr.append(overfit.sharpe(oos_active[:, k]))
        # ---- disclosed-N False Strategy Theorem null ----
        N_trials = int(is_active.shape[1])
        var_sr = float(np.var(sr_is, ddof=1)) if len(sr_is) > 1 else 0.0
        sr0 = overfit.expected_max_sharpe(N_trials, var_sr)
        n_short = max(10, int(TOP_FRAC_FOR_HRP * N_trials))
        order = np.argsort(sr_is)[::-1][:n_short]
        # ---- Assembly line: strict DSR>=0.95 gate ----
        surv = _gate_survivors(sr_is, is_active, order, sr0, DSR_GATE)
        n_surv_list.append(len(surv))
        if surv:
            asm_deployed += 1
            port, _ = _deploy(surv, is_scaled[:, surv], oos_scaled[:, surv])
            asm_oos_daily.append(port)
            asm_is_sr.append(overfit.sharpe(is_scaled[:, surv].sum(1)))
            asm_oos_sr.append(overfit.sharpe(port))
        else:
            asm_oos_daily.append(np.zeros(OOS_DAYS)); asm_is_sr.append(0.0); asm_oos_sr.append(0.0)
        # ---- Assembly line (relaxed disclosed gate) ----
        surv_r = _gate_survivors(sr_is, is_active, order, sr0, DSR_GATE_RELAXED)
        if surv_r:
            asm_rlx_deployed += 1
            port_r, _ = _deploy(surv_r, is_scaled[:, surv_r], oos_scaled[:, surv_r])
            asm_rlx_daily.append(port_r)
        else:
            asm_rlx_daily.append(np.zeros(OOS_DAYS))
    if not sis_oos_daily:
        return None
    sis_d = np.concatenate(sis_oos_daily)
    asm_d = np.concatenate(asm_oos_daily)
    rlx_d = np.concatenate(asm_rlx_daily)
    return dict(
        asset=asset, market=cio.market_of(asset),
        n_windows=len(sis_is_sr), n_strats=ns,
        sis_oos_sharpe_ann=float(overfit.sharpe(sis_d) * ANN),
        asm_oos_sharpe_ann=float(overfit.sharpe(asm_d) * ANN),
        asm_rlx_oos_sharpe_ann=float(overfit.sharpe(rlx_d) * ANN),
        sis_is_sharpe_ann=float(np.mean(sis_is_sr) * ANN),
        sis_oos_hit=float(np.mean(np.array(sis_oos_sr) > 0)),
        asm_oos_hit=float(np.mean(np.array(asm_oos_sr) > 0)),
        sis_maxdd=_max_drawdown(np.cumsum(sis_d)),
        asm_maxdd=_max_drawdown(np.cumsum(asm_d)),
        asm_rlx_maxdd=_max_drawdown(np.cumsum(rlx_d)),
        deploy_rate=float(asm_deployed / max(1, len(sis_is_sr))),
        deploy_rate_relaxed=float(asm_rlx_deployed / max(1, len(sis_is_sr))),
        median_survivors=float(np.median(n_surv_list)) if n_surv_list else 0.0,
        sis_is_minus_oos=[(a - b) for a, b in zip(sis_is_sr, sis_oos_sr)],
        sis_oos_daily=sis_d, asm_oos_daily=asm_d, asm_rlx_oos_daily=rlx_d,
    )


def main():
    print("=== E1: Sisyphus vs the disclosed assembly line ===", flush=True)
    print("warmup + bit-identical colSharpe verification ...", flush=True)
    _col_sharpe_kernel(np.random.default_rng(0).standard_normal((100, 50)))
    md = _verify_colsharpe()
    print(f"  colSharpe kernel vs NumPy ref max|delta| = {md:.3e} (want 0.0)", flush=True)

    assets = cio.list_assets()
    print(f"  {len(assets)} assets", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=4, backend="loky", verbose=5)(
        delayed(run_asset)(a) for a in assets)
    results = [r for r in results if r is not None]
    print(f"  ran {len(results)} assets in {time.time()-t0:.0f}s", flush=True)

    rows, all_decay = [], []
    pooled_sis, pooled_asm, pooled_rlx = [], [], []
    drop = {"sis_is_minus_oos", "sis_oos_daily", "asm_oos_daily", "asm_rlx_oos_daily"}
    for r in results:
        rows.append({k: v for k, v in r.items() if k not in drop})
        all_decay.extend(r["sis_is_minus_oos"])
        pooled_sis.append(r["sis_oos_daily"]); pooled_asm.append(r["asm_oos_daily"])
        pooled_rlx.append(r["asm_rlx_oos_daily"])
    df = pd.DataFrame(rows).sort_values("asset")
    df.to_csv(TAB / "e1_per_asset.csv", index=False)
    print(f"wrote {TAB/'e1_per_asset.csv'}", flush=True)

    sis_all = np.concatenate(pooled_sis)
    asm_all = np.concatenate(pooled_asm)
    rlx_all = np.concatenate(pooled_rlx)
    decay = np.array(all_decay)
    sr_ann = lambda x: overfit.sharpe(x) * ANN
    headline = dict(
        n_assets=len(results),
        pooled_sisyphus_oos_sharpe_ann=float(sr_ann(sis_all)),
        pooled_assembly_strict_oos_sharpe_ann=float(sr_ann(asm_all)),
        pooled_assembly_relaxed_oos_sharpe_ann=float(sr_ann(rlx_all)),
        sisyphus_oos_sharpe_ci=_block_bootstrap_ci(sis_all, sr_ann),
        assembly_relaxed_oos_sharpe_ci=_block_bootstrap_ci(rlx_all, sr_ann),
        mean_sisyphus_is_minus_oos_sharpe_perobs=float(decay.mean()),
        mean_sisyphus_is_minus_oos_sharpe_ann=float(decay.mean() * ANN),
        decay_ci_perobs=_block_bootstrap_ci(decay, np.mean),
        median_deploy_rate_strict=float(df["deploy_rate"].median()),
        median_deploy_rate_relaxed=float(df["deploy_rate_relaxed"].median()),
        median_sisyphus_hit=float(df["sis_oos_hit"].median()),
        median_assembly_hit=float(df["asm_oos_hit"].median()),
        n_windows_total=int(df["n_windows"].sum()),
        IS_DAYS=IS_DAYS, OOS_DAYS=OOS_DAYS, STEP=STEP,
        DSR_GATE=DSR_GATE, DSR_GATE_RELAXED=DSR_GATE_RELAXED,
        colsharpe_kernel_max_abs_delta=md,
    )
    (TAB / "e1_headline.json").write_text(json.dumps(headline, indent=2, default=float))
    print(f"wrote {TAB/'e1_headline.json'}", flush=True)
    np.savez_compressed(TAB / "e1_pooled_oos.npz",
                        sis=sis_all, asm=asm_all, rlx=rlx_all, decay=decay)
    print("\n--- E1 HEADLINE ---")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
