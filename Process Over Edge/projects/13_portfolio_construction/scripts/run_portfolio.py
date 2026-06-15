#!/usr/bin/env python3
"""
Project 13, Portfolio Construction at scale: covariance DENOISING / DETONING,
HIERARCHICAL RISK PARITY (HRP), NESTED CLUSTERED OPTIMIZATION (NCO), and
THEORY-IMPLIED CORRELATION (TIC), benchmarked WALK-FORWARD against the
Markowitz curse (mean-variance / min-variance on the raw sample covariance),
inverse-variance, and naive 1/N.

Primary sources (formulas quoted in-line where they bite):
  - AFML (López de Prado 2018) Ch.16  : HRP (tree clustering, quasi-diagonalisation,
                                         recursive bisection, no matrix inversion).
  - "A Robust Estimator of the Efficient Frontier" (LdP 2019) / ML4AM Ch.2,4,7:
                                         MP denoising (constant-residual eigenvalue),
                                         detoning (drop the market eigenvector), NCO.
  - "Estimation of Theory-Implied Correlation Matrices" (LdP & Lewis 2018): TIC.
  - Headline overfitting metric via lib/overfit.py (DSR / False Strategy Theorem).

METHODOLOGY (house rules):
  * Real data only, multi-market (Crypto + US Equities + Forex).
  * Causal-only, WALK-FORWARD: weights are estimated on an IN-SAMPLE window and held
    over the *following* OOS window; we NEVER score weights on the data that built them.
  * Headline metric = realised OOS portfolio variance (annualised vol), plus
    concentration (HHI, effective-N, condition number of the cov used) and the DSR of
    the realised OOS portfolio return stream against the menu of allocators tried.
  * Two universes (different experiments):
      (1) ACROSS ASSETS   , daily close-to-close returns of ~40 instruments
                             (crypto perps + equity ETFs + FX majors).
      (2) ACROSS STRATEGIES, the LdP use-case: a few hundred per-strategy daily PnL
                             series per market, allocate the risk budget across them.

RAM (WSL ~46 GB; covariance is N x N in the universe size):
  * universe=assets     : N ~ 40  -> cov is 40x40, trivial.
  * universe=strategies : N capped via --n-strat (default 300/market) -> 300x300 cov,
                          ~0.7 MB; the return panel is the only sizeable object
                          (T_days x N x 8 B; ~2800 x 300 x 8 ≈ 6.7 MB/market).
  Peak RAM estimate (full run, all markets, strategies universe, 300/mkt): < 1.5 GB.
  No N x N is ever materialised beyond the (bounded) universe; nothing is tiled because
  nothing is large. numpy.linalg.eigh dominates; clustering loops are the only Python
  hot spots and are Numba'd (recursive-bisection inverse-variance + cluster-var).

PERFORMANCE: profile first (`--profile`). The eigendecomposition + sklearn linkage are
C/Fortran already; only the HRP recursive bisection and the per-cluster variance loop
are pure-Python hot spots, so those get njit kernels, verified bit-identical against the
numpy reference (see _selftest_numba()).
"""
from __future__ import annotations
import sys, os, glob, time, argparse, warnings, cProfile, pstats, io
import numpy as np
import pandas as pd

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
import style as ST  # noqa: F401  (figures done in a separate pass)

warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRYPTO_1M = cfg.CRYPTO_1M
FX_1M     = cfg.FX_1M
ETF_DIR   = cfg.EQUITY_1M
PNL_BASE  = cfg.PNL_DAILY
ANN       = np.sqrt(252.0)        # daily -> annualised vol/Sharpe for display
SEED      = 7

# A light asset taxonomy for TIC (class -> members). Used only when --tic is on.
TAXONOMY = {
    "crypto_majors":  ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "TRX", "LTC"],
    "crypto_alts":    ["AVAX", "LINK", "DOT", "ATOM", "NEAR", "APT", "ARB", "SUI",
                        "UNI", "AAVE", "ICP", "ALGO", "APE", "ETC", "BCH", "XLM",
                        "HBAR", "ZEC", "1000SHIB"],
    "equity_broad":   ["SPY", "QQQ", "IWM"],
    "equity_sector":  ["XLE", "XLF", "XLK", "XLV"],
    "equity_vol":     ["VXX", "UVXY"],
    "fx_usd":         ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    "fx_cross":       ["EURGBP"],
}


# ============================================================================ #
# Numba kernels (the only pure-Python hot spots: HRP recursive bisection)
# ============================================================================ #
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*a, **k):                       # graceful fallback decorator
        def wrap(f): return f
        return wrap if (a and callable(a[0])) is False else a[0]


@njit(cache=True)
def _cluster_var(cov, idx):
    """Inverse-variance portfolio variance of a sub-cluster (AFML getClusterVar).
    w_i ∝ 1/cov_ii (normalised within the cluster); var = w' Σ w."""
    n = idx.shape[0]
    iv = np.empty(n)
    s = 0.0
    for i in range(n):
        v = cov[idx[i], idx[i]]
        iv[i] = 1.0 / v if v > 0 else 0.0
        s += iv[i]
    if s > 0:
        for i in range(n):
            iv[i] /= s
    var = 0.0
    for i in range(n):
        for j in range(n):
            var += iv[i] * cov[idx[i], idx[j]] * iv[j]
    return var


@njit(cache=True)
def _hrp_recursive_bisection(cov, sort_ix):
    """AFML Ch.16 recursive bisection over a quasi-diagonalised order `sort_ix`.
    Splits each cluster in half, scales the two halves by the inverse of their
    inverse-variance cluster variance (alpha = 1 - varL/(varL+varR)), recursing.
    Returns weights aligned to ORIGINAL asset indices. No matrix inversion."""
    n = sort_ix.shape[0]
    w = np.ones(n)
    # stack of (start, end) half-open slices into sort_ix; process LIFO
    starts = np.empty(n, np.int64)
    ends = np.empty(n, np.int64)
    top = 0
    starts[0] = 0; ends[0] = n; top = 1
    while top > 0:
        top -= 1
        s = starts[top]; e = ends[top]
        if e - s <= 1:
            continue
        mid = s + (e - s) // 2
        left = sort_ix[s:mid]
        right = sort_ix[mid:e]
        vL = _cluster_var(cov, left)
        vR = _cluster_var(cov, right)
        denom = vL + vR
        alpha = 1.0 - (vL / denom) if denom > 0 else 0.5
        for k in range(s, mid):
            w[sort_ix[k]] *= alpha
        for k in range(mid, e):
            w[sort_ix[k]] *= (1.0 - alpha)
        starts[top] = s; ends[top] = mid; top += 1
        starts[top] = mid; ends[top] = e; top += 1
    return w


def _hrp_recursive_bisection_ref(cov, sort_ix):
    """Pure-numpy reference for bit-identity verification (mirrors the njit logic)."""
    n = sort_ix.shape[0]
    w = np.ones(n)
    stack = [(0, n)]
    def cvar(idx):
        d = np.diag(cov)[idx]
        iv = np.where(d > 0, 1.0 / d, 0.0)
        ssum = iv.sum()
        if ssum > 0:
            iv = iv / ssum
        return float(iv @ cov[np.ix_(idx, idx)] @ iv)
    while stack:
        s, e = stack.pop()
        if e - s <= 1:
            continue
        mid = s + (e - s) // 2
        left = sort_ix[s:mid]; right = sort_ix[mid:e]
        vL, vR = cvar(left), cvar(right)
        denom = vL + vR
        alpha = 1.0 - (vL / denom) if denom > 0 else 0.5
        w[left] *= alpha
        w[right] *= (1.0 - alpha)
        stack.append((s, mid)); stack.append((mid, e))
    return w


# ============================================================================ #
# Covariance estimators: MP denoising + detoning  (ML4AM Ch.2)
# ============================================================================ #
def _corr_from_cov(cov):
    std = np.sqrt(np.diag(cov))
    std = np.where(std > 0, std, 1.0)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1, 1), std


def _cov_from_corr(corr, std):
    return corr * np.outer(std, std)


def _mp_pdf(var, q, pts=1000):
    """Marčenko-Pastur density for ratio q=T/N, noise variance `var`."""
    lo = var * (1 - np.sqrt(1.0 / q)) ** 2
    hi = var * (1 + np.sqrt(1.0 / q)) ** 2
    x = np.linspace(lo, hi, pts)
    pdf = q / (2 * np.pi * var * x) * np.sqrt(np.clip((hi - x) * (x - lo), 0, None))
    return pd.Series(pdf, index=x)


def _fit_mp(eigvals, q, bwidth=0.01):
    """Find the noise variance var* that best fits the empirical eigenvalue density
    to the MP law (LdP findMaxEval / errPDFs). Returns (var*, lambda_max)."""
    from scipy.optimize import minimize
    from sklearn.neighbors import KernelDensity
    ev = eigvals.reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bwidth).fit(ev)

    def err(var):
        var = float(var[0])
        if var <= 0:
            return 1e12
        pdf0 = _mp_pdf(var, q)
        x = pdf0.index.values.reshape(-1, 1)
        pdf1 = np.exp(kde.score_samples(x))
        return float(np.sum((pdf1 - pdf0.values) ** 2))

    res = minimize(err, x0=np.array([0.5]), bounds=[(1e-5, 1 - 1e-5)])
    var = float(res.x[0]) if res.success else 1.0
    lam_max = var * (1 + np.sqrt(1.0 / q)) ** 2
    return var, lam_max


def denoise_corr(corr, q, bwidth=0.01):
    """Constant-residual-eigenvalue denoising (ML4AM Ch.2 denoisedCorr).
    Eigenvalues below the MP edge are replaced by their common average so the
    matrix trace is preserved but the noisy bulk is flattened."""
    w, v = np.linalg.eigh(corr)
    order = np.argsort(w)[::-1]
    w = w[order]; v = v[: order]
    var, lam_max = _fit_mp(w, q, bwidth)
    n_facts = int((w > lam_max).sum())          # # eigenvalues above the MP edge = signal
    w_ = w.copy()
    if n_facts < len(w):
        tail = w_[n_facts:]
        w_[n_facts:] = tail.sum() / float(len(tail))  # constant residual
    corr_d = v @ np.diag(w_) @ v.T
    corr_d, _ = _corr_from_cov(corr_d)          # renormalise to unit diagonal
    return corr_d, n_facts, lam_max


def detone_corr(corr, n_market=1):
    """Remove the top `n_market` eigenvector(s) (the market component) then
    renormalise (ML4AM detoned-correlation). Detoning helps clustering by
    stripping the dominant common factor that swamps the off-diagonal structure."""
    w, v = np.linalg.eigh(corr)
    order = np.argsort(w)[::-1]
    w = w[order]; v = v[: order]
    wm = w[:n_market]; vm = v[: :n_market]
    corr_m = vm @ np.diag(wm) @ vm.T
    corr_t = corr - corr_m
    corr_t, _ = _corr_from_cov(corr_t)
    return corr_t


# ============================================================================ #
# Allocators
# ============================================================================ #
def w_equal(cov):
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


def w_inverse_variance(cov):
    iv = 1.0 / np.where(np.diag(cov) > 0, np.diag(cov), np.inf)
    return iv / iv.sum()


def w_min_variance(cov, long_only=True):
    """Min-variance (the Markowitz curse on a RAW sample cov): w ∝ Σ⁻¹·1.
    Uses a pseudo-inverse for numerical safety; optionally clips to long-only and
    renormalises (so we report the *practitioner's* curse, not a blown-up short)."""
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        inv = np.linalg.pinv(cov)
    except Exception:
        inv = np.linalg.pinv(cov + 1e-10 * np.eye(n))
    w = inv @ ones
    s = w.sum()
    w = w / s if s != 0 else np.full(n, 1.0 / n)
    if long_only:
        w = np.clip(w, 0, None)
        ws = w.sum()
        w = w / ws if ws > 0 else np.full(n, 1.0 / n)
    return w


def w_mean_variance(cov, mu, long_only=True):
    """Max-Sharpe / mean-variance tangency: w ∝ Σ⁻¹·μ (the classic Markowitz
    estimate that LdP shows is wildly unstable on raw sample inputs)."""
    n = cov.shape[0]
    try:
        inv = np.linalg.pinv(cov)
    except Exception:
        inv = np.linalg.pinv(cov + 1e-10 * np.eye(n))
    w = inv @ mu
    s = np.abs(w).sum()
    w = w / s if s != 0 else np.full(n, 1.0 / n)
    if long_only:
        w = np.clip(w, 0, None)
        ws = w.sum()
        w = w / ws if ws > 0 else np.full(n, 1.0 / n)
    return w


def _quasi_diag(link, n_items):
    """AFML getQuasiDiag: depth-first unwrap of a SciPy linkage into leaf order."""
    from scipy.cluster.hierarchy import to_tree
    root = to_tree(link, rd=False)
    order = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            order.append(node.id)
        else:
            # push right then left so left is processed first (preserves AFML order)
            stack.append(node.right)
            stack.append(node.left)
    return order


def _linkage_from_corr(corr, method="single"):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, 1))   # AFML correlation distance
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method=method), dist


def w_hrp(cov, use_numba=True):
    """Hierarchical Risk Parity (AFML Ch.16). Cluster -> quasi-diagonalise ->
    recursive bisection. No matrix inversion -> immune to the Markowitz curse."""
    corr, _ = _corr_from_cov(cov)
    link, _ = _linkage_from_corr(corr, method="single")
    sort_ix = np.asarray(_quasi_diag(link, cov.shape[0]), dtype=np.int64)
    if use_numba and _HAVE_NUMBA:
        return _hrp_recursive_bisection(cov, sort_ix)
    return _hrp_recursive_bisection_ref(cov, sort_ix)


def w_nco(cov, mu=None, max_k=None, long_only=True):
    """Nested Clustered Optimization (ML4AM Ch.7). Cluster the (denoised) corr with
    KMeans over a silhouette-chosen k; solve a min-variance (or mean-variance if mu
    given) WITHIN each cluster, then ACROSS the reduced cluster cov; compose.
    The nesting tames the condition number that wrecks one-shot Markowitz."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    n = cov.shape[0]
    corr, _ = _corr_from_cov(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, 1))
    kmax = max_k or max(2, min(int(n / 2), 25))
    best_k, best_q, best_lab = 2, -1e9, None
    rng = SEED
    for k in range(2, kmax + 1):
        if k >= n:
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=rng).fit_predict(dist)
        if len(set(lab)) < 2:
            continue
        sil = silhouette_samples(dist, lab)
        # LdP's clustering quality q = mean/std of silhouette
        grp = pd.Series(sil).groupby(lab)
        q = grp.mean().mean() / (grp.std().mean() + 1e-12)
        if q > best_q:
            best_q, best_k, best_lab = q, k, lab
    lab = best_lab if best_lab is not None else np.zeros(n, int)
    clusters = {c: np.where(lab == c)[0] for c in np.unique(lab)}

    # intra-cluster weights (min-var within each cluster)
    w_intra = np.zeros(n)
    for c, idx in clusters.items():
        sub = cov[np.ix_(idx, idx)]
        wc = w_min_variance(sub, long_only=long_only)
        w_intra[idx] = wc
    # reduced cluster-level covariance, then allocate across clusters
    cl_ids = list(clusters.keys())
    redcov = np.zeros((len(cl_ids), len(cl_ids)))
    redmu = np.zeros(len(cl_ids))
    for a, ca in enumerate(cl_ids):
        wa = w_intra[clusters[ca]]
        for b, cb in enumerate(cl_ids):
            wb = w_intra[clusters[cb]]
            redcov[a, b] = wa @ cov[np.ix_(clusters[ca], clusters[cb])] @ wb
        if mu is not None:
            redmu[a] = wa @ mu[clusters[ca]]
    if mu is not None:
        w_inter = w_mean_variance(redcov, redmu, long_only=long_only)
    else:
        w_inter = w_min_variance(redcov, long_only=long_only)
    # compose
    w = np.zeros(n)
    for a, c in enumerate(cl_ids):
        w[clusters[c]] = w_intra[clusters[c]] * w_inter[a]
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def tic_corr(corr, members, classes, rho=0.05):
    """Theory-Implied Correlation (LdP & Lewis 2018), light version. Builds a
    block correlation implied by a one-level taxonomy (within-class corr = sample
    within-class mean; cross-class = global mean shrunk by rho) and shrinks the
    empirical corr toward it. Returns a blended, PSD-projected correlation."""
    n = len(members)
    cls = np.array([classes.get(m, "other") for m in members])
    theory = np.full((n, n), 0.0)
    iu = np.triu_indices(n, 1)
    same = cls[iu[0]] == cls[iu[1]]
    within = corr[iu][same]
    cross = corr[iu][~same]
    wm = float(np.mean(within)) if within.size else 0.0
    cm = float(np.mean(cross)) if cross.size else 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                theory[i, j] = 1.0
            elif cls[i] == cls[j]:
                theory[i, j] = wm
            else:
                theory[i, j] = cm * (1.0 - rho)
    blended = 0.5 * corr + 0.5 * theory
    # PSD projection (clip negative eigenvalues)
    w, v = np.linalg.eigh(blended)
    w = np.clip(w, 1e-8, None)
    blended = v @ np.diag(w) @ v.T
    blended, _ = _corr_from_cov(blended)
    return blended


# ============================================================================ #
# Concentration / conditioning diagnostics
# ============================================================================ #
def hhi(w):
    w = np.abs(w); s = w.sum()
    if s <= 0:
        return 1.0
    p = w / s
    return float(np.sum(p * p))


def eff_n_from_w(w):
    h = hhi(w)
    return float(1.0 / h) if h > 0 else 1.0


def cond_number(cov):
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return float(ev.max() / ev.min()) if ev.size and ev.min() > 0 else np.inf


# ============================================================================ #
# Data loading
# ============================================================================ #
def _daily_close_from_ohlc_parquet(path, close_col="close", time_col=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    md = pq.read_schema(path)
    df = pf.read().to_pandas()
    if time_col is None:
        # crypto: open_time column; fx: 'key' is the index
        if "open_time" in df.columns:
            t = pd.DatetimeIndex(pd.to_datetime(df["open_time"], utc=True))
        elif "key" in df.columns:
            t = pd.DatetimeIndex(pd.to_datetime(df["key"], utc=True))
        elif isinstance(df.index, pd.DatetimeIndex):
            t = df.index.tz_localize("UTC") if df.index.tz is None else df.index
        else:
            raise ValueError(f"no time column in {path}")
    else:
        t = pd.DatetimeIndex(pd.to_datetime(df[time_col], utc=True))
    s = pd.Series(df[close_col].to_numpy(), index=t.tz_convert("UTC"))
    daily = s.groupby(s.index.normalize()).last()   # causal: last 1m close of the day
    daily.index = daily.index.tz_localize(None)
    return daily


CACHE_DIR = f"{PROJ}/../../data_cache/p13"


def load_assets_universe(use_cache=True):
    """Daily close-to-close LOG returns for ~40 instruments across 3 markets.
    Returns (returns_df [dates x instrument], class_map). Long-only / spot-agnostic:
    returns are an allocation experiment, not a trade simulation.

    PROFILE FINDING: ~98% of wall time is gzip/parquet loading (ETF CSVs dominate),
    not the portfolio math. We cache the assembled daily-return panel to parquet so
    reruns are instant; pass use_cache=False to force a rebuild."""
    import json
    cpath = os.path.join(os.path.normpath(CACHE_DIR), "assets_daily_returns.parquet")
    mpath = os.path.join(os.path.normpath(CACHE_DIR), "assets_classmap.json")
    if use_cache and os.path.exists(cpath) and os.path.exists(mpath):
        ret = pd.read_parquet(cpath)
        with open(mpath) as fh:
            classmap = json.load(fh)
        return ret, classmap
    ret, classmap = _build_assets_universe()
    os.makedirs(os.path.normpath(CACHE_DIR), exist_ok=True)
    ret.to_parquet(cpath)
    with open(mpath, "w") as fh:
        json.dump(classmap, fh)
    return ret, classmap


def _build_assets_universe():
    series = {}
    classmap = {}
    for f in sorted(glob.glob(f"{CRYPTO_1M}/*_1m.parquet")):
        sym = os.path.basename(f).replace("USDT_1m.parquet", "")
        try:
            series[sym] = _daily_close_from_ohlc_parquet(f)
            classmap[sym] = next((c for c, m in TAXONOMY.items() if sym in m), "crypto_alts")
        except Exception as e:
            print(f"  [skip crypto {sym}] {e}")
    for f in sorted(glob.glob(f"{FX_1M}/*_fx1m.parquet")):
        sym = os.path.basename(f).replace("_fx1m.parquet", "")
        try:
            series[sym] = _daily_close_from_ohlc_parquet(f)
            classmap[sym] = next((c for c, m in TAXONOMY.items() if sym in m), "fx_usd")
        except Exception as e:
            print(f"  [skip fx {sym}] {e}")
    # ETFs: daily LastTradePrice per TradeDate (last RTH+ext bar = causal close)
    tickers = sorted(set(os.path.basename(x).split("_")[0].split(".")[0]
                         for x in glob.glob(f"{ETF_DIR}/*.csv.gz")))
    for tk in tickers:
        fs = sorted(glob.glob(f"{ETF_DIR}/{tk}_20*.csv.gz"))
        if not fs:
            continue
        parts = []
        for f in fs:
            try:
                d = pd.read_csv(f, usecols=["TradeDate", "LastTradePrice"])
                parts.append(d)
            except Exception:
                pass
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        daily = d.groupby("TradeDate")["LastTradePrice"].last()
        daily.index = pd.to_datetime(daily.index)
        series[tk] = daily
        classmap[tk] = next((c for c, m in TAXONOMY.items() if tk in m), "equity_broad")

    px = pd.DataFrame(series).sort_index()
    ret = np.log(px / px.shift(1))
    ret = ret.dropna(how="all")
    return ret, classmap


def load_strategies_universe(n_strat=300, max_markets=None, seed=SEED):
    """The LdP use-case: sample up to n_strat strategies per market from the daily
    PnL corpus and allocate across them. Returns {market: returns_df}.
    Sampling prefers LOW mutual correlation (greedy de-correlation on a random
    over-sample) so the universe is a genuine diversification problem, not 300 clones."""
    import pyarrow.dataset as ds
    rng = np.random.default_rng(seed)
    assets = sorted(os.path.basename(p).replace("asset=", "")
                    for p in glob.glob(f"{PNL_BASE}/asset=*"))
    if max_markets:
        assets = assets[:max_markets]
    out = {}
    for a in assets:
        fs = glob.glob(f"{PNL_BASE}/asset={a}/*.parquet")
        if not fs:
            continue
        tbl = ds.dataset(fs).to_table(columns=["strategy_name", "date", "pnl_sum"])
        df = tbl.to_pandas()
        uniq = df["strategy_name"].unique()
        # over-sample then greedily de-correlate down to n_strat
        over = min(len(uniq), n_strat * 4)
        pick0 = uniq if len(uniq) <= over else rng.choice(uniq, over, replace=False)
        sub = df[df["strategy_name"].isin(set(pick0))]
        m = sub.pivot_table(index="date", columns="strategy_name", values="pnl_sum",
                            aggfunc="sum", fill_value=0.0).sort_index()
        if m.shape[1] > n_strat:
            m = _greedy_decorrelate(m, n_strat, rng)
        m.index = pd.to_datetime(m.index)
        out[a] = m
    return out


def _greedy_decorrelate(m, k, rng):
    """Pick k columns minimising mutual |corr| greedily (cheap diversity sampler)."""
    R = m.to_numpy(float)
    Z = R - R.mean(0)
    sd = Z.std(0); sd[sd == 0] = 1.0
    Z = Z / sd
    n = Z.shape[1]
    # seed with the highest-variance strategy
    chosen = [int(np.argmax(R.std(0)))]
    maxabs = np.abs(Z.T @ Z[: chosen[0]]) / Z.shape[0]
    while len(chosen) < k:
        maxabs[chosen] = np.inf
        nxt = int(np.argmin(maxabs))
        chosen.append(nxt)
        c = np.abs(Z.T @ Z[: nxt]) / Z.shape[0]
        maxabs = np.maximum(maxabs, c)
    cols = m.columns[sorted(chosen)]
    return m[cols]


# ============================================================================ #
# Walk-forward engine
# ============================================================================ #
ALLOCATORS = ["1/N", "inv_var", "min_var_raw", "mean_var_raw",
              "hrp", "hrp_denoise", "nco", "nco_denoise_detone", "tic_nco"]


def _build_cov(ret_is, method, classmap=None, members=None, use_tic=False):
    """Return (cov_for_allocator, diag_cond_number, n_facts) for the named pre-processing.
    `ret_is` is an IS return matrix (T_is x N). All causal (IS only)."""
    cov = np.cov(ret_is, rowvar=False)
    n = cov.shape[0]
    T = ret_is.shape[0]
    q = T / float(n)
    nf = n
    if method == "raw":
        return cov, cond_number(cov), nf
    corr, std = _corr_from_cov(cov)
    if method in ("denoise", "denoise_detone", "tic"):
        if q > 1.0:
            corr_d, nf, _ = denoise_corr(corr, q)
        else:
            corr_d = corr  # MP needs T>N; fall back to raw corr when under-determined
        if method == "denoise_detone":
            corr_d = detone_corr(corr_d, n_market=1)
        if method == "tic" and use_tic and members is not None and classmap is not None:
            corr_d = tic_corr(corr_d, members, classmap)
        cov_d = _cov_from_corr(corr_d, std)
        return cov_d, cond_number(cov_d), nf
    raise ValueError(method)


def _weights_for(name, ret_is, classmap=None, members=None, use_numba=True):
    mu = ret_is.mean(0)
    if name == "1/N":
        cov, cond, nf = _build_cov(ret_is, "raw"); return w_equal(cov), cond, nf
    if name == "inv_var":
        cov, cond, nf = _build_cov(ret_is, "raw"); return w_inverse_variance(cov), cond, nf
    if name == "min_var_raw":
        cov, cond, nf = _build_cov(ret_is, "raw"); return w_min_variance(cov), cond, nf
    if name == "mean_var_raw":
        cov, cond, nf = _build_cov(ret_is, "raw"); return w_mean_variance(cov, mu), cond, nf
    if name == "hrp":
        cov, cond, nf = _build_cov(ret_is, "raw"); return w_hrp(cov, use_numba), cond, nf
    if name == "hrp_denoise":
        cov, cond, nf = _build_cov(ret_is, "denoise"); return w_hrp(cov, use_numba), cond, nf
    if name == "nco":
        cov, cond, nf = _build_cov(ret_is, "raw"); return w_nco(cov), cond, nf
    if name == "nco_denoise_detone":
        cov, cond, nf = _build_cov(ret_is, "denoise_detone"); return w_nco(cov), cond, nf
    if name == "tic_nco":
        cov, cond, nf = _build_cov(ret_is, "tic", classmap, members, use_tic=True)
        return w_nco(cov), cond, nf
    raise ValueError(name)


def walk_forward(ret_df, classmap=None, is_win=252, oos_win=63, step=63,
                 allocators=ALLOCATORS, use_numba=True, use_tic=True, label="",
                 vol_target=0.0):
    """Rolling walk-forward. For each fold: estimate weights on [t-is_win, t)
    (IN-SAMPLE), HOLD them over [t, t+oos_win) (OUT-OF-SAMPLE), collect the realised
    daily OOS portfolio returns. NEVER scores weights on the data that built them.

    Returns a dict {allocator: {oos_ret: np.array, hhi: [...], cond: [...], nf: [...]}}.

    vol_target>0: per-column volatility normalisation. Each series is divided by its
    IN-SAMPLE daily std and multiplied by (vol_target/ANN) so every leg enters the
    allocator at a common, comparable risk. This is causal (IS-only scaler applied to
    the OOS block) and fixes the raw-dollar-unit magnitude problem in the strategies
    universe so Sharpe is interpretable; the allocator ranking is unaffected by the
    common scale but cross-leg risk is now apples-to-apples.
    """
    members = list(ret_df.columns)
    R = ret_df.to_numpy(float)
    # require complete rows for the cov estimate; forward-fill returns to 0 (flat day)
    R = np.nan_to_num(R, nan=0.0)
    T, N = R.shape
    allocs = [a for a in allocators if not (a == "tic_nco" and not use_tic)]
    res = {a: dict(oos=[], hhi=[], cond=[], nf=[], dates=[]) for a in allocs}
    starts = list(range(is_win, T - 1, step))
    for t in starts:
        is_block = R[t - is_win:t]
        oos_block = R[t:min(t + oos_win, T)]
        if oos_block.shape[0] < 5:
            continue
        # drop columns that are entirely flat in the IS block (degenerate variance)
        active = is_block.std(0) > 0
        if active.sum() < 4:
            continue
        idx = np.where(active)[0]
        is_a = is_block[: idx]
        oos_a = oos_block[: idx]
        if vol_target > 0:
            # causal per-leg vol scaler from the IS block only, applied to both IS & OOS
            sd = is_a.std(0)
            sd = np.where(sd > 0, sd, 1.0)
            scaler = (vol_target / ANN) / sd
            is_a = is_a * scaler
            oos_a = oos_a * scaler
        cmap = classmap
        mem = [members[i] for i in idx]
        for a in allocs:
            try:
                w, cond, nf = _weights_for(a, is_a, cmap, mem, use_numba)
            except Exception:
                w = np.full(idx.size, 1.0 / idx.size); cond, nf = np.nan, idx.size
            port = oos_a @ w               # realised OOS daily portfolio returns
            res[a]["oos"].append(port)
            res[a]["hhi"].append(hhi(w))
            res[a]["cond"].append(cond)
            res[a]["nf"].append(nf)
    out = {}
    for a in allocs:
        if not res[a]["oos"]:
            continue
        oos = np.concatenate(res[a]["oos"])
        out[a] = dict(
            oos_ret=oos,
            oos_vol_ann=float(oos.std(ddof=1) * ANN),
            oos_sharpe_ann=float(OF.sharpe(oos) * ANN),
            mean_hhi=float(np.mean(res[a]["hhi"])),
            mean_eff_n=float(np.mean([1.0 / h if h > 0 else 1.0 for h in res[a]["hhi"]])),
            mean_cond=float(np.nanmean(res[a]["cond"])),
            n_folds=len(res[a]["oos"]),
            n_oos_days=int(oos.size))
    return out, members


def summarise_market(market_label, wf_out):
    """Build a per-market DataFrame; DSR of each allocator's OOS stream against the
    menu of allocators tried (the multiple-testing correction we actually ran)."""
    from scipy import stats as ss
    if not wf_out:
        return pd.DataFrame(columns=["market", "allocator", "oos_vol_ann"])
    rows = []
    sr_trials = np.array([v["oos_sharpe_ann"] / ANN for v in wf_out.values()])  # per-obs
    for a, v in wf_out.items():
        oos = v["oos_ret"]
        sk = float(ss.skew(oos)) if oos.size > 2 else 0.0
        ku = float(ss.kurtosis(oos, fisher=False)) if oos.size > 2 else 3.0
        d = OF.deflated_sharpe_ratio(OF.sharpe(oos), oos.size, sk, ku, sr_trials)
        rows.append(dict(
            market=market_label, allocator=a,
            oos_vol_ann=round(v["oos_vol_ann"], 4),
            oos_sharpe_ann=round(v["oos_sharpe_ann"], 3),
            dsr=round(float(d["dsr"]), 3),
            mean_eff_n=round(v["mean_eff_n"], 1),
            mean_hhi=round(v["mean_hhi"], 4),
            mean_cond=round(v["mean_cond"], 1),
            n_folds=v["n_folds"], n_oos_days=v["n_oos_days"]))
    return pd.DataFrame(rows).sort_values("oos_vol_ann").reset_index(drop=True)


# ============================================================================ #
# Numba self-test (bit-identity) + profiling
# ============================================================================ #
def _selftest_numba():
    rng = np.random.default_rng(0)
    ok = True
    for n in (8, 17, 40, 120):
        X = rng.standard_normal((400, n))
        cov = np.cov(X, rowvar=False)
        corr, _ = _corr_from_cov(cov)
        link, _ = _linkage_from_corr(corr, "single")
        sort_ix = np.asarray(_quasi_diag(link, n), np.int64)
        w_nb = _hrp_recursive_bisection(cov, sort_ix)
        w_ref = _hrp_recursive_bisection_ref(cov, sort_ix)
        maxdiff = float(np.max(np.abs(w_nb - w_ref)))
        same = np.allclose(w_nb, w_ref, rtol=0, atol=1e-12)
        ok = ok and same
        print(f"  HRP bisection n={n:3d}: max|Δw|={maxdiff:.2e}  bit-identical={same}")
    print(f"  numba available={_HAVE_NUMBA}; overall bit-identical={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=["assets", "strategies"], default="assets")
    ap.add_argument("--n-strat", type=int, default=300, help="strategies per market")
    ap.add_argument("--max-markets", type=int, default=None,
                    help="cap # strategy markets (strategies universe)")
    ap.add_argument("--is-win", type=int, default=252)
    ap.add_argument("--oos-win", type=int, default=63)
    ap.add_argument("--step", type=int, default=63)
    ap.add_argument("--no-tic", action="store_true")
    ap.add_argument("--vol-target", type=float, default=0.0,
                    help="per-leg annualised vol target (causal IS-scaler); "
                         "0 = off (raw units). Use e.g. 0.10 to display clean Sharpes.")
    ap.add_argument("--no-numba", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-core run: few instruments / strategies, short window")
    ap.add_argument("--profile", action="store_true", help="cProfile the run")
    ap.add_argument("--selftest", action="store_true", help="numba bit-identity test only")
    ap.add_argument("--tag", default="", help="suffix for output tables")
    args = ap.parse_args()

    print(f"[verify] numba bit-identity self-test:")
    _selftest_numba()
    if args.selftest:
        return

    t0 = time.time()
    use_numba = not args.no_numba
    use_tic = not args.no_tic
    all_rows = []

    def _run():
        nonlocal all_rows
        if args.universe == "assets":
            print("\n[load] assets universe (crypto + ETFs + FX, daily close-to-close)...")
            ret, classmap = load_assets_universe()
            if args.smoke:
                # tiny but VALID: the 12 instruments with the most data in the
                # recent window; keep the union of dates (walk_forward 0-fills
                # non-trading days, matching the full-run cross-market handling).
                tail = ret.iloc[-450:]
                keep = list(tail.notna().sum().sort_values(ascending=False).index[:12])
                ret = ret[keep].iloc[-400:].dropna(how="all")
            print(f"  panel: {ret.shape[0]} days x {ret.shape[1]} instruments")
            wf, _ = walk_forward(ret, classmap, is_win=args.is_win, oos_win=args.oos_win,
                                 step=args.step, use_numba=use_numba, use_tic=use_tic,
                                 label="assets", vol_target=args.vol_target)
            df = summarise_market("assets", wf)
            print(df.to_string(index=False))
            all_rows.append(df)
        else:
            print(f"\n[load] strategies universe ({args.n_strat}/market)...")
            mm = 2 if args.smoke else args.max_markets
            ns = 40 if args.smoke else args.n_strat
            mats = load_strategies_universe(n_strat=ns, max_markets=mm)
            for mk, m in mats.items():
                if args.smoke:
                    m = m.iloc[-500:]
                print(f"\n  market {mk}: {m.shape[0]} days x {m.shape[1]} strategies")
                wf, _ = walk_forward(m, None, is_win=args.is_win, oos_win=args.oos_win,
                                     step=args.step, use_numba=use_numba,
                                     use_tic=False, label=mk, vol_target=args.vol_target)
                df = summarise_market(mk, wf)
                print(df.to_string(index=False))
                all_rows.append(df)

    if args.profile:
        pr = cProfile.Profile(); pr.enable(); _run(); pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
        print("\n=== cProfile (cumulative top 30) ===\n" + s.getvalue())
    else:
        _run()

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        tag = (args.tag or args.universe) + ("_smoke" if args.smoke else "")
        os.makedirs(f"{PROJ}/tables", exist_ok=True)
        p = f"{PROJ}/tables/portfolio_{tag}.csv"
        out.to_csv(p, index=False)
        print(f"\n[write] {p}  ({len(out)} rows)")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
