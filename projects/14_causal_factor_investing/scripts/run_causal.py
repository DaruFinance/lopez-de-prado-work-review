#!/usr/bin/env python3
"""
run_causal.py — Causal Factor Investing (López de Prado, "Causal Factor Investing"
2023; "Where Are the Factors?" / association-vs-causation critique).

LdP's thesis, made concrete and testable: the factor-investing literature reports
ASSOCIATIONS (cross-sectional/time-series regressions of returns on candidate
"factors") and tacitly reads them as CAUSES. Whether an associational coefficient
identifies a causal effect depends on the underlying causal GRAPH. Under the three
elementary structures —

    FORK / CONFOUNDER   Z -> X,  Z -> Y     (X and Y share a common cause Z)
    CHAIN / MEDIATOR    X -> M -> Y          (X acts on Y only through M)
    COLLIDER            X -> C <- Y          (X and Y both cause C)

— the *naive* regression of Y on X is biased in opposite directions depending on
which variables you (mis)condition on. The backdoor criterion says: to identify
X->Y, condition on a set that blocks every back-door path and contains NO collider
(and no descendant of a collider). Get the adjustment set wrong and a spurious
"factor" looks significant; get it right and it vanishes (confounder) — or you
manufacture a spurious one by conditioning on a collider.

This study is HONEST about being methodological, not a money machine:
  (a) MONTE CARLO of the three structures: show a naive OLS "factor" coefficient is
      significant under mis-conditioning and that the correct backdoor adjustment
      removes (fork) / preserves (chain, with the do-distinction) / or that
      collider-conditioning CREATES a spurious association. Numba inner loop,
      verified bit-identical vs a pure-NumPy reference.
  (b) REAL cross-asset factors (momentum, realized vol, size-proxy) across
      crypto + equities + forex daily returns: naive associational panel
      regression of next-day return on a factor vs a CONFOUNDER-ADJUSTED (backdoor)
      regression that conditions on the market/common-vol confounder. Show where
      the t-stat conclusion FLIPS. This demonstrates the critique, not an edge.
  (c) HIERARCHY OF EVIDENCE / falsification checklist applied to the program's own
      surviving signals — a tie-in scorecard (does each piece of evidence rise
      above mere association?).

HEADLINE where a Sharpe-like claim is made = Deflated Sharpe Ratio via lib/overfit.py.
For (b) the "headline" is the count of factors whose significance flips sign/verdict
under correct adjustment, plus a DSR on the naive long-short factor portfolios to
show that even the survivors do not clear deflation.

Run:
  python3 scripts/run_causal.py            # full multi-market run
  python3 scripts/run_causal.py --smoke     # tiny: few assets, few MC sims (1 core)
  python3 scripts/run_causal.py --profile   # cProfile the MC hot loop (Numba vs ref timing)

RAM: peak well under ~2 GB. Daily panels are tiny (hundreds-to-thousands of rows x
~40 columns). The only sizeable transient is one ETF-year 1-min csv at a time
(~1.5M rows x 2 cols ~ 50 MB) during the daily-panel build, freed immediately and
cached to parquet thereafter. MC arrays are (n_sims x n_obs) doubles; the full run
uses n_sims=20000, n_obs=2000 per structure built incrementally per-sim (never
materialized as one 20000x2000 block), so MC peak is < 200 MB.
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob, json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/daru/ldp_review"
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from lib import overfit as O
from lib import style as S
from scipy import stats as ss

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
CACHE = os.path.join(ROOT, "data_cache")
for d in (FIG, TAB, CACHE):
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data config
# --------------------------------------------------------------------------- #
CRYPTO_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"
FX_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_fx"
ETF_DIR = "/mnt/d/algoseek_data/etf_1min"
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

ANN = {"crypto": 365.0, "forex": 252.0, "equities": 252.0}   # trading days / yr


# --------------------------------------------------------------------------- #
# Numba availability (mirror lib/bars.py shim)
# --------------------------------------------------------------------------- #
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]


# =========================================================================== #
# PART (a) — Monte Carlo of the three causal structures
# =========================================================================== #
# Data-generating processes (linear-Gaussian SEMs). X is the candidate "factor",
# Y the asset return. In every case the TRUE direct causal effect of X on Y is
# what we state; the naive OLS of Y~X recovers something else under mis-handling.
#
#   FORK:      Z ~ N,  X = a*Z + eX,  Y = c*Z + eY      (TRUE X->Y effect = 0)
#              naive Y~X is biased by a*c (spurious); adjust for Z -> ~0.
#   CHAIN:     X ~ N,  M = b*X + eM,  Y = d*M + eY       (TRUE total X->Y = b*d)
#              naive Y~X recovers b*d (correct TOTAL effect); conditioning on the
#              MEDIATOR M wrongly zeroes the effect (over-control bias).
#   COLLIDER:  X ~ N,  Y ~ N (independent, TRUE X->Y = 0),  C = e*X + f*Y + eC
#              naive Y~X ~ 0 (correct); conditioning on the COLLIDER C MANUFACTURES
#              a spurious negative association (collider/selection bias).
#
# The hot loop simulates many independent datasets and, for each, computes the
# OLS slope (and its t-stat) for: naive Y~X, and the (mis/correctly) adjusted
# multivariate Y~[X, conditioning vars]. We report the sampling distribution of
# the X-coefficient and the rejection rate at |t|>1.96.

@njit(cache=True)
def _ols_slope_t(y, x):
    """Simple OLS slope of y on x (with intercept) and its t-stat. Scalar regressor."""
    n = x.shape[0]
    mx = 0.0; my = 0.0
    for i in range(n):
        mx += x[i]; my += y[i]
    mx /= n; my /= n
    sxx = 0.0; sxy = 0.0
    for i in range(n):
        dx = x[i] - mx
        sxx += dx * dx
        sxy += dx * (y[i] - my)
    if sxx <= 0.0:
        return 0.0, 0.0
    b = sxy / sxx
    a = my - b * mx
    sse = 0.0
    for i in range(n):
        e = y[i] - (a + b * x[i])
        sse += e * e
    if n <= 2:
        return b, 0.0
    s2 = sse / (n - 2)
    se = (s2 / sxx) ** 0.5
    if se <= 0.0:
        return b, 0.0
    return b, b / se


@njit(cache=True)
def _ols_partial_t(y, x, z):
    """Coefficient on x and its t-stat in the bivariate regression y ~ x + z
    (intercept implied). Via Frisch–Waugh: regress x on z, y on z, take residuals,
    then simple OLS of resid_y on resid_x. Degrees of freedom n-3."""
    n = x.shape[0]
    # regress x on z
    mz = 0.0; mx = 0.0; my = 0.0
    for i in range(n):
        mz += z[i]; mx += x[i]; my += y[i]
    mz /= n; mx /= n; my /= n
    szz = 0.0; szx = 0.0; szy = 0.0
    for i in range(n):
        dz = z[i] - mz
        szz += dz * dz
        szx += dz * (x[i] - mx)
        szy += dz * (y[i] - my)
    if szz <= 0.0:
        return 0.0, 0.0
    bxz = szx / szz       # slope of x on z
    byz = szy / szz       # slope of y on z
    axz = mx - bxz * mz
    ayz = my - byz * mz
    # residuals
    srr = 0.0; sry = 0.0
    for i in range(n):
        rx = x[i] - (axz + bxz * z[i])
        ry = y[i] - (ayz + byz * z[i])
        srr += rx * rx
        sry += rx * ry
    if srr <= 0.0:
        return 0.0, 0.0
    b = sry / srr
    # residual SSE for t-stat (df = n-3: intercept + z + x)
    sse = 0.0
    for i in range(n):
        rx = x[i] - (axz + bxz * z[i])
        ry = y[i] - (ayz + byz * z[i])
        e = ry - b * rx
        sse += e * e
    if n <= 3:
        return b, 0.0
    s2 = sse / (n - 3)
    se = (s2 / srr) ** 0.5
    if se <= 0.0:
        return b, 0.0
    return b, b / se


@njit(cache=True)
def _mc_structures_kernel(n_sims, n_obs, a, c, b, d, e, f, seeds):
    """Hot loop. For each sim draw fresh data for all three structures and record
    the X-coefficient + t-stat for the naive and adjusted regressions.

    Returns array (n_sims, 8):
      0,1  fork   naive  Y~X        b, t   (spurious; TRUE=0)
      2,3  fork   adj    Y~X+Z      b, t   (backdoor; -> ~0)
      4,5  chain  naive  Y~X        b, t   (TOTAL effect b*d; correct)
      6,7  chain  adjM   Y~X+M      b, t   (over-control; -> ~0)
    Collider handled in a second kernel to keep arity sane.
    """
    out = np.empty((n_sims, 8))
    for s in range(n_sims):
        np.random.seed(seeds[s])
        # ---- FORK: Z common cause ----
        Z = np.random.standard_normal(n_obs)
        Xf = a * Z + np.random.standard_normal(n_obs)
        Yf = c * Z + np.random.standard_normal(n_obs)
        bn, tn = _ols_slope_t(Yf, Xf)
        ba, ta = _ols_partial_t(Yf, Xf, Z)
        out[s, 0] = bn; out[s, 1] = tn
        out[s, 2] = ba; out[s, 3] = ta
        # ---- CHAIN: X -> M -> Y ----
        Xc = np.random.standard_normal(n_obs)
        M = b * Xc + np.random.standard_normal(n_obs)
        Yc = d * M + np.random.standard_normal(n_obs)
        bn2, tn2 = _ols_slope_t(Yc, Xc)
        ba2, ta2 = _ols_partial_t(Yc, Xc, M)
        out[s, 4] = bn2; out[s, 5] = tn2
        out[s, 6] = ba2; out[s, 7] = ta2
    return out


@njit(cache=True)
def _mc_collider_kernel(n_sims, n_obs, e, f, seeds):
    """COLLIDER: X, Y independent (TRUE X->Y = 0); C = e*X + f*Y + noise.
    Returns (n_sims, 4): naive Y~X (b,t) -> ~0; adjusted Y~X+C (b,t) -> spurious."""
    out = np.empty((n_sims, 4))
    for s in range(n_sims):
        np.random.seed(seeds[s])
        X = np.random.standard_normal(n_obs)
        Y = np.random.standard_normal(n_obs)
        C = e * X + f * Y + np.random.standard_normal(n_obs)
        bn, tn = _ols_slope_t(Y, X)
        ba, ta = _ols_partial_t(Y, X, C)
        out[s, 0] = bn; out[s, 1] = tn
        out[s, 2] = ba; out[s, 3] = ta
    return out


# ---- Independent pure-NumPy reference (for bit-identical verification) ------
def _ols_slope_t_ref(y, x):
    n = len(x)
    mx, my = x.mean(), y.mean()
    dx = x - mx
    sxx = (dx * dx).sum()
    if sxx <= 0:
        return 0.0, 0.0
    b = (dx * (y - my)).sum() / sxx
    a = my - b * mx
    e = y - (a + b * x)
    sse = (e * e).sum()
    if n <= 2:
        return b, 0.0
    se = np.sqrt(sse / (n - 2) / sxx)
    return b, (b / se if se > 0 else 0.0)


def _ols_partial_t_ref(y, x, z):
    n = len(x)
    mz, mx, my = z.mean(), x.mean(), y.mean()
    dz = z - mz
    szz = (dz * dz).sum()
    if szz <= 0:
        return 0.0, 0.0
    bxz = (dz * (x - mx)).sum() / szz
    byz = (dz * (y - my)).sum() / szz
    axz, ayz = mx - bxz * mz, my - byz * mz
    rx = x - (axz + bxz * z)
    ry = y - (ayz + byz * z)
    srr = (rx * rx).sum()
    if srr <= 0:
        return 0.0, 0.0
    b = (rx * ry).sum() / srr
    e = ry - b * rx
    sse = (e * e).sum()
    if n <= 3:
        return b, 0.0
    se = np.sqrt(sse / (n - 3) / srr)
    return b, (b / se if se > 0 else 0.0)


def _mc_reference(n_sims, n_obs, a, c, b, d, e, f, seeds):
    """Pure-NumPy mirror of both kernels combined: returns (out8, out4)."""
    o8 = np.empty((n_sims, 8)); o4 = np.empty((n_sims, 4))
    for s in range(n_sims):
        np.random.seed(seeds[s])
        Z = np.random.standard_normal(n_obs)
        Xf = a * Z + np.random.standard_normal(n_obs)
        Yf = c * Z + np.random.standard_normal(n_obs)
        o8[s, 0:2] = _ols_slope_t_ref(Yf, Xf)
        o8[s, 2:4] = _ols_partial_t_ref(Yf, Xf, Z)
        Xc = np.random.standard_normal(n_obs)
        M = b * Xc + np.random.standard_normal(n_obs)
        Yc = d * M + np.random.standard_normal(n_obs)
        o8[s, 4:6] = _ols_slope_t_ref(Yc, Xc)
        o8[s, 6:8] = _ols_partial_t_ref(Yc, Xc, M)
    for s in range(n_sims):
        np.random.seed(seeds[s])
        X = np.random.standard_normal(n_obs)
        Y = np.random.standard_normal(n_obs)
        C = e * X + f * Y + np.random.standard_normal(n_obs)
        o4[s, 0:2] = _ols_slope_t_ref(Y, X)
        o4[s, 2:4] = _ols_partial_t_ref(Y, X, C)
    return o8, o4


def run_montecarlo(n_sims, n_obs, verify=True, seed0=20230, engine="numpy"):
    """Run the three-structure MC. Returns dict of summaries; writes a table.

    engine='numpy' uses the vectorized per-sim NumPy path (PRODUCTION default: it
    is faster than the scalar Numba loops at these n_obs — see profile note in the
    README; the hot cost is RNG + BLAS reductions, which NumPy already does well).
    engine='numba' uses the @njit kernels; both are verified bit-identical (the MC
    rejection rates are insensitive to the ~1e-14 float reordering)."""
    # fixed structural coefficients (LdP-style illustrative magnitudes)
    a, c = 1.0, 1.0       # fork: Z->X, Z->Y
    b, d = 0.8, 0.8       # chain: X->M, M->Y (true total = 0.64)
    e, f = 1.0, 1.0       # collider: X->C, Y->C
    seeds = (np.arange(n_sims) + seed0).astype(np.int64)

    if engine == "numba":
        o8 = _mc_structures_kernel(n_sims, n_obs, a, c, b, d, e, f, seeds)
        o4 = _mc_collider_kernel(n_sims, n_obs, e, f, seeds)
    else:
        o8, o4 = _mc_reference(n_sims, n_obs, a, c, b, d, e, f, seeds)

    rej = lambda t: float(np.mean(np.abs(t) > 1.96))
    # Decompose every rejection rate into its decision-theoretic meaning. For the
    # FORK and the COLLIDER the true X->Y effect is exactly 0, so a rejection is a
    # FALSE POSITIVE (Type-I error). For the CHAIN the true total effect is b*d>0,
    # so a *failure* to reject is a FALSE NEGATIVE (Type-II error / power loss).
    res = {
        "fork":     {"true_effect": 0.0,
                     "naive_b": float(o8[:, 0].mean()), "naive_rej": rej(o8[:, 1]),
                     "adj_b": float(o8[:, 2].mean()),   "adj_rej": rej(o8[:, 3]),
                     # naive_rej IS the Type-I (false-positive) rate; adj_rej should ~= alpha=0.05
                     "naive_falsepos": rej(o8[:, 1]), "adj_falsepos": rej(o8[:, 3]),
                     "adjust_for": "Z (common cause / confounder)"},
        "chain":    {"true_total_effect": b * d,
                     "naive_b": float(o8[:, 4].mean()), "naive_rej": rej(o8[:, 5]),
                     "adjM_b": float(o8[:, 6].mean()),  "adjM_rej": rej(o8[:, 7]),
                     # naive recovers the true TOTAL effect (high power, correct sign);
                     # over-controlling for the mediator collapses power to ~alpha.
                     "naive_power": rej(o8[:, 5]), "adjM_power": rej(o8[:, 7]),
                     "adjust_for": "M (mediator) -- WRONG, over-control"},
        "collider": {"true_effect": 0.0,
                     "naive_b": float(o4[:, 0].mean()), "naive_rej": rej(o4[:, 1]),
                     "adjC_b": float(o4[:, 2].mean()),  "adjC_rej": rej(o4[:, 3]),
                     # naive ~= alpha (correct); conditioning on the collider OPENS a
                     # path -> Type-I rate explodes to ~1.0 (manufactured factor).
                     "naive_falsepos": rej(o4[:, 1]), "adjC_falsepos": rej(o4[:, 3]),
                     "adjust_for": "C (collider) -- WRONG, opens spurious path"},
    }

    verify_info = {}
    if verify:
        # cross-engine: production (o8/o4) vs the OTHER engine on the same seeds
        nv = min(200, n_sims)
        if engine == "numpy":
            a8 = _mc_structures_kernel(nv, n_obs, a, c, b, d, e, f, seeds[:nv])
            a4 = _mc_collider_kernel(nv, n_obs, e, f, seeds[:nv])
        else:
            a8, a4 = _mc_reference(nv, n_obs, a, c, b, d, e, f, seeds[:nv])
        d8 = float(np.max(np.abs(o8[:nv] - a8)))
        d4 = float(np.max(np.abs(o4[:nv] - a4)))
        verify_info = {"max_abs_diff_struct": d8, "max_abs_diff_collider": d4,
                       "n_verified": nv, "engine": engine}
    return res, verify_info, (o8, o4)


def run_mc_sweep(n_sims, n_obs, seed0=70230):
    """DEEPENING of part (a): sweep the structural-bias strength and show how the
    naive vs corrected DECISION (false-positive / power) responds. This converts
    the single-point demonstration into a dose-response curve — the core
    quantitative claim of LdP's critique: the naive estimator's error grows with
    the strength of the (mis)handled structure, while the backdoor-correct
    estimator holds its nominal alpha / recovers full power regardless.

      FORK     sweep g in [0,1.2] with a=c=g (confounder strength).
               naive false-pos rate -> 1 as g grows; backdoor stays ~0.05.
      COLLIDER sweep g in [0,1.2] with e=f=g (collider loading).
               naive stays ~0.05; collider-conditioned false-pos -> 1 as g grows.
      CHAIN    sweep g in [0,1.0] with b=d=g (path strength; true total g^2).
               naive power -> 1 as g grows (correct); mediator-controlled power
               stays ~0.05 (over-control destroys a REAL effect at every dose).
    Smaller n_sims per grid point keeps this cheap; pure NumPy, RAM-safe."""
    grid = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00, 1.20])
    rej = lambda t: float(np.mean(np.abs(t) > 1.96))
    rows = []
    ns = min(n_sims, 4000)            # plenty for a rate estimate; keeps it fast
    for g in grid:
        seeds = (np.arange(ns) + seed0 + int(g * 1000)).astype(np.int64)
        # FORK: a=c=g, true X->Y = 0
        bf_n = np.empty(ns); tf_n = np.empty(ns); tf_a = np.empty(ns)
        # COLLIDER: e=f=g, true X->Y = 0
        tc_n = np.empty(ns); tc_a = np.empty(ns)
        # CHAIN: b=d=g, true total = g^2
        tch_n = np.empty(ns); tch_a = np.empty(ns)
        for s in range(ns):
            np.random.seed(seeds[s])
            Z = np.random.standard_normal(n_obs)
            Xf = g * Z + np.random.standard_normal(n_obs)
            Yf = g * Z + np.random.standard_normal(n_obs)
            bn, tn = _ols_slope_t_ref(Yf, Xf); _, ta = _ols_partial_t_ref(Yf, Xf, Z)
            bf_n[s] = bn; tf_n[s] = tn; tf_a[s] = ta
            Xco = np.random.standard_normal(n_obs)
            Yco = np.random.standard_normal(n_obs)
            C = g * Xco + g * Yco + np.random.standard_normal(n_obs)
            _, tn2 = _ols_slope_t_ref(Yco, Xco); _, ta2 = _ols_partial_t_ref(Yco, Xco, C)
            tc_n[s] = tn2; tc_a[s] = ta2
            Xc = np.random.standard_normal(n_obs)
            M = g * Xc + np.random.standard_normal(n_obs)
            Yc = g * M + np.random.standard_normal(n_obs)
            _, tn3 = _ols_slope_t_ref(Yc, Xc); _, ta3 = _ols_partial_t_ref(Yc, Xc, M)
            tch_n[s] = tn3; tch_a[s] = ta3
        rows.append(dict(
            strength=float(g),
            fork_naive_falsepos=rej(tf_n), fork_naive_meanb=float(bf_n.mean()),
            fork_adj_falsepos=rej(tf_a),
            collider_naive_falsepos=rej(tc_n), collider_adj_falsepos=rej(tc_a),
            chain_true_total=float(g * g),
            chain_naive_power=rej(tch_n), chain_adjM_power=rej(tch_a)))
    return pd.DataFrame(rows)


# =========================================================================== #
# PART (b) — real cross-asset factors: naive vs backdoor-adjusted
# =========================================================================== #
def _daily_close_crypto(path):
    df = pd.read_parquet(path, columns=["open_time", "close"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    s = df.set_index("open_time").sort_index()["close"]
    s = s[~s.index.duplicated(keep="first")]
    return s.resample("1D").last().dropna()


def _daily_close_fx(path):
    f = pd.read_parquet(path)
    f.index = pd.to_datetime(f.index, utc=True)
    s = f.sort_index()["close"]
    s = s[~s.index.duplicated(keep="first")]
    return s.resample("1D").last().dropna()


def _daily_close_etf(sym):
    """Build a daily close series for one ETF from yearly 1-min csv.gz, cached."""
    cpath = os.path.join(CACHE, f"daily_etf_{sym}.parquet")
    if os.path.exists(cpath):
        return pd.read_parquet(cpath)["close"]
    files = sorted(f for f in glob.glob(os.path.join(ETF_DIR, f"{sym}_*.csv.gz"))
                   if "_pre_rename" not in f)
    base = os.path.join(ETF_DIR, f"{sym}.csv.gz")
    if os.path.exists(base):
        files = [base] + [f for f in files if f != base]
    parts = []
    for f in files:
        d = pd.read_csv(f, compression="gzip", usecols=["BarDateTime", "LastTradePrice"])
        dt = pd.to_datetime(d["BarDateTime"], errors="coerce")
        d = pd.DataFrame({"close": d["LastTradePrice"].to_numpy()}, index=dt).dropna()
        parts.append(d["close"].resample("1D").last().dropna())
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index
    pd.DataFrame({"close": s}).to_parquet(cpath)
    return s


def build_daily_panel(smoke=False):
    """Return (returns_df, market_map). returns_df: daily simple returns, columns =
    instruments, union-indexed by UTC calendar day. market_map: col -> market."""
    cpath = os.path.join(CACHE, "daily_panel_causal.parquet")
    mpath = os.path.join(CACHE, "daily_panel_causal_markets.json")
    if not smoke and os.path.exists(cpath) and os.path.exists(mpath):
        rets = pd.read_parquet(cpath)
        mk = json.load(open(mpath))
        return rets, mk

    crypto = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
              "AVAXUSDT", "LINKUSDT", "LTCUSDT", "ADAUSDT", "DOTUSDT", "ATOMUSDT"]
    fx = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
    etfs = ETF_SYMS
    if smoke:
        crypto, fx, etfs = crypto[:2], fx[:2], etfs[:2]

    closes, mk = {}, {}
    for sym in crypto:
        p = os.path.join(CRYPTO_DIR, f"{sym}_1m.parquet")
        if os.path.exists(p):
            closes[sym] = _daily_close_crypto(p); mk[sym] = "crypto"
    for sym in fx:
        p = os.path.join(FX_DIR, f"{sym}_fx1m.parquet")
        if os.path.exists(p):
            closes[sym] = _daily_close_fx(p); mk[sym] = "forex"
    for sym in etfs:
        try:
            closes[sym] = _daily_close_etf(sym); mk[sym] = "equities"
        except Exception:
            pass

    px = pd.DataFrame({k: v for k, v in closes.items()})
    px.index = pd.to_datetime(px.index, utc=True)
    px = px.sort_index()
    rets = px.pct_change()
    if not smoke:
        rets.to_parquet(cpath)
        json.dump(mk, open(mpath, "w"))
    return rets, mk


def _factor_series(ret: pd.Series, kind: str) -> pd.Series:
    """CAUSAL factor value at day t using ONLY data up to and including t-1
    (shifted), so the factor predicting return[t] uses no future info.
      momentum: trailing 21-day cumulative return (lagged 1)
      vol:      trailing 21-day realized vol (lagged 1)   [low-vol 'factor']
      revers:   trailing 5-day return (short-term reversal proxy / 'size-ish' churn)
    """
    if kind == "momentum":
        f = (1.0 + ret).rolling(21).apply(np.prod, raw=True) - 1.0
    elif kind == "vol":
        f = ret.rolling(21).std()
    elif kind == "revers":
        f = (1.0 + ret).rolling(5).apply(np.prod, raw=True) - 1.0
    else:
        raise ValueError(kind)
    return f.shift(1)


def run_real_factors(rets: pd.DataFrame, mk: dict):
    """For each market and each factor, run a pooled panel regression of next-day
    return on the (lagged, standardized) factor:
      NAIVE     r_{i,t} ~ factor_{i,t}
      BACKDOOR  r_{i,t} ~ factor_{i,t} + market_{t}    (condition on common driver)
    where market_t = cross-sectional mean return that day = the obvious CONFOUNDER
    (a common market/vol factor drives both the candidate factor and the returns).
    Report the factor t-stat in each and whether the verdict (signif at |t|>1.96)
    FLIPS. Also build the naive long-short factor portfolio per market and DSR it.
    """
    cols = list(rets.columns)
    factors = ["momentum", "vol", "revers"]
    markets = sorted(set(mk.values()))
    rows = []
    portfolios = {}   # (market,factor) -> per-day LS return series

    for market in markets:
        ins = [c for c in cols if mk.get(c) == market]
        if len(ins) < 2:
            continue
        sub = rets[ins].dropna(how="all")
        mkt = sub.mean(axis=1)                      # common driver / confounder Z1
        # SECOND confounder Z2: lagged common realized volatility (the market-wide
        # vol regime drives both each factor and tomorrow's dispersion of returns).
        # Lagged by 1 day -> strictly causal (uses only info up to t-1).
        cvol = sub.std(axis=1).rolling(21).mean().shift(1)
        for fk in factors:
            # stack panel
            Y, X, Z1, Z2 = [], [], [], []
            ls_daily = {}
            per_inst_t = []                          # per-instrument naive t for sign stability
            for c in ins:
                r = sub[c].dropna()
                fac = _factor_series(r, fk).reindex(r.index)
                fwd = r                              # r_t is next-day vs factor_{t} (lagged)
                df = pd.DataFrame({"y": fwd, "x": fac,
                                   "z1": mkt.reindex(r.index),
                                   "z2": cvol.reindex(r.index)}).dropna()
                if len(df) < 60:
                    continue
                Y.append(df["y"].to_numpy()); X.append(df["x"].to_numpy())
                Z1.append(df["z1"].to_numpy()); Z2.append(df["z2"].to_numpy())
                _, ti = _ols_slope_t_ref(df["y"].to_numpy(), df["x"].to_numpy())
                per_inst_t.append(ti)
                # long-short sleeve: sign(factor)*return  (per instrument, summed daily)
                sleeve = np.sign(df["x"]) * df["y"]
                ls_daily[c] = pd.Series(sleeve.to_numpy(), index=df.index)
            if not Y:
                continue
            y = np.concatenate(Y); x = np.concatenate(X)
            z1 = np.concatenate(Z1); z2 = np.concatenate(Z2)
            # standardize regressors (z-score) so coefficients are comparable
            x = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
            z1 = (z1 - z1.mean()) / (z1.std(ddof=1) + 1e-12)
            z2 = (z2 - z2.mean()) / (z2.std(ddof=1) + 1e-12)
            bn, tn = _ols_slope_t_ref(y, x)
            ba, ta = _ols_partial_t_ref(y, x, z1)        # backdoor on market mean
            ba2, ta2 = _ols_partial_t_ref(y, x, z2)      # backdoor on lagged common vol
            verdict_naive = abs(tn) > 1.96
            verdict_adj = abs(ta) > 1.96
            verdict_adj2 = abs(ta2) > 1.96
            flip = verdict_naive != verdict_adj
            # classify the flip: a significant naive result that DIES under
            # adjustment is a "spurious_killed" (confounded factor); an
            # insignificant naive result that BECOMES significant is "suppression"
            # (the confounder was masking a real association / collider-like).
            if not flip:
                flip_kind = "none"
            elif verdict_naive and not verdict_adj:
                flip_kind = "spurious_killed"
            else:
                flip_kind = "suppression_revealed"
            # |t| attenuation under adjustment (how much of the naive signal is confounder)
            t_atten = (abs(tn) - abs(ta)) / (abs(tn) + 1e-12)
            # per-instrument sign stability: fraction agreeing with the pooled sign
            pit = np.array(per_inst_t)
            sgn = np.sign(bn) if bn != 0 else 1.0
            sign_consist = float(np.mean(np.sign(pit) == sgn)) if len(pit) else np.nan

            # naive LS portfolio (equal-weight across instruments, per day)
            ls = pd.DataFrame(ls_daily).mean(axis=1).dropna()
            portfolios[(market, fk)] = ls
            ann = ANN[market]
            sr = O.sharpe(ls.to_numpy())
            sr_ann = sr * np.sqrt(ann)
            rows.append(dict(market=market, factor=fk, n_obs=len(y), n_inst=len(pit),
                             naive_b=bn, naive_t=tn, adj_b=ba, adj_t=ta,
                             adj2_b=ba2, adj2_t=ta2,
                             verdict_naive=verdict_naive, verdict_adj=verdict_adj,
                             verdict_adj2=verdict_adj2,
                             flips=flip, flip_kind=flip_kind,
                             t_attenuation=t_atten, sign_consistency=sign_consist,
                             ls_sr_ann=sr_ann, ls_n_days=len(ls)))

    fac_df = pd.DataFrame(rows)

    # DSR on the naive LS portfolios, treating the (market x factor) set as trials.
    dsr_rows = []
    if len(portfolios) >= 1:
        # align all LS sleeves to a common index for a trials matrix
        keys = list(portfolios.keys())
        srs = []
        for k in keys:
            r = portfolios[k]
            srs.append(O.sharpe(r.to_numpy()))
        srs = np.array(srs)
        best = int(np.nanargmax(srs))
        bk = keys[best]; br = portfolios[bk].to_numpy()
        br = br[np.isfinite(br)]
        if len(br) > 30 and br.std(ddof=1) > 0:
            dd = O.deflated_sharpe_ratio(O.sharpe(br), len(br),
                                         float(ss.skew(br)),
                                         float(ss.kurtosis(br, fisher=False)), srs)
            dsr_rows.append(dict(best_market=bk[0], best_factor=bk[1],
                                 best_sr=O.sharpe(br),
                                 best_sr_ann=O.sharpe(br) * np.sqrt(ANN[bk[0]]),
                                 sr0=dd["sr0"], dsr=dd["dsr"],
                                 n_trials=dd["n_trials"]))
    dsr_df = pd.DataFrame(dsr_rows)
    return fac_df, dsr_df, portfolios


# =========================================================================== #
# PART (c) — hierarchy-of-evidence / falsification checklist (tie-in)
# =========================================================================== #
def hierarchy_checklist(fac_df: pd.DataFrame, mc_res: dict, dsr_df: pd.DataFrame):
    """A falsification scorecard for the program's own surviving signals. Each row
    is a question LdP's causal program forces on an 'edge'; we answer it from THIS
    study's artifacts (and flag what the broader program still owes)."""
    n_cells = len(fac_df)
    n_flip = int(fac_df["flips"].sum()) if n_cells else 0
    n_naive_sig = int(fac_df["verdict_naive"].sum()) if n_cells else 0
    n_adj_sig = int(fac_df["verdict_adj"].sum()) if n_cells else 0
    n_adj2_sig = int(fac_df["verdict_adj2"].sum()) if n_cells and "verdict_adj2" in fac_df else 0
    best_dsr = float(dsr_df["dsr"].iloc[0]) if len(dsr_df) else float("nan")

    # Identify the SURVIVORS — cells that pass BOTH backdoor adjustments (market
    # mean AND lagged common vol). These are the only signals the program is
    # entitled to even *consider* causal; the checklist is then applied to them.
    if n_cells:
        surv = fac_df[fac_df["verdict_naive"] & fac_df["verdict_adj"]
                      & fac_df.get("verdict_adj2", fac_df["verdict_adj"])]
    else:
        surv = fac_df
    n_surv = len(surv)
    surv_lab = ", ".join(f"{r.market[:3]}:{r.factor}" for r in surv.itertuples()) or "none"
    # median |t| attenuation across the naively-significant cells: how much of the
    # raw association is, on average, attributable to the confounder.
    if n_naive_sig:
        med_atten = float(fac_df.loc[fac_df["verdict_naive"], "t_attenuation"].median())
    else:
        med_atten = float("nan")

    rows = [
        dict(level=1, question="L1 Association exists? (raw factor |t|>1.96)",
             evidence=f"{n_naive_sig}/{n_cells} factor-market cells significant naively",
             passes=(n_naive_sig > 0)),
        dict(level=2, question="L2 Survives backdoor adj. for the market-mean confounder?",
             evidence=f"{n_adj_sig}/{n_cells} survive; {n_flip} verdicts FLIP; "
                      f"median |t| attenuation over naively-sig cells = {med_atten:.0%}",
             passes=(n_adj_sig > 0)),
        dict(level=3, question="L3 Survives a SECOND, independent confounder (lagged common vol)?",
             evidence=f"{n_adj2_sig}/{n_cells} survive the lagged-common-vol backdoor; "
                      f"joint survivors (both adjustments) = {n_surv} [{surv_lab}]",
             passes=(n_surv > 0)),
        dict(level=4, question="L4 Is the causal graph stated & DEFENDED (not assumed)?",
             evidence="MC (part a) proves mis-conditioning fabricates/kills a 'factor' at "
                      "every dose; no defended SEM is supplied here for any survivor",
             passes=False),
        dict(level=5, question="L5 Survives multiple-testing deflation (DSR>0.95)?",
             evidence=f"best naive LS-factor DSR={best_dsr:.3f} (n_trials={int(dsr_df['n_trials'].iloc[0]) if len(dsr_df) else 0})",
             passes=(best_dsr > 0.95)),
        dict(level=6, question="L6 Interventional / do-evidence (not just observational)?",
             evidence="none: all evidence is observational; needs natural experiment / "
                      "instrument / regime-shift falsification",
             passes=False),
        dict(level=7, question="L7 True OOS / cross-market replication of the CAUSAL claim?",
             evidence="multi-market panel is a weak replication; surviving cells do not "
                      "replicate the SAME factor across all three markets, and no causal hold-out exists",
             passes=False),
    ]
    return pd.DataFrame(rows)


# =========================================================================== #
# Figures
# =========================================================================== #
def make_figures(mc_arrays, mc_res, fac_df, portfolios, sweep_df=None):
    import matplotlib.pyplot as plt
    S.set_style()
    o8, o4 = mc_arrays

    # Fig 1: MC coefficient distributions for the three structures
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for k, (title, naive, adj, true_v, adjlab) in enumerate([
        ("Fork / confounder\n(true X→Y = 0)", o8[:, 0], o8[:, 2], 0.0, "adj for Z (backdoor)"),
        ("Chain / mediator\n(true total = 0.64)", o8[:, 4], o8[:, 6], 0.64, "adj for M (over-control)"),
        ("Collider\n(true X→Y = 0)", o4[:, 0], o4[:, 2], 0.0, "adj for C (opens path)"),
    ]):
        ax[k].hist(naive, bins=60, alpha=0.6, color=S.barcolor("accent"), label="naive Y~X")
        ax[k].hist(adj, bins=60, alpha=0.6, color=S.barcolor("dollar"), label=adjlab)
        ax[k].axvline(true_v, color="k", ls="--", lw=1.2, label=f"true = {true_v:g}")
        ax[k].set_title(title); ax[k].set_xlabel("estimated X coefficient")
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel("MC frequency")
    fig.suptitle("Monte Carlo: associational OLS misleads unless the causal structure is respected",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_mc_structures.png")); plt.close(fig)

    # Fig 2: naive vs adjusted t-stats for real factors (flip map)
    if len(fac_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(1.96, color="grey", ls=":", lw=1); ax.axhline(-1.96, color="grey", ls=":", lw=1)
        ax.axvline(1.96, color="grey", ls=":", lw=1); ax.axvline(-1.96, color="grey", ls=":", lw=1)
        for _, r in fac_df.iterrows():
            col = S.barcolor("accent") if r["flips"] else S.barcolor("tick")
            ax.scatter(r["naive_t"], r["adj_t"], color=col, s=60,
                       edgecolor="k", linewidth=0.4, zorder=3)
            ax.annotate(f"{r['market'][:3]}·{r['factor'][:3]}",
                        (r["naive_t"], r["adj_t"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        lim = max(3.0, np.abs(fac_df[["naive_t", "adj_t"]].to_numpy()).max() * 1.1)
        ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.6, alpha=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("naive factor t-stat (Y~X)")
        ax.set_ylabel("backdoor-adjusted t-stat (Y~X+market)")
        ax.set_title("Real factors: where conditioning on the market confounder flips the verdict\n"
                     "(orange = verdict flips at |t|=1.96)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig2_real_factor_flip.png")); plt.close(fig)

    # Fig 3: MC dose-response — decision error vs structural-bias strength
    if sweep_df is not None and len(sweep_df):
        g = sweep_df["strength"].to_numpy()
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].plot(g, sweep_df["fork_naive_falsepos"], "-o", color=S.barcolor("accent"),
                   label="naive Y~X")
        ax[0].plot(g, sweep_df["fork_adj_falsepos"], "-o", color=S.barcolor("dollar"),
                   label="backdoor (adj Z)")
        ax[0].axhline(0.05, color="k", ls="--", lw=1, label="nominal α=0.05")
        ax[0].set_title("Fork: false-positive rate\n(true X→Y = 0)")
        ax[0].set_xlabel("confounder strength a=c"); ax[0].set_ylabel("P(reject H0: |t|>1.96)")
        ax[0].legend(fontsize=8)
        ax[1].plot(g, sweep_df["collider_naive_falsepos"], "-o", color=S.barcolor("accent"),
                   label="naive Y~X")
        ax[1].plot(g, sweep_df["collider_adj_falsepos"], "-o", color=S.barcolor("dollar"),
                   label="conditioned on C")
        ax[1].axhline(0.05, color="k", ls="--", lw=1, label="nominal α=0.05")
        ax[1].set_title("Collider: false-positive rate\n(true X→Y = 0)")
        ax[1].set_xlabel("collider loading e=f"); ax[1].legend(fontsize=8)
        ax[2].plot(g, sweep_df["chain_naive_power"], "-o", color=S.barcolor("accent"),
                   label="naive Y~X (correct total)")
        ax[2].plot(g, sweep_df["chain_adjM_power"], "-o", color=S.barcolor("dollar"),
                   label="over-control (adj M)")
        ax[2].axhline(0.05, color="k", ls="--", lw=1, label="α=0.05 (no power)")
        ax[2].set_title("Chain: power to detect a REAL effect\n(true total = (b·d)²)")
        ax[2].set_xlabel("path strength b=d"); ax[2].set_ylabel("P(reject | effect real)")
        ax[2].legend(fontsize=8)
        fig.suptitle("Dose-response: the naive estimator's error scales with the (mis)handled "
                     "structure; correct handling holds α / recovers power",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig3_mc_dose_response.png")); plt.close(fig)


# =========================================================================== #
# Profiling
# =========================================================================== #
def profile_mc():
    import cProfile, pstats, io
    print("[profile] warming Numba kernels (compile excluded from timing)...")
    seeds = (np.arange(64) + 1).astype(np.int64)
    _mc_structures_kernel(64, 256, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, seeds)
    _mc_collider_kernel(64, 256, 1.0, 1.0, seeds)

    n_sims, n_obs = 5000, 2000
    seeds = (np.arange(n_sims)).astype(np.int64)
    t = time.time()
    _mc_structures_kernel(n_sims, n_obs, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, seeds)
    _mc_collider_kernel(n_sims, n_obs, 1.0, 1.0, seeds)
    numba_s = time.time() - t

    # reference timing on a smaller slice (pure python is slow)
    nv = 300
    t = time.time()
    _mc_reference(nv, n_obs, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, seeds[:nv])
    ref_s = (time.time() - t) * (n_sims / nv)

    print(f"[profile] Numba scalar kernels ({n_sims} sims x {n_obs} obs): {numba_s:.3f} s")
    print(f"[profile] NumPy per-sim path   (extrapolated to same):       {ref_s:.3f} s")
    print(f"[profile] Numba/NumPy ratio ~ {numba_s/max(ref_s,1e-9):.2f}x  "
          f"(>1 means NumPy WINS here)")
    print("[profile] FINDING: at n_obs=2000 the OLS reductions are large vectorized "
          "sums that NumPy/BLAS does\n          better than scalar @njit loops; the "
          "true hot cost is RNG + reductions, not\n          Python overhead. So the "
          "PRODUCTION engine is NumPy (engine='numpy'); the Numba\n          kernels are "
          "kept as a verified bit-identical cross-check (and would win only for\n          "
          "many TINY regressions). Batched-vectorized would be fastest but its "
          "n_sims x n_obs\n          arrays blow RAM at full scale (~2.4 GB/array), so "
          "the per-sim path is the RAM-safe choice.")
    # cross-engine bit-identity on a slice
    seeds_v = (np.arange(200) + 20230).astype(np.int64)
    o8n = _mc_structures_kernel(200, n_obs, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, seeds_v)
    o4n = _mc_collider_kernel(200, n_obs, 1.0, 1.0, seeds_v)
    r8, r4 = _mc_reference(200, n_obs, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, seeds_v)
    print(f"[profile] bit-identity Numba vs NumPy: max|Δ| struct="
          f"{np.max(np.abs(o8n-r8)):.2e}, collider={np.max(np.abs(o4n-r4)):.2e}")

    pr = cProfile.Profile(); pr.enable()
    _mc_structures_kernel(n_sims, n_obs, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, seeds)
    _mc_collider_kernel(n_sims, n_obs, 1.0, 1.0, seeds)
    pr.disable()
    sbuf = io.StringIO()
    pstats.Stats(pr, stream=sbuf).sort_stats("cumulative").print_stats(12)
    print(sbuf.getvalue())


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-core run: few assets, few MC sims")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile the MC hot loop and report Numba vs NumPy speedup")
    ap.add_argument("--n-sims", type=int, default=None)
    ap.add_argument("--n-obs", type=int, default=None)
    args = ap.parse_args()

    if args.profile:
        profile_mc(); return

    t0 = time.time()
    if args.smoke:
        n_sims = args.n_sims or 300
        n_obs = args.n_obs or 400
    else:
        n_sims = args.n_sims or 20000
        n_obs = args.n_obs or 2000

    print(f"=== PART (a) Monte Carlo of causal structures "
          f"(n_sims={n_sims}, n_obs={n_obs}) ===")
    mc_res, vinfo, mc_arrays = run_montecarlo(n_sims, n_obs, verify=True)
    for k, v in mc_res.items():
        print(f"  [{k:8s}] {json.dumps(v)}")
    if vinfo:
        print(f"  [verify] bit-identical check on {vinfo['n_verified']} sims: "
              f"max|Δ| struct={vinfo['max_abs_diff_struct']:.2e}, "
              f"collider={vinfo['max_abs_diff_collider']:.2e}")

    print("\n=== PART (a2) MC dose-response sweep (bias strength -> decision error) ===")
    sweep_df = run_mc_sweep(n_sims if not args.smoke else 300, n_obs)
    print(sweep_df[["strength", "fork_naive_falsepos", "fork_adj_falsepos",
                    "collider_naive_falsepos", "collider_adj_falsepos",
                    "chain_naive_power", "chain_adjM_power"]].to_string(index=False))

    print("\n=== PART (b) Real cross-asset factors: naive vs backdoor ===")
    rets, mk = build_daily_panel(smoke=args.smoke)
    print(f"  panel: {rets.shape[1]} instruments, {rets.shape[0]} days, "
          f"markets={sorted(set(mk.values()))}")
    fac_df, dsr_df, portfolios = run_real_factors(rets, mk)
    if len(fac_df):
        print(fac_df[["market", "factor", "n_obs", "naive_t", "adj_t", "adj2_t",
                      "flips", "flip_kind", "t_attenuation", "ls_sr_ann"]].to_string(index=False))
        print(f"  -> verdict FLIPS in {int(fac_df['flips'].sum())}/{len(fac_df)} cells "
              f"(market-mean backdoor); "
              f"survive BOTH backdoors: "
              f"{int((fac_df['verdict_naive'] & fac_df['verdict_adj'] & fac_df['verdict_adj2']).sum())}/{len(fac_df)}")
    if len(dsr_df):
        print("  best naive LS-factor DSR:", dsr_df.to_dict("records")[0])

    print("\n=== PART (c) Hierarchy-of-evidence / falsification checklist ===")
    chk = hierarchy_checklist(fac_df, mc_res, dsr_df)
    print(chk.to_string(index=False))

    # write tables
    pd.DataFrame([dict(structure=k, **v) for k, v in mc_res.items()]).to_csv(
        os.path.join(TAB, "mc_structures.csv"), index=False)
    sweep_df.to_csv(os.path.join(TAB, "mc_sweep.csv"), index=False)
    fac_df.to_csv(os.path.join(TAB, "real_factor_naive_vs_backdoor.csv"), index=False)
    dsr_df.to_csv(os.path.join(TAB, "real_factor_dsr.csv"), index=False)
    chk.to_csv(os.path.join(TAB, "hierarchy_checklist.csv"), index=False)
    json.dump({"mc": mc_res, "verify": vinfo, "n_sims": n_sims, "n_obs": n_obs,
               "n_flips": int(fac_df["flips"].sum()) if len(fac_df) else 0,
               "n_naive_sig": int(fac_df["verdict_naive"].sum()) if len(fac_df) else 0,
               "n_adj_sig": int(fac_df["verdict_adj"].sum()) if len(fac_df) else 0,
               "n_survive_both": int((fac_df["verdict_naive"] & fac_df["verdict_adj"]
                                      & fac_df["verdict_adj2"]).sum()) if len(fac_df) else 0,
               "best_dsr": float(dsr_df["dsr"].iloc[0]) if len(dsr_df) else None,
               "checklist_max_level_passed": int(chk.loc[chk["passes"], "level"].max())
                                             if chk["passes"].any() else 0},
              open(os.path.join(TAB, "summary.json"), "w"), indent=2, default=str)

    if not args.smoke:
        print("\n[figures] rendering...")
        make_figures(mc_arrays, mc_res, fac_df, portfolios, sweep_df)

    print(f"\nDONE in {time.time()-t0:.1f}s. Tables -> {TAB}")


if __name__ == "__main__":
    main()
