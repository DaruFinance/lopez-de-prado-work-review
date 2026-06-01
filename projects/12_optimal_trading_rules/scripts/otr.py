"""
otr.py, Optimal Trading Rules without backtesting (OU process) + Triple Penance.

Self-contained library imported by run_optimal_trading_rules.py. Implements the
two López de Prado pieces this project reproduces:

  1. OPTIMAL TRADING RULES WITHOUT BACKTESTING
     "Determining Optimal Trading Rules without Backtesting" (Bailey & LdP, 2014);
     AFML Ch.13. Given a price/spread series believed to mean-revert, fit a
     discrete Ornstein-Uhlenbeck process by OLS on the recursion

         x_t = E0 + phi * (x_{t-1} - E0) + sigma * eps_t ,   eps_t ~ N(0,1)

     (equivalently x_t - x_{t-1} = (1-phi)*(E0 - x_{t-1}) + sigma*eps_t). From the
     fitted (E0, phi, sigma) we derive HALF-LIFE = -ln(2)/ln(phi). We then build a
     MONTE-CARLO mesh over a grid of (profit-take, stop-loss) thresholds: for each
     (pt, sl) cell we simulate many OU paths starting at the long-run mean, exit
     at the FIRST touch of pt or sl (or a max-horizon cap), and record the Sharpe
     of the per-path P&L. The cell with the maximum mean Sharpe is the OPTIMAL
     RULE. This MC mesh is the heavy hot loop -> Numba kernel.

     *** SANCTIONED SYNTHETIC STEP (labelled as such) ***
     The OU mesh is a Monte-Carlo on a FITTED data-generating process. The OU
     parameters are FIT to a REAL, causal in-sample series (no lookahead), and the
     resulting (pt, sl) rule is then VALIDATED on REAL out-of-sample data with full
     intrabar-OHLC exits and costs. So: synthetic only inside the rule-derivation
     step LdP designed to be backtest-free; every reported headline is real-OOS.

  2. TRIPLE PENANCE
     "The Sharpe Ratio Efficient Frontier" / "Drawdown and Time Under Water"
     framework, formalised in Bailey & LdP, "Stop-Outs Under Serial Correlation
     and the Triple Penance Rule" (2015). Under an AR(1) (first-order serially
     correlated) Gaussian return process, the maximum expected drawdown and the
     maximum expected time-under-water both scale with phi; the "triple penance"
     result is that for a Gaussian-AR(1) process at the usual confidence the time
     under water is ~3x the time spent reaching the maximum drawdown. We measure
     phi (AR(1)) of the realised strategy returns and compare the implied
     serial-correlation-aware max-drawdown / TuW bound to the naive IID bound.

Volatility / mean for the entry signal are causal rolling statistics. All bar
scans are Numba @njit with a pure-Python reference for bit-identical verification.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as ss

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False
    prange = range
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]


_BASE_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume"]


# --------------------------------------------------------------------------- #
# Data loaders (robust; mirror tbm.py conventions)
# --------------------------------------------------------------------------- #
def load_base_crypto(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["open_time", *_BASE_COLS])
    ot = df["open_time"]
    if pd.api.types.is_datetime64_any_dtype(ot):
        ot = pd.to_datetime(ot, utc=True)
    elif np.issubdtype(np.asarray(ot).dtype, np.number):
        ot = pd.to_datetime(ot, unit="ms", utc=True)
    else:
        ot = pd.to_datetime(ot, utc=True)
    df = df.assign(open_time=ot).set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)]
    return df


def load_base_fx(path: str) -> pd.DataFrame:
    """FX 1m parquet: open/high/low/close/count, NO volume. Add proxy activity
    cols so tick bars (threshold_bars on 'count') can be built by the shared agg."""
    f = pd.read_parquet(path)
    f = f[["open", "high", "low", "close", "count"]].copy()
    f.index = pd.to_datetime(f.index, utc=True)
    f = f.sort_index()
    f = f[~f.index.duplicated(keep="first")]
    f = f[(f["close"] > 0) & (f["count"] > 0)]
    f["volume"] = f["count"].astype(float)
    f["quote_volume"] = f["count"].astype(float)
    f["taker_buy_volume"] = f["count"].astype(float) / 2.0
    f["taker_buy_quote_volume"] = f["count"].astype(float) / 2.0
    f.index.name = "open_time"
    return f


# --------------------------------------------------------------------------- #
# OU process fit (OLS on the mean-reversion recursion), causal, in-sample only
# --------------------------------------------------------------------------- #
def fit_ou(x: np.ndarray) -> dict:
    """Fit a discrete OU / AR(1) process to series x by OLS on
        x_t = a + phi * x_{t-1} + e_t,    phi in (0,1) for mean reversion.
    Returns E0 = a/(1-phi) (long-run mean), phi, sigma (resid std), and the
    half-life -ln2/ln(phi). x must be a stationary level series (e.g. a z-scored
    price or a residual spread). Uses only the data passed in (caller passes the
    IN-SAMPLE slice), so the fit is causal."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return dict(E0=0.0, phi=0.0, sigma=1.0, half_life=np.inf, ok=False)
    y = x[1:]
    xl = x[:-1]
    X = np.column_stack([np.ones_like(xl), xl])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, phi = float(beta[0]), float(beta[1])
    resid = y - X @ beta
    sigma = float(resid.std(ddof=2)) if len(resid) > 2 else float(resid.std())
    # clamp phi to a usable mean-reverting range; flag non-mean-reverting fits
    ok = (0.0 < phi < 1.0) and np.isfinite(sigma) and sigma > 0
    E0 = a / (1.0 - phi) if abs(1.0 - phi) > 1e-12 else 0.0
    half_life = (-np.log(2.0) / np.log(phi)) if (0.0 < phi < 1.0) else np.inf
    return dict(E0=E0, phi=phi, sigma=sigma, half_life=half_life, ok=bool(ok))


# --------------------------------------------------------------------------- #
# OU optimal-trading-rule Monte-Carlo mesh, THE HEAVY HOT LOOP (Numba kernel)
# --------------------------------------------------------------------------- #
@njit(cache=True, parallel=False)
def _ou_mesh_kernel(E0, phi, sigma, x0, pt_grid, sl_grid,
                    n_paths, max_horizon, rand):
    """Monte-Carlo OU first-touch Sharpe over a (pt, sl) mesh.

    For each grid cell (i=pt, j=sl):
      simulate n_paths OU paths starting at x0; each step
          x = E0 + phi*(x - E0) + sigma*eps
      enter LONG the spread at x0 (betting on reversion up to E0); record the
      per-path P&L = (exit_level - x0) when the path first reaches:
          + pt above x0  (profit-take)   -> +pt
          - sl below x0  (stop-loss)     -> -sl
          or the horizon cap            -> (x_T - x0)
      The cell value = Sharpe of the n_paths P&L vector (mean/std). Returns the
      Sharpe mesh and the mean-P&L mesh. `rand` is a pre-drawn standard-normal
      array of shape (n_paths, max_horizon) reused across cells for a common
      random-number stream (variance reduction + reproducibility)."""
    ni = pt_grid.shape[0]
    nj = sl_grid.shape[0]
    sharpe = np.empty((ni, nj), np.float64)
    meanpnl = np.empty((ni, nj), np.float64)
    for i in range(ni):
        pt = pt_grid[i]
        for j in range(nj):
            sl = sl_grid[j]
            s = 0.0
            ss_ = 0.0
            for p in range(n_paths):
                x = x0
                pnl = x - x0          # default if never moves (0)
                hit = False
                for t in range(max_horizon):
                    x = E0 + phi * (x - E0) + sigma * rand[p, t]
                    if x - x0 >= pt:          # profit-take (reversion up)
                        pnl = pt
                        hit = True
                        break
                    if x0 - x >= sl:          # stop-loss (moved further away)
                        pnl = -sl
                        hit = True
                        break
                if not hit:
                    pnl = x - x0              # mark-to-market at horizon
                s += pnl
                ss_ += pnl * pnl
            mu = s / n_paths
            var = ss_ / n_paths - mu * mu
            sd = np.sqrt(var) if var > 0.0 else 0.0
            sharpe[i, j] = (mu / sd) if sd > 0.0 else 0.0
            meanpnl[i, j] = mu
    return sharpe, meanpnl


def ou_mesh(E0, phi, sigma, x0, pt_grid, sl_grid, n_paths, max_horizon,
            seed=0):
    """Driver around the Numba MC mesh kernel. Pre-draws the common random
    stream and returns (sharpe_mesh, meanpnl_mesh)."""
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((int(n_paths), int(max_horizon))).astype(np.float64)
    pt_grid = np.ascontiguousarray(np.asarray(pt_grid, np.float64))
    sl_grid = np.ascontiguousarray(np.asarray(sl_grid, np.float64))
    return _ou_mesh_kernel(float(E0), float(phi), float(sigma), float(x0),
                           pt_grid, sl_grid, int(n_paths), int(max_horizon),
                           rand)


def ou_mesh_reference(E0, phi, sigma, x0, pt_grid, sl_grid, n_paths,
                      max_horizon, seed=0):
    """Pure-Python/NumPy reference for bit-identical verification of the kernel.

    Independently written (vectorised per cell over paths, explicit time loop),
    using the SAME pre-drawn random stream so results must match to FP rounding."""
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((int(n_paths), int(max_horizon))).astype(np.float64)
    pt_grid = np.asarray(pt_grid, np.float64)
    sl_grid = np.asarray(sl_grid, np.float64)
    ni, nj = len(pt_grid), len(sl_grid)
    sharpe = np.empty((ni, nj)); meanpnl = np.empty((ni, nj))
    for i in range(ni):
        pt = pt_grid[i]
        for j in range(nj):
            sl = sl_grid[j]
            pnl = np.empty(n_paths)
            for p in range(n_paths):
                x = x0
                v = x - x0
                hit = False
                for t in range(max_horizon):
                    x = E0 + phi * (x - E0) + sigma * rand[p, t]
                    if x - x0 >= pt:
                        v = pt; hit = True; break
                    if x0 - x >= sl:
                        v = -sl; hit = True; break
                if not hit:
                    v = x - x0
                pnl[p] = v
            mu = pnl.mean()
            sd = pnl.std()          # population std, matches kernel's ss/N form
            sharpe[i, j] = (mu / sd) if sd > 0 else 0.0
            meanpnl[i, j] = mu
    return sharpe, meanpnl


def optimal_rule(sharpe_mesh, pt_grid, sl_grid):
    """Argmax cell of the Sharpe mesh -> (pt*, sl*, sharpe*)."""
    i, j = np.unravel_index(int(np.argmax(sharpe_mesh)), sharpe_mesh.shape)
    return float(pt_grid[i]), float(sl_grid[j]), float(sharpe_mesh[i, j]), (i, j)


# --------------------------------------------------------------------------- #
# Causal mean-reversion entry signal on a z-scored level series
# --------------------------------------------------------------------------- #
def zscore_level(close: np.ndarray, span: int) -> np.ndarray:
    """Causal rolling z-score of LOG price: (logP - EWMA_mean)/EWMA_std. This is
    the stationary mean-reverting LEVEL series we fit the OU to and trade. Output
    at t uses only data <= t."""
    lp = pd.Series(np.log(np.asarray(close, np.float64)))
    mu = lp.ewm(span=span, adjust=True).mean()
    sd = lp.ewm(span=span, adjust=True).std()
    z = ((lp - mu) / sd.replace(0, np.nan)).to_numpy()
    z[~np.isfinite(z)] = 0.0
    return z


def mr_entry_events(z: np.ndarray, entry_z: float, warm: int) -> tuple[np.ndarray, np.ndarray]:
    """Vol-scaled mean-reversion entries. An event fires when |z| first crosses
    entry_z (a fresh extreme), with side = -sign(z) (fade the deviation, bet on
    reversion to the mean). Returns (event indices, side). Causal: uses z[t] only.
    Consecutive bars beyond the threshold collapse to the first crossing."""
    z = np.asarray(z, np.float64)
    over = np.abs(z) >= entry_z
    fresh = over.copy()
    fresh[1:] &= ~over[:-1]              # only the first bar of each excursion
    idx = np.where(fresh)[0]
    idx = idx[idx > warm]
    side = -np.sign(z[idx]).astype(np.int64)
    keep = side != 0
    return idx[keep].astype(np.int64), side[keep]


# --------------------------------------------------------------------------- #
# OOS exit applier with full intrabar OHLC + costs, Numba kernel
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _apply_rule_kernel(ev_idx, side, entry_px, sigma_px, pt_mult, sl_mult,
                       max_hold, open_, high, low, close):
    """Apply a fixed (pt_mult, sl_mult) vol-scaled rule on REAL OOS bars using
    full intrabar OHLC first-touch (high/low, not close-only). pt_mult/sl_mult are
    in units of the per-event price sigma. Returns side-signed gross log-returns,
    exit bar, label, hold. Pessimistic ordering: adverse (stop) barrier checked
    first when both touched in the same bar."""
    n_ev = ev_idx.shape[0]
    n = close.shape[0]
    out_ret = np.empty(n_ev, np.float64)
    out_touch = np.empty(n_ev, np.int64)
    out_label = np.empty(n_ev, np.int64)
    out_hold = np.empty(n_ev, np.int64)
    for k in range(n_ev):
        i0 = ev_idx[k]
        s = side[k]
        entry = entry_px[k]
        sig = sigma_px[k]
        up = entry * np.exp(pt_mult * sig)
        dn = entry * np.exp(-sl_mult * sig)
        j_end = i0 + max_hold
        if j_end > n - 1:
            j_end = n - 1
        touched = -1
        label = 0
        exit_px = close[j_end]
        for j in range(i0 + 1, j_end + 1):
            hi = high[j]
            lo = low[j]
            if s > 0:                     # long: profit if price rises to up
                if lo <= dn:              # adverse first
                    touched = j; label = -1; exit_px = dn; break
                if hi >= up:
                    touched = j; label = 1; exit_px = up; break
            else:                         # short: profit if price falls to dn
                if hi >= up:              # adverse first
                    touched = j; label = -1; exit_px = up; break
                if lo <= dn:
                    touched = j; label = 1; exit_px = dn; break
        if touched < 0:
            touched = j_end; label = 0; exit_px = close[j_end]
        gross = s * (np.log(exit_px) - np.log(entry))
        out_ret[k] = gross
        out_touch[k] = touched
        out_label[k] = label
        out_hold[k] = touched - i0
    return out_ret, out_touch, out_label, out_hold


def apply_rule(close, high, low, open_, ev_idx, side, sigma_px,
               pt_mult, sl_mult, max_hold):
    """Driver wrapper. entry price = close[i0]; sigma_px is the per-event price
    volatility (fraction) used to scale the barriers."""
    ev_idx = np.ascontiguousarray(np.asarray(ev_idx, np.int64))
    side = np.ascontiguousarray(np.asarray(side, np.int64))
    f = lambda x: np.ascontiguousarray(np.asarray(x, np.float64))
    close = f(close)
    entry_px = np.ascontiguousarray(close[ev_idx])
    ret, touch, lab, hold = _apply_rule_kernel(
        ev_idx, side, entry_px, f(sigma_px[ev_idx]) if sigma_px.shape == close.shape else f(sigma_px),
        float(pt_mult), float(sl_mult), int(max_hold),
        f(open_), f(high), f(low), close)
    return pd.DataFrame({"ev_idx": ev_idx, "side": side, "touch": touch,
                         "label": lab, "ret_gross": ret, "hold": hold})


# --------------------------------------------------------------------------- #
# Triple Penance, serial-correlation-aware max-drawdown / time-under-water
# --------------------------------------------------------------------------- #
def ar1_phi(returns: np.ndarray) -> float:
    """Lag-1 autocorrelation (AR(1) phi) of a return series."""
    r = np.asarray(returns, np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 0.0
    r0 = r - r.mean()
    denom = (r0 * r0).sum()
    if denom <= 0:
        return 0.0
    return float((r0[1:] * r0[:-1]).sum() / denom)


def triple_penance(returns: np.ndarray, conf: float = 0.95) -> dict:
    """Serial-correlation-aware maximum drawdown & time-under-water bounds under a
    Gaussian AR(1) return process (Bailey & LdP 2015, "Stop-Outs Under Serial
    Correlation / the Triple Penance Rule").

    Model: r_t mean phi-AR(1). Let mu, sigma be the (per-bar) mean & std of the
    *innovation-equivalent* process. The maximum drawdown at confidence `conf`
    for a process with positive drift mu and volatility sigma is the classic
        MaxDD = (quantile of the running-min of an arithmetic BM with drift).
    LdP give the closed forms for the maximum time under water (TuW) and show that
    under AR(1) both the MaxDD and the TuW inflate by a serial-correlation factor
        k(phi) = (1+phi)/(1-phi)               (the long-run-variance multiplier),
    so the effective volatility is sigma_eff = sigma * sqrt(k(phi)). The "triple
    penance" is that the expected TuW is ~3x the time-to-reach-max-drawdown for a
    Gaussian process at the standard confidence.

    Returns the naive (IID) and serial-correlation-adjusted MaxDD / MaxTuW (in
    bars), the AR(1) phi, and the variance-inflation factor k(phi). When mu<=0 the
    drawdown is unbounded (returns inf), flagged via `bounded`."""
    r = np.asarray(returns, np.float64)
    r = r[np.isfinite(r)]
    out = dict(phi=0.0, k=1.0, mu=0.0, sigma=0.0, bounded=False,
               maxdd_iid=np.inf, maxdd_ar1=np.inf,
               tuw_iid=np.inf, tuw_ar1=np.inf)
    if len(r) < 10:
        return out
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    phi = ar1_phi(r)
    phi = max(min(phi, 0.999), -0.999)
    k = (1.0 + phi) / (1.0 - phi) if abs(1.0 - phi) > 1e-9 else np.inf
    out.update(phi=phi, k=float(k), mu=mu, sigma=sigma)
    if sigma <= 0 or mu <= 0 or not np.isfinite(k):
        return out                          # unbounded / degenerate
    out["bounded"] = True
    za = ss.norm.ppf(conf)
    # Maximum drawdown (magnitude, in return units) at confidence conf for an
    # arithmetic process with per-bar drift mu and per-bar vol s:
    #   MaxDD ≈ (za*s)^2 / (4*mu)   (López de Prado's quadratic golden-section
    #   approximation to the max of the running deficit). Time under water:
    #   MaxTuW ≈ (za*s / mu)^2   (bars). Both grow with effective vol.
    def dd_tuw(s):
        maxdd = (za * s) ** 2 / (4.0 * mu)
        tuw = (za * s / mu) ** 2
        return maxdd, tuw
    dd_i, tuw_i = dd_tuw(sigma)
    dd_a, tuw_a = dd_tuw(sigma * np.sqrt(k))
    out.update(maxdd_iid=float(dd_i), maxdd_ar1=float(dd_a),
               tuw_iid=float(tuw_i), tuw_ar1=float(tuw_a))
    return out
