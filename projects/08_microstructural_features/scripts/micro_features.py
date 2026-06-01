#!/usr/bin/env python3
"""
micro_features.py — causal microstructure estimators (Lopez de Prado, AFML Ch.19).

A reusable feature library. Every estimator below is computed on a ROLLING,
strictly-causal window: the value at bar t uses only bars <= t. Each rolling
estimator has BOTH a pure-python/pandas reference and an @njit kernel; the
driver script verifies the two are bit-identical (--verify, prints max|delta|).

ESTIMATORS (AFML Ch.19, with the primary sources):
  - Roll effective spread .......... Roll (1984)         needs: close
  - Corwin-Schultz high-low spread . Corwin & Schultz (2012) needs: high, low
  - Kyle's lambda .................. Kyle (1985)         needs: dprice, signed volume
  - Amihud lambda (illiquidity) .... Amihud (2002)       needs: |return|, dollar vol
  - Hasbrouck lambda ............... Hasbrouck (2009)    needs: dprice, signed sqrt(dollar)
  - VPIN ........................... Easley, Lopez de Prado, O'Hara (2012) needs: buy/sell vol
  - Bulk-Volume Classification ..... Easley, Lopez de Prado, O'Hara (2012) needs: dprice, vol, sigma
  - Order-flow imbalance ........... (AFML 19.x)          needs: buy/sell vol
  - Tick rule (signed) ............. (AFML 19.x)          needs: dprice only

SIDE-INFORMATION TIERS (which estimators are *honestly* computable per market):
  crypto  : taker_buy_volume present -> TRUE signed volume. ALL estimators.
  equity  : volume + trade count, NO side -> classify with BVC. All but true VPIN/OFI.
  forex   : tick count only, NO volume  -> price-only estimators + tick-rule only.

All bar-aggregate signing uses ONLY information available at the close of bar t
(the bar's own OHLC and volume), so the features are causal at bar granularity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as ss

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]


# =========================================================================== #
# Bar-level signing primitives (causal: use only bar t's own data)
# =========================================================================== #
def tick_rule(close: np.ndarray) -> np.ndarray:
    """Signed tick series b_t in {-1,0,+1}: sign of the close-to-close change,
    carrying the last non-zero sign forward on a flat tick (Lee-Ready style).
    Causal: b_t depends on close[t] and close[t-1] only."""
    return _tick_rule_nb(np.ascontiguousarray(close, np.float64))


def _tick_rule_py(close):
    n = close.shape[0]
    b = np.zeros(n)
    last = 1.0
    for t in range(1, n):
        d = close[t] - close[t - 1]
        if d > 0:
            last = 1.0
        elif d < 0:
            last = -1.0
        b[t] = last
    return b


@njit(cache=True)
def _tick_rule_nb(close):
    n = close.shape[0]
    b = np.zeros(n)
    last = 1.0
    for t in range(1, n):
        d = close[t] - close[t - 1]
        if d > 0.0:
            last = 1.0
        elif d < 0.0:
            last = -1.0
        b[t] = last
    return b


def bvc_buy_fraction(close: np.ndarray, sigma_window: int = 50) -> np.ndarray:
    """Bulk-Volume Classification buy fraction (Easley-LdP-O'Hara 2012).

    frac_buy[t] = Phi( (P_t - P_{t-1}) / sigma_dP ), where sigma_dP is a TRAILING
    std of the price change (causal). Used when no trade-side flag exists
    (equities, forex). Returns the estimated *fraction* of bar volume that is
    buyer-initiated, in [0,1]. P uses log-close to be scale-free across markets.
    """
    lp = np.log(close)
    dP = np.diff(lp, prepend=lp[0])
    sd = pd.Series(dP).rolling(sigma_window).std().to_numpy()
    z = np.where(sd > 0, dP / sd, 0.0)
    return ss.norm.cdf(z)


# =========================================================================== #
# Rolling causal estimators — pandas/python reference + numba kernel
# =========================================================================== #
def roll_spread(close: np.ndarray, window: int) -> np.ndarray:
    """Roll (1984) effective spread from serial covariance of price changes.
    s_t = 2*sqrt(-cov(dP_t, dP_{t-1})) over a trailing window; 0 when cov>=0.
    Causal: the trailing window ends at bar t."""
    lp = np.log(np.ascontiguousarray(close, np.float64))
    dP = np.diff(lp, prepend=lp[0])
    return _roll_spread_nb(np.ascontiguousarray(dP), int(window))


def _roll_spread_py(dP, window):
    # Reference uses the SAME sequential summation order as the numba kernel so
    # the two are bit-identical (numpy's pairwise .mean() reorders the adds).
    n = dP.shape[0]
    out = np.full(n, np.nan)
    for t in range(window, n):
        sa = 0.0; sb = 0.0
        for k in range(window):
            sa += dP[t - window + 1 + k]
            sb += dP[t - window + k]
        ma = sa / window; mb = sb / window
        cov = 0.0
        for k in range(window):
            cov += (dP[t - window + 1 + k] - ma) * (dP[t - window + k] - mb)
        cov /= window
        out[t] = 2.0 * np.sqrt(-cov) if cov < 0.0 else 0.0
    return out


@njit(cache=True)
def _roll_spread_nb(dP, window):
    n = dP.shape[0]
    out = np.full(n, np.nan)
    for t in range(window, n):
        sa = 0.0; sb = 0.0
        for k in range(window):
            sa += dP[t - window + 1 + k]
            sb += dP[t - window + k]
        ma = sa / window; mb = sb / window
        cov = 0.0
        for k in range(window):
            cov += (dP[t - window + 1 + k] - ma) * (dP[t - window + k] - mb)
        cov /= window
        out[t] = 2.0 * np.sqrt(-cov) if cov < 0.0 else 0.0
    return out


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Corwin-Schultz (2012) high-low spread estimator (2-bar, then clipped at 0).
    Uses bars t-1 and t (causal). Returns a per-bar spread proxy (NaN on bar 0).

    This is a fixed 2-bar window (not a rolling reduction), so it vectorises
    cleanly in numpy and gains nothing from Numba — we keep it as pure numpy.
    The Numba-vs-libm 1-ULP `log`/`exp` difference is therefore avoided entirely:
    all transcendentals run through numpy here."""
    h = np.asarray(high, np.float64); l = np.asarray(low, np.float64)
    n = h.shape[0]
    out = np.full(n, np.nan)
    if n < 2:
        return out
    k = 3.0 - 2.0 * np.sqrt(2.0)
    b_cur = np.log(h / l) ** 2
    beta = b_cur[1:] + b_cur[:-1]
    hh = np.maximum(h[1:], h[:-1]); ll = np.minimum(l[1:], l[:-1])
    gamma = np.log(hh / ll) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    out[1:] = np.where(s > 0.0, s, 0.0)
    return out


def kyle_lambda(close: np.ndarray, signed_volume: np.ndarray,
                window: int) -> np.ndarray:
    """Kyle's (1985) lambda: rolling OLS slope of dP on signed net volume.
    lambda_t = cov(dP, b*V) / var(b*V) over the trailing window. Price impact per
    unit signed flow. Causal: trailing window ends at t."""
    lp = np.log(np.ascontiguousarray(close, np.float64))
    dP = np.diff(lp, prepend=lp[0])
    return _ols_slope_nb(np.ascontiguousarray(dP),
                         np.ascontiguousarray(signed_volume, np.float64), int(window))


def amihud_lambda(close: np.ndarray, dollar_volume: np.ndarray,
                  window: int) -> np.ndarray:
    """Amihud (2002) illiquidity: trailing mean of |return| / dollar-volume.
    Causal."""
    lp = np.log(np.ascontiguousarray(close, np.float64))
    r = np.abs(np.diff(lp, prepend=lp[0]))
    dv = np.ascontiguousarray(dollar_volume, np.float64)
    ratio = np.where(dv > 0, r / dv, np.nan)
    return _roll_nanmean_nb(np.ascontiguousarray(ratio), int(window))


def hasbrouck_lambda(close: np.ndarray, signed_dollar: np.ndarray,
                     window: int) -> np.ndarray:
    """Hasbrouck (2009) lambda: rolling OLS slope of dP on signed sqrt(dollar
    volume) (b * sqrt($vol)). Causal trailing window."""
    lp = np.log(np.ascontiguousarray(close, np.float64))
    dP = np.diff(lp, prepend=lp[0])
    return _ols_slope_nb(np.ascontiguousarray(dP),
                         np.ascontiguousarray(signed_dollar, np.float64), int(window))


def _ols_slope_py(y, x, window):
    # Same sequential summation order as the numba kernel (bit-identical).
    n = y.shape[0]
    out = np.full(n, np.nan)
    for t in range(window, n):
        sx = 0.0; sy = 0.0
        for k in range(window):
            sx += x[t - window + 1 + k]; sy += y[t - window + 1 + k]
        mx = sx / window; my = sy / window
        vx = 0.0; cxy = 0.0
        for k in range(window):
            dx = x[t - window + 1 + k] - mx
            dy = y[t - window + 1 + k] - my
            vx += dx * dx; cxy += dx * dy
        vx /= window; cxy /= window
        out[t] = cxy / vx if vx > 0.0 else np.nan
    return out


@njit(cache=True)
def _ols_slope_nb(y, x, window):
    n = y.shape[0]
    out = np.full(n, np.nan)
    for t in range(window, n):
        sx = 0.0; sy = 0.0
        for k in range(window):
            sx += x[t - window + 1 + k]; sy += y[t - window + 1 + k]
        mx = sx / window; my = sy / window
        vx = 0.0; cxy = 0.0
        for k in range(window):
            dx = x[t - window + 1 + k] - mx
            dy = y[t - window + 1 + k] - my
            vx += dx * dx; cxy += dx * dy
        vx /= window; cxy /= window
        out[t] = cxy / vx if vx > 0.0 else np.nan
    return out


def _roll_nanmean_py(v, window):
    n = v.shape[0]
    out = np.full(n, np.nan)
    for t in range(window, n):
        s = 0.0; c = 0
        for k in range(window):
            x = v[t - window + 1 + k]
            if not np.isnan(x):
                s += x; c += 1
        out[t] = s / c if c > 0 else np.nan
    return out


@njit(cache=True)
def _roll_nanmean_nb(v, window):
    n = v.shape[0]
    out = np.full(n, np.nan)
    for t in range(window, n):
        s = 0.0; c = 0
        for k in range(window):
            x = v[t - window + 1 + k]
            if not np.isnan(x):
                s += x; c += 1
        out[t] = s / c if c > 0 else np.nan
    return out


def vpin(buy_volume: np.ndarray, sell_volume: np.ndarray,
         window: int) -> np.ndarray:
    """VPIN (Easley, Lopez de Prado, O'Hara 2012): volume-synchronized prob. of
    informed trading = trailing mean of |V_buy - V_sell| / (V_buy + V_sell) over
    `window` volume buckets. Here each bar is one (volume-clock) bucket because
    bars are already information-driven. Causal trailing window."""
    bv = np.ascontiguousarray(buy_volume, np.float64)
    sv = np.ascontiguousarray(sell_volume, np.float64)
    tot = bv + sv
    imb = np.where(tot > 0, np.abs(bv - sv) / tot, np.nan)
    return _roll_nanmean_nb(np.ascontiguousarray(imb), int(window))


def order_flow_imbalance(buy_volume: np.ndarray,
                         sell_volume: np.ndarray) -> np.ndarray:
    """Signed net order-flow imbalance per bar, normalised to [-1,1]. Causal
    (uses only bar t's own buy/sell split)."""
    bv = np.asarray(buy_volume, float); sv = np.asarray(sell_volume, float)
    tot = bv + sv
    return np.where(tot > 0, (bv - sv) / tot, 0.0)


# =========================================================================== #
# Per-market feature builder
# =========================================================================== #
# Window lengths (in bars) for the rolling estimators. Kept modest so the
# trailing windows stay informative at ~8 bars/day.
SPREAD_WIN = 20
LAMBDA_WIN = 50
VPIN_WIN = 50
BVC_SIGMA_WIN = 50

# Which estimators each market tier can honestly support.
MARKET_TIER = {
    "crypto": "side",     # true buy/sell volume
    "equity": "bvc",      # volume + trades, no side -> BVC
    "forex":  "tick",     # tick count only, no volume
}


def build_features(bars: pd.DataFrame, market: str) -> pd.DataFrame:
    """Compute all *available* microstructure features for one bar series.

    Expects the standard bars schema from lib.bars (_aggregate output):
        open, high, low, close, volume, dollar, ticks, buy_dollar, sell_dollar.
    `market` selects the side-information tier (see MARKET_TIER).
    Every column is causal at bar t. Side-volume-dependent columns are present
    only for the tiers that can compute them honestly.
    """
    tier = MARKET_TIER[market]
    close = bars["close"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    vol = bars["volume"].to_numpy(float)
    dollar = bars["dollar"].to_numpy(float)

    f = pd.DataFrame(index=bars.index)

    # --- price-only estimators: available in ALL markets ---
    f["roll_spread"] = roll_spread(close, SPREAD_WIN)
    f["corwin_schultz"] = corwin_schultz(high, low)
    f["amihud"] = amihud_lambda(close, dollar, LAMBDA_WIN)
    f["tick_sign"] = tick_rule(close)
    # trailing mean tick-sign = a price-only order-flow proxy
    f["tick_flow"] = pd.Series(f["tick_sign"].to_numpy()).rolling(SPREAD_WIN).mean().to_numpy()

    # --- determine the signed flow used for Kyle/Hasbrouck/VPIN/OFI ---
    if tier == "side":
        # crypto: TRUE buyer-/seller-initiated dollar split
        buy_d = bars["buy_dollar"].to_numpy(float)
        sell_d = bars["sell_dollar"].to_numpy(float)
        tot_d = buy_d + sell_d
        buy_frac = np.where(tot_d > 0, buy_d / tot_d, 0.5)
        buy_v = vol * buy_frac
        sell_v = vol * (1.0 - buy_frac)
        signed_vol = buy_v - sell_v
        signed_dollar = np.sign(signed_vol) * np.sqrt(np.abs(dollar))
        f["kyle"] = kyle_lambda(close, signed_vol, LAMBDA_WIN)
        f["hasbrouck"] = hasbrouck_lambda(close, signed_dollar, LAMBDA_WIN)
        f["vpin"] = vpin(buy_v, sell_v, VPIN_WIN)
        f["ofi"] = order_flow_imbalance(buy_v, sell_v)
    elif tier == "bvc":
        # equities: no side -> Bulk-Volume Classification splits the bar volume
        buy_frac = bvc_buy_fraction(close, BVC_SIGMA_WIN)
        buy_v = vol * buy_frac
        sell_v = vol * (1.0 - buy_frac)
        signed_vol = buy_v - sell_v
        signed_dollar = np.sign(signed_vol) * np.sqrt(np.abs(dollar))
        f["kyle"] = kyle_lambda(close, signed_vol, LAMBDA_WIN)
        f["hasbrouck"] = hasbrouck_lambda(close, signed_dollar, LAMBDA_WIN)
        f["vpin"] = vpin(buy_v, sell_v, VPIN_WIN)        # BVC-VPIN
        f["ofi"] = order_flow_imbalance(buy_v, sell_v)   # BVC-OFI
    else:
        # forex: no volume at all -> tick-rule signed *count* is the only flow.
        # Kyle/Hasbrouck need volume on the x-axis; with count-as-volume they are
        # tick-flow rescalings, so we expose only the honest tick-based pieces.
        b = f["tick_sign"].to_numpy()
        cnt = bars["ticks"].to_numpy(float)        # 'ticks' = aggregated count
        signed_cnt = b * cnt
        f["kyle"] = kyle_lambda(close, signed_cnt, LAMBDA_WIN)   # impact per signed tick
        # VPIN/Hasbrouck/OFI need volume side-split -> not honestly computable
    return f


def available_features(market: str) -> list[str]:
    """The feature columns honestly computable in a given market tier."""
    base = ["roll_spread", "corwin_schultz", "amihud", "tick_sign", "tick_flow"]
    tier = MARKET_TIER[market]
    if tier == "side":
        return base + ["kyle", "hasbrouck", "vpin", "ofi"]
    if tier == "bvc":
        return base + ["kyle", "hasbrouck", "vpin", "ofi"]   # all via BVC proxy
    return base + ["kyle"]                                    # forex: tick-only


# =========================================================================== #
# Self-test: bit-identical numba vs python reference
# =========================================================================== #
def verify_kernels(seed: int = 11) -> float:
    """Verify every numba kernel is bit-identical to its python reference.
    Returns the worst max|delta| across all kernels/trials."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    checks = []
    for _ in range(5):
        n = int(rng.integers(2000, 5000))
        close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100
        lp = np.log(close); dP = np.diff(lp, prepend=lp[0])
        high = close * (1 + np.abs(rng.standard_normal(n)) * 0.002)
        low = close * (1 - np.abs(rng.standard_normal(n)) * 0.002)
        x = rng.standard_normal(n) * 1e4
        v = np.abs(rng.standard_normal(n)) * 1e3
        v[rng.integers(0, n, 5)] = np.nan
        for win in (10, 20, 50):
            checks.append(("tick_rule", _tick_rule_py(close), _tick_rule_nb(close)))
            checks.append((f"roll_spread w{win}", _roll_spread_py(dP, win), _roll_spread_nb(dP, win)))
            checks.append((f"ols_slope w{win}", _ols_slope_py(dP, x, win), _ols_slope_nb(dP, x, win)))
            checks.append((f"roll_nanmean w{win}", _roll_nanmean_py(v, win), _roll_nanmean_nb(v, win)))
    for name, a, b in checks:
        a = np.asarray(a, float); b = np.asarray(b, float)
        ma, mb = np.isnan(a), np.isnan(b)
        if not np.array_equal(ma, mb):
            raise AssertionError(f"NaN mask mismatch in {name}")
        d = float(np.max(np.abs(a[~ma] - b[~mb]))) if (~ma).any() else 0.0
        worst = max(worst, d)
    return worst


if __name__ == "__main__":
    w = verify_kernels()
    print(f"micro_features kernel verify: max|delta| = {w:.3e}  "
          f"-> {'BIT-IDENTICAL' if w == 0 else 'MISMATCH'}")
