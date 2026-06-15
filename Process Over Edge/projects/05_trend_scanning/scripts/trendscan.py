"""
trendscan.py, Trend-Scanning labels (Lopez de Prado, ML4AM Ch.5) + a fixed-horizon
control labeller, both as Numba kernels with independent NumPy references for
bit-identical verification.

TREND-SCANNING (LdP "trend-scanning method"):
  For each observation t, regress price (here log-price) on a time index over each
  forward look-ahead window of length L in {L_min..., L_max}:
        y_j = a + b * j,    j = 0,1,...,L-1,   y_j = logprice[t + j]
  compute the slope b and its t-value  t_b = b / se(b)  with
        se(b) = sqrt( SSE/(L-2) / Sxx ),   Sxx = sum (x_j - xbar)^2.
  The label for t is the SIGN of the t-value at the horizon L* that MAXIMISES |t_b|
  (the most statistically significant local trend), and |t_b(L*)| is the
  CONFIDENCE / meta-label magnitude. We also return L* (the horizon that won) and
  the realised log-return over [t, t+L*] for P&L.

  This is forward-looking BY CONSTRUCTION, it is a LABEL (the supervised target),
  not a feature. The strategy that consumes it is causal: the model is trained on
  past (event,label) pairs under purged CV and only acts on out-of-fold scores.

FIXED-HORIZON control: regress over a single fixed L for every observation
  (the "fixed-window" labeller LdP contrasts trend-scanning against). Same OLS
  t-value machinery, no horizon search.

The per-observation multi-horizon OLS t-value scan is the HOT LOOP. We accumulate
Sx, Sxx, Sy, Sxy, Syy incrementally as L grows (O(1) per extra bar), so the whole
scan is O(n * (L_max - L_min)) with tiny constants, no per-horizon refit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]


_BASE_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume"]


# --------------------------------------------------------------------------- #
# Loaders (mirror project 03; read-only)
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
# Trend-scanning kernel (the hot loop), Numba
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _trend_scan_kernel(y, l_min, l_max):
    """Per-observation multi-horizon OLS-slope t-value scan over log-price y.

    For each t with at least l_min forward bars available, scan horizons
    L = l_min..l_max (capped by data end). Regress y[t..t+L-1] on x=0..L-1,
    pick the L* maximising |t_value of slope|. Returns, per t:
      t_val  : signed t-value at L* (the trend-scan statistic)
      label  : sign(t_val)  (+1 up-trend, -1 down-trend, 0 if undefined)
      lstar  : the winning horizon length L* (bars)
      ret    : realised log-return y[t+L*-1] - y[t] over the winning window
    Observations without l_min forward bars get label 0 / NaN t_val.

    Incremental accumulation: as L grows by one bar we add the new (x=L-1, y_new)
    point to running sums Sx,Sxx,Sy,Sxy,Syy in O(1); slope/t recomputed in O(1).
    For x = 0..L-1 (integers): Sx and Sxx have closed forms but we accumulate
    them too so the kernel is self-contained and matches the reference exactly.
    """
    n = y.shape[0]
    t_val = np.full(n, np.nan)
    label = np.zeros(n, np.int64)
    lstar = np.zeros(n, np.int64)
    ret = np.full(n, np.nan)
    for t in range(n):
        max_l = n - t                      # bars available forward incl. t
        if max_l < l_min:
            continue
        hi = l_max if l_max < max_l else max_l
        # running sums over the first l_min-1 points, then extend
        Sx = 0.0; Sxx = 0.0; Sy = 0.0; Sxy = 0.0; Syy = 0.0
        # seed with points x=0..(l_min-2)
        for j in range(l_min - 1):
            xj = float(j); yj = y[t + j]
            Sx += xj; Sxx += xj * xj; Sy += yj; Sxy += xj * yj; Syy += yj * yj
        best_abs_t = -1.0
        best_t = np.nan; best_l = 0
        for L in range(l_min, hi + 1):
            j = L - 1                       # new point index
            xj = float(j); yj = y[t + j]
            Sx += xj; Sxx += xj * xj; Sy += yj; Sxy += xj * yj; Syy += yj * yj
            Lf = float(L)
            # centered sums (match the reference exactly; numerically stable):
            #   Sxx_c = sum (x-xbar)^2 = Sxx - Sx^2/L
            #   Sxy_c = sum (x-xbar)(y-ybar) = Sxy - Sx*Sy/L
            #   Syy_c = sum (y-ybar)^2 = Syy - Sy^2/L
            sxx_c = Sxx - Sx * Sx / Lf
            if sxx_c <= 0.0:
                continue
            sxy_c = Sxy - Sx * Sy / Lf
            syy_c = Syy - Sy * Sy / Lf
            b = sxy_c / sxx_c                             # OLS slope
            # SSE = Syy_c - b * Sxy_c  (residual sum of squares, centered form)
            sse = syy_c - b * sxy_c
            if sse < 0.0:
                sse = 0.0
            if L > 2:
                s2 = sse / (Lf - 2.0)
                var_b = s2 / sxx_c                        # Var(b) = s2 / Sxx_centered
                if var_b > 0.0:
                    tb = b / np.sqrt(var_b)
                else:
                    tb = 0.0
            else:
                tb = 0.0                                 # t undefined for L<=2
            ab = tb if tb >= 0.0 else -tb
            if ab > best_abs_t:
                best_abs_t = ab; best_t = tb; best_l = L
        if best_l == 0:
            continue
        t_val[t] = best_t
        lstar[t] = best_l
        ret[t] = y[t + best_l - 1] - y[t]
        if best_t > 0.0:
            label[t] = 1
        elif best_t < 0.0:
            label[t] = -1
        else:
            label[t] = 0
    return t_val, label, lstar, ret


@njit(cache=True)
def _fixed_horizon_kernel(y, L):
    """Fixed-horizon OLS-slope t-value labeller: same statistic at a single L for
    every observation that has L forward bars. Control for trend-scanning."""
    n = y.shape[0]
    t_val = np.full(n, np.nan)
    label = np.zeros(n, np.int64)
    ret = np.full(n, np.nan)
    Lf = float(L)
    for t in range(n):
        if t + L > n:
            break
        Sx = 0.0; Sxx = 0.0; Sy = 0.0; Sxy = 0.0; Syy = 0.0
        for j in range(L):
            xj = float(j); yj = y[t + j]
            Sx += xj; Sxx += xj * xj; Sy += yj; Sxy += xj * yj; Syy += yj * yj
        sxx_c = Sxx - Sx * Sx / Lf
        if sxx_c <= 0.0:
            continue
        sxy_c = Sxy - Sx * Sy / Lf
        syy_c = Syy - Sy * Sy / Lf
        b = sxy_c / sxx_c
        sse = syy_c - b * sxy_c
        if sse < 0.0:
            sse = 0.0
        if L > 2:
            s2 = sse / (Lf - 2.0)
            var_b = s2 / sxx_c
            tb = b / np.sqrt(var_b) if var_b > 0.0 else 0.0
        else:
            tb = 0.0
        t_val[t] = tb
        ret[t] = y[t + L - 1] - y[t]
        label[t] = 1 if tb > 0.0 else (-1 if tb < 0.0 else 0)
    return t_val, label, ret


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def trend_scan(close: np.ndarray, l_min: int, l_max: int) -> pd.DataFrame:
    """Trend-scanning labels over log(close). Returns DataFrame per observation:
    t_val, label, lstar, ret (log-return over winning window)."""
    y = np.ascontiguousarray(np.log(np.asarray(close, np.float64)))
    tv, lab, ls, rt = _trend_scan_kernel(y, int(l_min), int(l_max))
    return pd.DataFrame({"t_val": tv, "label": lab, "lstar": ls, "ret": rt})


def fixed_horizon(close: np.ndarray, L: int) -> pd.DataFrame:
    y = np.ascontiguousarray(np.log(np.asarray(close, np.float64)))
    tv, lab, rt = _fixed_horizon_kernel(y, int(L))
    return pd.DataFrame({"t_val": tv, "label": lab, "ret": rt})


@njit(cache=True)
def _causal_hold_ret(y, ev_idx, hold):
    """Realised log-return of entering long at close of ev_idx[k] and exiting
    `hold` bars later (or at the last bar). UNSIGNED; caller multiplies by side.
    This is a CAUSALLY EXECUTABLE exit (a fixed forward hold), decoupled from the
    label's lookahead-optimal window, so trade P&L contains no label-endpoint
    selection bias. Returns ret and the actual bars held."""
    n = y.shape[0]
    m = ev_idx.shape[0]
    out = np.empty(m, np.float64)
    hbar = np.empty(m, np.int64)
    for k in range(m):
        i0 = ev_idx[k]
        j = i0 + hold
        if j > n - 1:
            j = n - 1
        out[k] = y[j] - y[i0]
        hbar[k] = j - i0
    return out, hbar


def causal_hold_ret(close, ev_idx, hold):
    """Driver: unsigned realised log-return over a fixed forward hold (causal)."""
    y = np.ascontiguousarray(np.log(np.asarray(close, np.float64)))
    ev = np.ascontiguousarray(np.asarray(ev_idx, np.int64))
    r, h = _causal_hold_ret(y, ev, int(hold))
    return r, h


# --------------------------------------------------------------------------- #
# NumPy reference (independent of the kernel) for bit-identical verification
# --------------------------------------------------------------------------- #
def trend_scan_reference(close, l_min, l_max):
    """Independent NumPy reference: closed-form OLS slope t-value via np.polyfit-
    equivalent normal equations, full refit per horizon (no incremental sums).
    Deliberately structured differently from the kernel."""
    y = np.log(np.asarray(close, float))
    n = len(y)
    tv = np.full(n, np.nan); lab = np.zeros(n, np.int64)
    ls = np.zeros(n, np.int64); rt = np.full(n, np.nan)
    for t in range(n):
        max_l = n - t
        if max_l < l_min:
            continue
        hi = min(l_max, max_l)
        best_abs = -1.0; best_t = np.nan; best_l = 0
        for L in range(l_min, hi + 1):
            x = np.arange(L, dtype=float)
            yy = y[t:t + L]
            xbar = x.mean(); ybar = yy.mean()
            sxx = np.sum((x - xbar) ** 2)
            sxy = np.sum((x - xbar) * (yy - ybar))
            if sxx <= 0:
                continue
            b = sxy / sxx
            a = ybar - b * xbar
            resid = yy - (a + b * x)
            sse = float(np.sum(resid ** 2))
            if L > 2:
                s2 = sse / (L - 2)
                var_b = s2 / sxx
                tb = b / np.sqrt(var_b) if var_b > 0 else 0.0
            else:
                tb = 0.0
            ab = abs(tb)
            if ab > best_abs:
                best_abs = ab; best_t = tb; best_l = L
        if best_l == 0:
            continue
        tv[t] = best_t; ls[t] = best_l; rt[t] = y[t + best_l - 1] - y[t]
        lab[t] = 1 if best_t > 0 else (-1 if best_t < 0 else 0)
    return pd.DataFrame({"t_val": tv, "label": lab, "lstar": ls, "ret": rt})


# --------------------------------------------------------------------------- #
# Causal features for the secondary (meta/size) model , reused from project 03
# --------------------------------------------------------------------------- #
def build_features(bars: pd.DataFrame, vol: np.ndarray) -> pd.DataFrame:
    """Causal feature matrix at each bar (no lookahead). Used to predict the sign
    AND/OR the confidence of the trend-scan label out-of-fold."""
    c = bars["close"]
    lp = np.log(c)
    r = lp.diff()

    def mom(k):
        return (lp - lp.shift(k)).to_numpy()

    delta = c.diff()
    up = delta.clip(lower=0.0); dn = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50.0)
    rsi_scaled = ((rsi - 50.0) / 50.0).to_numpy()

    vol_s = r.ewm(span=10).std().to_numpy()
    vol_l = r.ewm(span=50).std().to_numpy()
    vol_ratio = np.where(vol_l > 0, vol_s / vol_l, 1.0)

    if {"buy_dollar", "sell_dollar"}.issubset(bars.columns):
        num = bars["buy_dollar"] - bars["sell_dollar"]
        den = (bars["buy_dollar"] + bars["sell_dollar"]).replace(0, np.nan)
        ofi = (num / den).fillna(0.0).to_numpy()
    else:
        ofi = np.zeros(len(bars))

    rng = ((bars["high"] - bars["low"]) / c).ewm(span=10).mean().to_numpy()
    # trailing OLS-slope sign over a short causal window (past-only trend proxy)
    ema_f = c.ewm(span=10, adjust=True).mean()
    ema_s = c.ewm(span=50, adjust=True).mean()
    ma_gap = ((ema_f - ema_s) / c).to_numpy()

    feat = pd.DataFrame({
        "vol": vol,
        "ma_gap": ma_gap,
        "mom5": mom(5), "mom10": mom(10), "mom20": mom(20),
        "rsi": rsi_scaled,
        "vol_ratio": vol_ratio,
        "ofi": ofi,
        "range_atr": rng,
    }, index=range(len(bars)))
    return feat


def ewma_vol(close: np.ndarray, span: int) -> np.ndarray:
    """Causal EWMA std of close-to-close log returns (regime feature)."""
    lp = np.log(close)
    r = np.empty_like(lp); r[0] = 0.0; r[1:] = lp[1:] - lp[:-1]
    vol = pd.Series(r).ewm(span=span, adjust=True).std().to_numpy()
    vol[~np.isfinite(vol)] = 0.0
    return vol
