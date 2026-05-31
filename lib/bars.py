"""
bars.py — Information-driven bar construction (López de Prado, AFML Ch. 2).

Builds time, tick, volume, and dollar bars (plus order-flow imbalance) from a
base OHLCV series that carries trade count and taker-buy volume. Vectorised
cumulative-threshold sampling, O(n), so it scales to the full cross-section.

Base-bar schema expected (Binance USD-M perp 30m parquet):
    open_time, open, high, low, close, volume, quote_volume, count,
    taker_buy_volume, taker_buy_quote_volume

All builders return a DataFrame indexed by bar-close time with columns:
    open, high, low, close, volume, dollar, ticks, buy_dollar, sell_dollar,
    t_open  (bar-open timestamp), n_base (base bars consumed)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BASE_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
             "count", "taker_buy_volume", "taker_buy_quote_volume"]


def load_base(path: str) -> pd.DataFrame:
    """Load a base OHLCV parquet, clean, and index by UTC open_time."""
    df = pd.read_parquet(path, columns=["open_time", *BASE_COLS])
    # open_time may be ms epoch or datetime; normalise.
    ot = df["open_time"]
    if np.issubdtype(ot.dtype, np.number):
        ot = pd.to_datetime(ot, unit="ms", utc=True)
    else:
        ot = pd.to_datetime(ot, utc=True)
    df = df.assign(open_time=ot).set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # drop zero/negative-price or zero-activity rows (delisting tails, halts)
    df = df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)]
    return df


def load_base_equity_etf(path: str, rth: bool = False,
                         tz: str = "America/New_York") -> pd.DataFrame:
    """Load an Algoseek ETF 1-min csv.gz into the standard base-bar schema.

    Equities have no taker buy/sell split, so order-flow imbalance is set
    neutral (buy = sell = half dollar) — the bar-statistics study does not use
    it. Dollar notional = Volume * VWAP.

    BarDateTime is naive US-Eastern. If rth=True, localize to `tz` and keep only
    regular trading hours (09:30-16:00 ET); otherwise index is UTC-naive as-is.
    """
    cols = ["BarDateTime", "FirstTradePrice", "HighTradePrice", "LowTradePrice",
            "LastTradePrice", "VolumeWeightPrice", "Volume", "TotalTrades"]
    df = pd.read_csv(path, compression="gzip", usecols=cols)
    if rth:
        t = (pd.to_datetime(df["BarDateTime"], errors="coerce")
             .dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT"))
    else:
        t = pd.to_datetime(df["BarDateTime"], utc=True, errors="coerce")
    out = pd.DataFrame({
        "open":  df["FirstTradePrice"].to_numpy(),
        "high":  df["HighTradePrice"].to_numpy(),
        "low":   df["LowTradePrice"].to_numpy(),
        "close": df["LastTradePrice"].to_numpy(),
        "volume": df["Volume"].to_numpy(),
        "count": df["TotalTrades"].to_numpy(),
    }, index=t)
    vwap = df["VolumeWeightPrice"].to_numpy()
    out["quote_volume"] = out["volume"].to_numpy() * np.where(np.isfinite(vwap) & (vwap > 0),
                                                              vwap, out["close"].to_numpy())
    out["taker_buy_volume"] = out["volume"] / 2.0
    out["taker_buy_quote_volume"] = out["quote_volume"] / 2.0
    out = out[(out["close"] > 0) & (out["volume"] > 0) & (out["count"] > 0)]
    out = out[~out.index.isna()].sort_index()
    if rth:
        mins = out.index.hour * 60 + out.index.minute
        out = out[(mins >= 9 * 60 + 30) & (mins < 16 * 60)]
    return out


def session_log_returns(bars: pd.DataFrame) -> pd.Series:
    """Within-session close-to-close log returns (drops the overnight gap).

    For markets with daily sessions (equities), the first bar of each calendar
    day has no valid prior bar inside the session, so its return is dropped —
    otherwise the overnight gap is miscounted as a bar return and fattens tails.
    """
    lp = np.log(bars["close"])
    r = lp.diff()
    day = bars.index.tz_convert("America/New_York").date if bars.index.tz is not None else bars.index.date
    same_day = pd.Series(day, index=bars.index)
    keep = same_day.values == pd.Series(day, index=bars.index).shift(1).values
    return r[keep].dropna()


try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if (a and callable(a[0])) is False else a[0]


@njit(cache=True)
def _agg_kernel(gid, ts, o, h, l, c, vol, dol, cnt, bqv):
    """Single forward pass over contiguous gid runs -> per-bar OHLCV + sums.

    gid is non-decreasing (cumulative-threshold or arange//k), so a new bar
    starts whenever gid changes. O(n), no Python-object timestamp iteration.
    """
    n = gid.shape[0]
    to = np.empty(n, np.int64); tc = np.empty(n, np.int64)
    oo = np.empty(n, np.float64); hh = np.empty(n, np.float64)
    ll = np.empty(n, np.float64); cc = np.empty(n, np.float64)
    sv = np.empty(n, np.float64); sd = np.empty(n, np.float64); sc = np.empty(n, np.float64)
    sb = np.empty(n, np.float64); se = np.empty(n, np.float64); nb = np.empty(n, np.int64)
    G = -1
    for i in range(n):
        if i == 0 or gid[i] != gid[i - 1]:
            G += 1
            to[G] = ts[i]; oo[G] = o[i]; hh[G] = h[i]; ll[G] = l[i]
            sv[G] = 0.0; sd[G] = 0.0; sc[G] = 0.0; sb[G] = 0.0; se[G] = 0.0; nb[G] = 0
        if h[i] > hh[G]:
            hh[G] = h[i]
        if l[i] < ll[G]:
            ll[G] = l[i]
        cc[G] = c[i]; tc[G] = ts[i]
        sv[G] += vol[i]; sd[G] += dol[i]; sc[G] += cnt[i]
        sb[G] += bqv[i]; se[G] += dol[i] - bqv[i]; nb[G] += 1
    G += 1
    return (to[:G], tc[:G], oo[:G], hh[:G], ll[:G], cc[:G],
            sv[:G], sd[:G], sc[:G], sb[:G], se[:G], nb[:G])


def _aggregate(df: pd.DataFrame, gid: np.ndarray) -> pd.DataFrame:
    """Aggregate base rows into bars defined by non-decreasing integer `gid`.

    Numba single-pass kernel on int64-ns timestamps (avoids pandas' Python-object
    timestamp iteration, the dominant cost in the pandas-groupby version).
    """
    tz = df.index.tz
    ts = df.index.asi8                                     # int64 ns since epoch (UTC)
    f = lambda col: np.ascontiguousarray(df[col].to_numpy(np.float64))
    to, tc, oo, hh, ll, cc, sv, sd, sc, sb, se, nb = _agg_kernel(
        np.ascontiguousarray(gid.astype(np.int64)), np.ascontiguousarray(ts),
        f("open"), f("high"), f("low"), f("close"), f("volume"),
        f("quote_volume"), f("count"), f("taker_buy_quote_volume"))
    idx = pd.to_datetime(tc, utc=True)
    t_open = pd.to_datetime(to, utc=True)
    if tz is not None:
        idx = idx.tz_convert(tz); t_open = t_open.tz_convert(tz)
    else:
        idx = idx.tz_localize(None); t_open = t_open.tz_localize(None)
    return pd.DataFrame({"t_open": t_open, "open": oo, "high": hh, "low": ll,
                         "close": cc, "volume": sv, "dollar": sd, "ticks": sc,
                         "buy_dollar": sb, "sell_dollar": se, "n_base": nb},
                        index=pd.DatetimeIndex(idx, name="t_close"))


def threshold_bars(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    """Sample a new bar each time cumulative `col` crosses a multiple of `threshold`.

    col: 'count' -> tick bars, 'volume' -> volume bars, 'quote_volume' -> dollar bars.
    Vectorised: gid = floor(cumsum(col)/threshold).
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    gid = np.floor(np.cumsum(df[col].to_numpy()) / threshold).astype(np.int64)
    return _aggregate(df, gid)


def time_bars(df: pd.DataFrame, base_per_bar: int) -> pd.DataFrame:
    """Fixed-clock bars: group every `base_per_bar` consecutive base bars."""
    if base_per_bar < 1:
        raise ValueError("base_per_bar must be >= 1")
    gid = (np.arange(len(df)) // base_per_bar).astype(np.int64)
    return _aggregate(df, gid)


def matched_bars(df: pd.DataFrame, n_target: int) -> dict[str, pd.DataFrame]:
    """Build time/tick/volume/dollar bars all targeting ~n_target bars.

    Thresholds are set to total/ n_target so each information-driven series has
    approximately the same average frequency as the time-bar control — the
    apples-to-apples setup LdP uses to compare statistical properties.
    """
    n_target = int(max(1, n_target))
    base_per_bar = max(1, round(len(df) / n_target))
    out = {
        "time":   time_bars(df, base_per_bar),
        "tick":   threshold_bars(df, "count",        df["count"].sum() / n_target),
        "volume": threshold_bars(df, "volume",       df["volume"].sum() / n_target),
        "dollar": threshold_bars(df, "quote_volume", df["quote_volume"].sum() / n_target),
    }
    return out


def log_returns(bars: pd.DataFrame) -> pd.Series:
    """Close-to-close log returns of a bar series."""
    return np.log(bars["close"]).diff().dropna()


def order_flow_imbalance(bars: pd.DataFrame) -> pd.Series:
    """Signed taker dollar imbalance per bar, normalised to [-1, 1]."""
    num = bars["buy_dollar"] - bars["sell_dollar"]
    den = (bars["buy_dollar"] + bars["sell_dollar"]).replace(0, np.nan)
    return (num / den).fillna(0.0)
