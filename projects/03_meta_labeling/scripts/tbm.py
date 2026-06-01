"""
tbm.py, Triple-Barrier Labeling + Meta-Labeling (Lopez de Prado, AFML Ch.3, ML4AM Ch.5).

Self-contained library imported by run_meta_labeling.py. Two pieces:

  1. TRIPLE-BARRIER LABELING (the realised-outcome labeller)
     For each event t0 with a known SIDE s in {+1,-1}, set
        - upper (profit-take) horizontal barrier  at  +pt * sigma_t0   (in return space, side-adjusted)
        - lower (stop-loss)   horizontal barrier  at  -sl * sigma_t0
        - vertical (max-hold)  barrier            at  t0 + max_hold bars
     Scan FORWARD using full intrabar OHLC (high/low, NOT close-only) and record
     which barrier is touched FIRST. The realised path-return at first touch
     (side-signed, minus costs) is the trade P&L; the meta-label is 1[pnl>0].

  2. META-LABELING
     A PRIMARY model fixes the side (here a structural MA-crossover / momentum
     signal). Triple-barrier gives the realised outcome. A SECONDARY classifier
     (bagged trees) predicts P(primary's bet is profitable) from causal features.
     The meta-label decides whether to ACT and the BET SIZE (size ~ p, or the
     LdP m = 2*Phi(z)-1 sizing). It cannot flip the side; it can only veto / size.

Volatility = rolling EWMA of close-to-close returns (causal). Barriers / holding
/ lookbacks are IS-TUNABLE knobs sampled per WFO window, not separate strategies.

All loops that scan bars are Numba @njit; a pure-pandas reference
(`triple_barrier_reference`) verifies the kernel is bit-identical.
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


def load_base_crypto(path: str) -> pd.DataFrame:
    """Robust crypto base loader (mirrors lib.bars.load_base but tolerates an
    already-tz-aware datetime64 open_time, which the shared loader's
    np.issubdtype check rejects under newer NumPy). Read-only use of the data."""
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
    """FX base loader: HISTDATA-style 1m parquet with open/high/low/close/count
    and NO volume. Add proxy activity cols (volume=count, quote_volume=count,
    taker_buy_quote_volume=count/2) so tick bars (threshold_bars on 'count') can
    be built by the shared aggregator."""
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
# Volatility (causal EWMA of returns), LdP getDailyVol analogue
# --------------------------------------------------------------------------- #
def ewma_vol(close: np.ndarray, span: int) -> np.ndarray:
    """Causal EWMA standard deviation of close-to-close log returns.

    Output[t] uses returns up to and including t. r[0]=0. Matches pandas
    .ewm(span).std() on the return series (bias-corrected, adjust=True)."""
    lp = np.log(close)
    r = np.empty_like(lp)
    r[0] = 0.0
    r[1:] = lp[1:] - lp[:-1]
    s = pd.Series(r)
    vol = s.ewm(span=span, adjust=True).std().to_numpy()
    vol[~np.isfinite(vol)] = 0.0
    return vol


# --------------------------------------------------------------------------- #
# Triple-barrier first-touch scan, Numba kernel
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _triple_barrier_kernel(ev_idx, side, pt_mult, sl_mult, sigma, max_hold,
                           open_, high, low, close):
    """First-touch triple-barrier scan using full intrabar OHLC.

    For event k starting at bar i0 = ev_idx[k] with side s:
      entry price = close[i0]  (signal known at close of i0, enter at that close)
      up barrier  : entry * exp(+pt_mult*sigma[i0])
      dn barrier  : entry * exp(-sl_mult*sigma[i0])
      vertical    : i0 + max_hold  (or last bar)
    Scan bars j = i0+1 .. i0+max_hold. The side determines which barrier is the
    profit-take: for s=+1 PT=up, SL=dn; for s=-1 PT=dn (price falling is profit),
    SL=up. We test the SL barrier first within a bar when both are touched in the
    same bar (conservative: assume the adverse extreme hit first), matching the
    standard pessimistic intrabar convention.

    Returns per event: touch_bar (abs idx of exit), label (+1/-1/0 = pt/sl/vert),
    ret (side-signed gross log-return at exit, BEFORE costs), hold (bars held)."""
    n_ev = ev_idx.shape[0]
    n = close.shape[0]
    out_touch = np.empty(n_ev, np.int64)
    out_label = np.empty(n_ev, np.int64)
    out_ret = np.empty(n_ev, np.float64)
    out_hold = np.empty(n_ev, np.int64)
    for k in range(n_ev):
        i0 = ev_idx[k]
        s = side[k]
        entry = close[i0]
        sig = sigma[i0]
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
            # Pessimistic ordering: check the adverse (stop) barrier first.
            if s > 0:
                if lo <= dn:                      # stop-loss for a long
                    touched = j; label = -1; exit_px = dn; break
                if hi >= up:                      # profit-take for a long
                    touched = j; label = 1; exit_px = up; break
            else:
                if hi >= up:                      # stop-loss for a short
                    touched = j; label = -1; exit_px = up; break
                if lo <= dn:                      # profit-take for a short
                    touched = j; label = 1; exit_px = dn; break
        if touched < 0:                           # vertical barrier
            touched = j_end
            label = 0
            exit_px = close[j_end]
        gross = s * (np.log(exit_px) - np.log(entry))   # side-signed log return
        out_touch[k] = touched
        out_label[k] = label
        out_ret[k] = gross
        out_hold[k] = touched - i0
    return out_touch, out_label, out_ret, out_hold


def triple_barrier(close, high, low, open_, ev_idx, side, sigma,
                   pt_mult, sl_mult, max_hold):
    """Driver wrapper around the Numba kernel. Returns a DataFrame of outcomes."""
    ev_idx = np.ascontiguousarray(np.asarray(ev_idx, np.int64))
    side = np.ascontiguousarray(np.asarray(side, np.int64))
    f = lambda x: np.ascontiguousarray(np.asarray(x, np.float64))
    t, lab, ret, hold = _triple_barrier_kernel(
        ev_idx, side, float(pt_mult), float(sl_mult), f(sigma), int(max_hold),
        f(open_), f(high), f(low), f(close))
    return pd.DataFrame({"ev_idx": ev_idx, "side": side, "touch": t,
                         "label": lab, "ret_gross": ret, "hold": hold})


def triple_barrier_reference(close, high, low, ev_idx, side, sigma,
                             pt_mult, sl_mult, max_hold):
    """Pure-Python/NumPy reference implementation for bit-identical verification.

    Deliberately written independently of the kernel (no Numba), same convention
    (pessimistic: adverse barrier checked first within a bar)."""
    close = np.asarray(close, float); high = np.asarray(high, float)
    low = np.asarray(low, float); sigma = np.asarray(sigma, float)
    n = len(close)
    rows = []
    for k in range(len(ev_idx)):
        i0 = int(ev_idx[k]); s = int(side[k])
        entry = close[i0]; sig = sigma[i0]
        up = entry * np.exp(pt_mult * sig)
        dn = entry * np.exp(-sl_mult * sig)
        j_end = min(i0 + max_hold, n - 1)
        touched = -1; label = 0; exit_px = close[j_end]
        for j in range(i0 + 1, j_end + 1):
            if s > 0:
                if low[j] <= dn:
                    touched, label, exit_px = j, -1, dn; break
                if high[j] >= up:
                    touched, label, exit_px = j, 1, up; break
            else:
                if high[j] >= up:
                    touched, label, exit_px = j, -1, up; break
                if low[j] <= dn:
                    touched, label, exit_px = j, 1, dn; break
        if touched < 0:
            touched, label, exit_px = j_end, 0, close[j_end]
        gross = s * (np.log(exit_px) - np.log(entry))
        rows.append((i0, s, touched, label, gross, touched - i0))
    return pd.DataFrame(rows, columns=["ev_idx", "side", "touch", "label",
                                       "ret_gross", "hold"])


# --------------------------------------------------------------------------- #
# Primary signals (structural side decisions), causal
# --------------------------------------------------------------------------- #
def primary_ma_crossover(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Side = sign(EMA_fast - EMA_slow), evaluated at the close of each bar.

    Causal: at bar t uses only closes <= t. An event fires at every bar where a
    side is defined (non-zero); the caller subsamples to the crossover instants
    (where side changes) to avoid massively overlapping events."""
    c = pd.Series(close)
    ef = c.ewm(span=fast, adjust=True).mean()
    es = c.ewm(span=slow, adjust=True).mean()
    side = np.sign((ef - es).to_numpy())
    side[~np.isfinite(side)] = 0.0
    return side.astype(np.int64)


def crossover_events(side: np.ndarray) -> np.ndarray:
    """Bar indices where the MA-crossover side changes (the entry instants)."""
    s = side
    chg = np.where((s[1:] != s[:-1]) & (s[1:] != 0))[0] + 1
    return chg.astype(np.int64)


# --------------------------------------------------------------------------- #
# Causal features for the secondary (meta) model
# --------------------------------------------------------------------------- #
def build_features(bars: pd.DataFrame, side: np.ndarray, vol: np.ndarray,
                   fast: int, slow: int) -> pd.DataFrame:
    """Causal feature matrix evaluated at each bar (use rows at event indices).

    All features use only information available at the close of the bar. No
    forward-looking quantities. Features:
      side            : the primary's proposed direction (+1/-1)
      vol             : current EWMA vol (regime)
      ma_gap          : (EMA_fast-EMA_slow)/close  (signal strength)
      mom_*           : trailing log-return over k bars (3 horizons)
      rsi             : Wilder RSI(14) mapped to [-1,1]
      vol_ratio       : short vol / long vol (vol regime change)
      ofi             : order-flow imbalance (taker buy-sell), 0 for equities/fx
      range_atr       : (high-low)/close smoothed (intrabar range)
    """
    c = bars["close"]
    lp = np.log(c)
    r = lp.diff()
    ef = c.ewm(span=fast, adjust=True).mean()
    es = c.ewm(span=slow, adjust=True).mean()
    ma_gap = ((ef - es) / c).to_numpy()

    def mom(k):
        return (lp - lp.shift(k)).to_numpy()

    # Wilder RSI(14)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50.0)
    rsi_scaled = ((rsi - 50.0) / 50.0).to_numpy()

    vol_s = r.ewm(span=max(2, fast)).std().to_numpy()
    vol_l = r.ewm(span=max(4, slow)).std().to_numpy()
    vol_ratio = np.where(vol_l > 0, vol_s / vol_l, 1.0)

    if {"buy_dollar", "sell_dollar"}.issubset(bars.columns):
        num = bars["buy_dollar"] - bars["sell_dollar"]
        den = (bars["buy_dollar"] + bars["sell_dollar"]).replace(0, np.nan)
        ofi = (num / den).fillna(0.0).to_numpy()
    else:
        ofi = np.zeros(len(bars))

    rng = ((bars["high"] - bars["low"]) / c).ewm(span=fast).mean().to_numpy()

    feat = pd.DataFrame({
        "side": side.astype(float),
        "vol": vol,
        "ma_gap": ma_gap,
        "mom3": mom(3), "mom6": mom(6), "mom12": mom(12),
        "rsi": rsi_scaled,
        "vol_ratio": vol_ratio,
        "ofi": ofi,
        "range_atr": rng,
    }, index=range(len(bars)))
    return feat
