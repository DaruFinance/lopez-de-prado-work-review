"""
ta_grid.py — Numba-accelerated TA-strategy-grid engine (LdP-style market-agnostic
families) for the at-scale backtest-overfitting harness.

Sweeps the canonical technical families that LdP-style discretionary/quant rules
use — SMA crossover, EMA crossover, RSI level, MACD signal cross, ATR-channel
breakout, Stochastic — over fast/slow/threshold + stop-loss/take-profit grids.
Each (family x params x SL/TP) tuple is ONE structural strategy.

Design goals
------------
* Causal: a signal computed from bar t's CLOSE may only open a position at the
  OPEN of bar t+1 (no same-bar fill on the close that produced the signal).
* Intrabar OHLC exits: SL/TP are checked against each bar's HIGH/LOW (never
  close-only). SL is given priority over TP within a bar (conservative).
* Costed (REALISTIC, via lib/realism.py): per-fill cost is a per-bar, time-of-day
  half-spread + commission, charged on every entry and exit fill. FX adds a
  triple-Wednesday overnight swap on held positions and a weekend force-flat;
  equities add per-day short-borrow on shorts. No flat 5bp/3bp anymore.
* Session-aware (equities): positions are force-flat at the last bar of each
  trading day so no overnight gap is treated as a tradeable bar return.
* Session-aware (forex): positions are force-flat at the last open bar before the
  Fri 22:00 UTC weekend close; the real Fri->Sun price gap is kept (not clamped).
* RAM-bounded: features (indicators) are precomputed once per instrument; the
  hot sim loop streams strategies and writes per-strategy daily PnL.

The hot loop `_sim_grid_kernel` is a single Numba njit pass over all bars for one
parameter tuple; it is verified bit-identical against `_sim_one_ref` (pure NumPy).

Position convention: 1 long unit of notional (price units). PnL of a closed
trade = direction * (exit_px - entry_px) - costs, in price units. We aggregate to
daily pnl_sum per UTC calendar date. n_trades = closed trades attributed to the
exit date.
"""
from __future__ import annotations
import numpy as np
import realism as RZ

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        if a and callable(a[0]):
            return a[0]
        return deco
    def prange(*a):
        return range(*a)


# ---------------------------------------------------------------------------
# Indicator primitives (vectorised, causal — value at bar t uses bars <= t)
# ---------------------------------------------------------------------------
def sma(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    out = np.full(x.shape[0], np.nan)
    if n < 1 or n > x.shape[0]:
        return out
    c = np.cumsum(x)
    out[n - 1:] = (c[n - 1:] - np.concatenate(([0.0], c[:-n]))) / n
    return out


@njit(cache=True)
def _ema_k(x, n):
    m = x.shape[0]
    out = np.empty(m, np.float64)
    if m == 0:
        return out
    a = 2.0 / (n + 1.0)
    out[0] = x[0]
    for i in range(1, m):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    if n < 1 or x.shape[0] == 0:
        return np.full(x.shape[0], np.nan)
    return _ema_k(np.ascontiguousarray(x, np.float64), n)


@njit(cache=True)
def _rsi_k(close, n):
    m = close.shape[0]
    out = np.full(m, np.nan)
    ag = 0.0; al = 0.0
    for i in range(1, n + 1):
        d = close[i] - close[i - 1]
        if d > 0:
            ag += d
        else:
            al += -d
    ag /= n; al /= n
    rs = ag / al if al > 0 else 1e18
    out[n] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(n + 1, m):
        d = close[i] - close[i - 1]
        g = d if d > 0 else 0.0
        ls = -d if d < 0 else 0.0
        ag = (ag * (n - 1) + g) / n
        al = (al * (n - 1) + ls) / n
        rs = ag / al if al > 0 else 1e18
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RSI (causal). Returns NaN until n deltas are available."""
    n = int(n)
    if close.shape[0] <= n or n < 1:
        return np.full(close.shape[0], np.nan)
    return _rsi_k(np.ascontiguousarray(close, np.float64), n)


@njit(cache=True)
def _atr_k(high, low, close, n):
    m = close.shape[0]
    out = np.full(m, np.nan)
    tr_sum = 0.0
    prev = high[0] - low[0]
    for i in range(1, n + 1):
        a = high[i] - low[i]
        b = high[i] - close[i - 1]; b = b if b >= 0 else -b
        c = low[i] - close[i - 1];  c = c if c >= 0 else -c
        t = a if (a >= b and a >= c) else (b if b >= c else c)
        tr_sum += t
    out[n] = tr_sum / n
    for i in range(n + 1, m):
        a = high[i] - low[i]
        b = high[i] - close[i - 1]; b = b if b >= 0 else -b
        c = low[i] - close[i - 1];  c = c if c >= 0 else -c
        t = a if (a >= b and a >= c) else (b if b >= c else c)
        out[i] = (out[i - 1] * (n - 1) + t) / n
    return out


def atr(high, low, close, n: int) -> np.ndarray:
    """Wilder's ATR (causal)."""
    n = int(n)
    if close.shape[0] <= n or n < 1:
        return np.full(close.shape[0], np.nan)
    return _atr_k(np.ascontiguousarray(high, np.float64),
                  np.ascontiguousarray(low, np.float64),
                  np.ascontiguousarray(close, np.float64), n)


@njit(cache=True)
def _stoch_k_k(high, low, close, n):
    m = close.shape[0]
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        hh = high[i]; ll = low[i]
        for j in range(i - n + 1, i + 1):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        rng = hh - ll
        out[i] = 50.0 if rng <= 0 else 100.0 * (close[i] - ll) / rng
    return out


def stoch_k(high, low, close, n: int) -> np.ndarray:
    """Fast %K (causal): 100*(close-LL)/(HH-LL) over trailing n bars."""
    n = int(n)
    if close.shape[0] < n or n < 1:
        return np.full(close.shape[0], np.nan)
    return _stoch_k_k(np.ascontiguousarray(high, np.float64),
                      np.ascontiguousarray(low, np.float64),
                      np.ascontiguousarray(close, np.float64), n)


# ---------------------------------------------------------------------------
# Signal builders -> per-bar desired position in {-1,0,+1}, CAUSAL (uses <= t).
# The sim applies the signal at bar t with a 1-bar delay (fills at t+1 open).
# ---------------------------------------------------------------------------
def signal_ma_cross(fast_line, slow_line, allow_short: bool) -> np.ndarray:
    sig = np.zeros(fast_line.shape[0], np.int8)
    valid = np.isfinite(fast_line) & np.isfinite(slow_line)
    up = valid & (fast_line > slow_line)
    dn = valid & (fast_line < slow_line)
    sig[up] = 1
    sig[dn] = -1 if allow_short else 0
    return sig


def signal_rsi(rsi_line, lo: float, hi: float, allow_short: bool) -> np.ndarray:
    """Mean-reversion: long when RSI < lo (oversold), short/flat when RSI > hi."""
    sig = np.zeros(rsi_line.shape[0], np.int8)
    v = np.isfinite(rsi_line)
    sig[v & (rsi_line < lo)] = 1
    sig[v & (rsi_line > hi)] = -1 if allow_short else 0
    return sig


def signal_macd(close, fast, slow, signal_n, allow_short: bool):
    macd = ema(close, fast) - ema(close, slow)
    sig_line = ema(np.nan_to_num(macd), signal_n)
    sig = np.zeros(close.shape[0], np.int8)
    valid = np.isfinite(macd) & np.isfinite(sig_line)
    sig[valid & (macd > sig_line)] = 1
    sig[valid & (macd < sig_line)] = -1 if allow_short else 0
    # mask the leading region where the longer EMA has not warmed up
    warm = max(slow, fast) + signal_n
    sig[:warm] = 0
    return sig


def signal_atr_channel(close, atr_line, n_ma, mult, allow_short: bool):
    """Breakout: long when close > SMA(n)+mult*ATR, short when < SMA-mult*ATR."""
    mid = sma(close, n_ma)
    upper = mid + mult * atr_line
    lower = mid - mult * atr_line
    sig = np.zeros(close.shape[0], np.int8)
    v = np.isfinite(upper) & np.isfinite(lower)
    sig[v & (close > upper)] = 1
    sig[v & (close < lower)] = -1 if allow_short else 0
    return sig


def signal_stoch(k_line, lo: float, hi: float, allow_short: bool):
    sig = np.zeros(k_line.shape[0], np.int8)
    v = np.isfinite(k_line)
    sig[v & (k_line < lo)] = 1
    sig[v & (k_line > hi)] = -1 if allow_short else 0
    return sig


# ---------------------------------------------------------------------------
# Core sim — Numba hot loop. Operates on a precomputed per-bar target position.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _sim_grid_kernel(open_, high, low, close, day_id, sig,
                     cost_frac, cost_abs, sl_frac, tp_frac, allow_short,
                     force_flat, rollover_flag, swap_mult, swap_px_night,
                     borrow_daily, day_date_i8):
    """One pass over all bars for ONE strategy, with REALISTIC frictions.

    Args
    ----
    open_,high,low,close : f64[n] bar OHLC
    day_id               : i32[n] non-decreasing calendar-day id (force-flat at
                           the last bar of each day when its id changes next bar)
    sig                  : i8[n] desired position in {-1,0,1} computed at bar t
                           from data <= t. Applied with 1-bar delay (fill t+1 open).
    cost_frac            : f64[n] per-FILL cost as FRACTION of fill price, indexed
                           by the FILL bar (equity: half-spread bp + commission).
    cost_abs             : f64[n] per-FILL cost in ABSOLUTE price units, indexed by
                           the FILL bar (FX: time-of-day half-spread in price).
                           Total per-fill cost = cost_frac[j]*fill_px + cost_abs[j].
    sl_frac, tp_frac     : stop / target as fraction of entry price (<=0 disables)
    force_flat           : i8[n] 1 at a bar that forces a flat close at this bar's
                           CLOSE (FX last-open-bar-before-weekend); 0 disables.
    rollover_flag        : i8[n] 1 at the daily rollover bar (FX swap accrual bar).
    swap_mult            : f64[n] swap multiplier at each bar (3.0 Wed, else 1.0).
    swap_px_night        : f64 scalar swap debit/night in price units (FX; 0 = off).
    borrow_daily         : f64 scalar daily short-borrow fraction (equity; 0 = off).
    day_date_i8          : i64[n] per-bar calendar-date key (for counting borrow
                           days held); only differences matter. 0-array disables.

    Returns
    -------
    day_pnl  : f64[n_days] summed PnL per day (price units)
    day_ntr  : i32[n_days] closed trades per day (attributed to exit day)
    """
    n = open_.shape[0]
    n_days = 0
    for i in range(n):
        if day_id[i] + 1 > n_days:
            n_days = day_id[i] + 1
    day_pnl = np.zeros(n_days, np.float64)
    day_ntr = np.zeros(n_days, np.int32)

    pos = 0            # current position direction in {-1,0,1}
    entry_px = 0.0
    entry_cost = 0.0   # cost charged at the entry fill (price units)
    fin_accrued = 0.0  # financing (swap/borrow) accrued over the life of the trade
    for i in range(n - 1):
        # ---- manage an open position over bar i+1 using intrabar OHLC ----
        # (entries were filled at open of i+1 in a PRIOR iteration; here bar i+1
        #  is the *next* bar. We process exits on the bar AFTER entry onward.)
        nxt = i + 1
        last_of_day = (day_id[nxt] != day_id[nxt + 1]) if nxt + 1 < n else True
        forced = (force_flat[nxt] == 1)

        if pos != 0:
            # ---- accrue financing on the held position at this bar ----
            # FX swap: charged when this bar is the daily rollover bar (debit both
            #          sides; conservative negative carry), x3 on Wednesday.
            if swap_px_night > 0.0 and rollover_flag[nxt] == 1:
                fin_accrued += swap_px_night * swap_mult[nxt]
            # Equity short borrow: charged per new calendar day held, shorts only.
            if borrow_daily > 0.0 and pos < 0 and day_date_i8[nxt] != day_date_i8[i]:
                fin_accrued += borrow_daily * entry_px

            o = open_[nxt]; h = high[nxt]; l = low[nxt]; c = close[nxt]
            exit_px = 0.0
            did_exit = 0
            if pos > 0:
                sl_px = entry_px * (1.0 - sl_frac) if sl_frac > 0.0 else -1.0
                tp_px = entry_px * (1.0 + tp_frac) if tp_frac > 0.0 else -1.0
                # gap through stop at open
                if sl_px > 0.0 and o <= sl_px:
                    exit_px = o; did_exit = 1
                elif tp_px > 0.0 and o >= tp_px:
                    exit_px = o; did_exit = 1
                elif sl_px > 0.0 and l <= sl_px:      # SL priority within bar
                    exit_px = sl_px; did_exit = 1
                elif tp_px > 0.0 and h >= tp_px:
                    exit_px = tp_px; did_exit = 1
            else:
                sl_px = entry_px * (1.0 + sl_frac) if sl_frac > 0.0 else -1.0
                tp_px = entry_px * (1.0 - tp_frac) if tp_frac > 0.0 else -1.0
                if sl_px > 0.0 and o >= sl_px:
                    exit_px = o; did_exit = 1
                elif tp_px > 0.0 and o <= tp_px:
                    exit_px = o; did_exit = 1
                elif sl_px > 0.0 and h >= sl_px:
                    exit_px = sl_px; did_exit = 1
                elif tp_px > 0.0 and l <= tp_px:
                    exit_px = tp_px; did_exit = 1

            # signal flip, session end, or forced weekend-flat closes at this close
            want = sig[i]                            # target decided at bar i, act at i+1
            if not allow_short and want < 0:
                want = 0
            if did_exit == 0 and (want != pos or last_of_day or forced):
                exit_px = c; did_exit = 1

            if did_exit == 1:
                gross = pos * (exit_px - entry_px)
                exit_cost = cost_frac[nxt] * exit_px + cost_abs[nxt]
                day_pnl[day_id[nxt]] += gross - entry_cost - exit_cost - fin_accrued
                day_ntr[day_id[nxt]] += 1
                pos = 0
                fin_accrued = 0.0

        # ---- open a new position at the OPEN of bar i+1 if flat and signalled ----
        # never open on a force-flat bar (would be flattened immediately).
        if pos == 0 and not last_of_day and not forced:
            want = sig[i]
            if not allow_short and want < 0:
                want = 0
            if want != 0:
                pos = want
                entry_px = open_[nxt]
                entry_cost = cost_frac[nxt] * entry_px + cost_abs[nxt]
                fin_accrued = 0.0

    return day_pnl, day_ntr


# Pure-NumPy reference (no numba) for bit-identical verification.
def _sim_one_ref(open_, high, low, close, day_id, sig,
                 cost_frac, cost_abs, sl_frac, tp_frac, allow_short,
                 force_flat, rollover_flag, swap_mult, swap_px_night,
                 borrow_daily, day_date_i8):
    n = open_.shape[0]
    n_days = int(day_id.max()) + 1 if n else 0
    day_pnl = np.zeros(n_days, np.float64)
    day_ntr = np.zeros(n_days, np.int32)
    pos = 0; entry_px = 0.0; entry_cost = 0.0; fin_accrued = 0.0
    for i in range(n - 1):
        nxt = i + 1
        last_of_day = (day_id[nxt] != day_id[nxt + 1]) if nxt + 1 < n else True
        forced = (force_flat[nxt] == 1)
        if pos != 0:
            if swap_px_night > 0.0 and rollover_flag[nxt] == 1:
                fin_accrued += swap_px_night * swap_mult[nxt]
            if borrow_daily > 0.0 and pos < 0 and day_date_i8[nxt] != day_date_i8[i]:
                fin_accrued += borrow_daily * entry_px
            o = open_[nxt]; h = high[nxt]; l = low[nxt]; c = close[nxt]
            exit_px = 0.0; did_exit = 0
            if pos > 0:
                sl_px = entry_px * (1.0 - sl_frac) if sl_frac > 0.0 else -1.0
                tp_px = entry_px * (1.0 + tp_frac) if tp_frac > 0.0 else -1.0
                if sl_px > 0.0 and o <= sl_px:
                    exit_px = o; did_exit = 1
                elif tp_px > 0.0 and o >= tp_px:
                    exit_px = o; did_exit = 1
                elif sl_px > 0.0 and l <= sl_px:
                    exit_px = sl_px; did_exit = 1
                elif tp_px > 0.0 and h >= tp_px:
                    exit_px = tp_px; did_exit = 1
            else:
                sl_px = entry_px * (1.0 + sl_frac) if sl_frac > 0.0 else -1.0
                tp_px = entry_px * (1.0 - tp_frac) if tp_frac > 0.0 else -1.0
                if sl_px > 0.0 and o >= sl_px:
                    exit_px = o; did_exit = 1
                elif tp_px > 0.0 and o <= tp_px:
                    exit_px = o; did_exit = 1
                elif sl_px > 0.0 and h >= sl_px:
                    exit_px = sl_px; did_exit = 1
                elif tp_px > 0.0 and l <= tp_px:
                    exit_px = tp_px; did_exit = 1
            want = sig[i]
            if not allow_short and want < 0:
                want = 0
            if did_exit == 0 and (want != pos or last_of_day or forced):
                exit_px = c; did_exit = 1
            if did_exit == 1:
                gross = pos * (exit_px - entry_px)
                exit_cost = cost_frac[nxt] * exit_px + cost_abs[nxt]
                day_pnl[day_id[nxt]] += gross - entry_cost - exit_cost - fin_accrued
                day_ntr[day_id[nxt]] += 1
                pos = 0; fin_accrued = 0.0
        if pos == 0 and not last_of_day and not forced:
            want = sig[i]
            if not allow_short and want < 0:
                want = 0
            if want != 0:
                pos = want; entry_px = open_[nxt]
                entry_cost = cost_frac[nxt] * entry_px + cost_abs[nxt]
                fin_accrued = 0.0
    return day_pnl, day_ntr


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
# DEPRECATED: the old flat per-fill cost. Costs now come from lib/realism.py via
# build_friction() (time-of-day half-spread + commission + FX swap/weekend +
# equity borrow). Kept only as legacy references; the kernel no longer uses them.
DEFAULT_FEE_BP = 5.0      # 0.05% per fill (DEPRECATED — see lib/realism.py)
DEFAULT_SLIP_BP = 3.0     # 0.03% per fill (DEPRECATED — see lib/realism.py)
SL_GRID = (0.0, 0.01, 0.02, 0.04)        # stop-loss fractions (0 = none)
TP_GRID = (0.0, 0.02, 0.04, 0.08)        # take-profit fractions (0 = none)


def _ma_pairs(fast_set, slow_set):
    out = []
    for f in fast_set:
        for s in slow_set:
            if f < s:
                out.append((f, s))
    return out


def build_grid(wide: bool = False):
    """Return a list of (family, name, params_dict) structural-strategy specs.

    `wide=False` ~ 2,500 combos (smoke/default). `wide=True` widens fast/slow and
    level grids to reach ~5,000+ combos per instrument for the overnight run.
    """
    if not wide:
        sma_fast = (5, 10, 20, 30, 50); sma_slow = (20, 50, 100, 150, 200)
        ema_fast = (5, 10, 20, 30, 50); ema_slow = (20, 50, 100, 150, 200)
        rsi_n = (7, 14, 21); rsi_lo = (20, 30); rsi_hi = (70, 80)
        macd = ((12, 26, 9), (5, 35, 5), (8, 21, 5), (10, 30, 7))
        atr_n = (14, 20); atr_ma = (20, 50); atr_mult = (1.5, 2.5)
        stoch_n = (9, 14, 21); st_lo = (20,); st_hi = (80,)
    else:
        sma_fast = (5, 10, 15, 20, 30, 50); sma_slow = (20, 40, 50, 100, 150, 200)
        ema_fast = (5, 10, 15, 20, 30, 50); ema_slow = (20, 40, 50, 100, 150, 200)
        rsi_n = (5, 7, 10, 14, 21); rsi_lo = (15, 20, 25, 30); rsi_hi = (70, 75, 80, 85)
        macd = ((12, 26, 9), (5, 35, 5), (8, 21, 5), (10, 30, 7), (6, 19, 9))
        atr_n = (10, 14, 20); atr_ma = (20, 50, 100); atr_mult = (1.0, 1.5, 2.0, 2.5, 3.0)
        stoch_n = (5, 9, 14, 21); st_lo = (15, 20, 25); st_hi = (75, 80, 85)

    specs = []
    for allow_short in (False, True):
        sfx = "_LS" if allow_short else "_LO"   # long-short vs long-only
        for f, s in _ma_pairs(sma_fast, sma_slow):
            for sl in SL_GRID:
                for tp in TP_GRID:
                    specs.append(("SMA", f"SMA_{f}_{s}_SL{sl}_TP{tp}{sfx}",
                                  dict(kind="ma", ma="sma", fast=f, slow=s,
                                       sl=sl, tp=tp, allow_short=allow_short)))
        for f, s in _ma_pairs(ema_fast, ema_slow):
            for sl in SL_GRID:
                for tp in TP_GRID:
                    specs.append(("EMA", f"EMA_{f}_{s}_SL{sl}_TP{tp}{sfx}",
                                  dict(kind="ma", ma="ema", fast=f, slow=s,
                                       sl=sl, tp=tp, allow_short=allow_short)))
        for nn in rsi_n:
            for lo in rsi_lo:
                for hi in rsi_hi:
                    for sl in SL_GRID:
                        for tp in (0.0, 0.02, 0.04):
                            specs.append(("RSI", f"RSI_{nn}_{lo}_{hi}_SL{sl}_TP{tp}{sfx}",
                                          dict(kind="rsi", n=nn, lo=lo, hi=hi,
                                               sl=sl, tp=tp, allow_short=allow_short)))
        for (mf, ms, mg) in macd:
            for sl in SL_GRID:
                for tp in TP_GRID:
                    specs.append(("MACD", f"MACD_{mf}_{ms}_{mg}_SL{sl}_TP{tp}{sfx}",
                                  dict(kind="macd", fast=mf, slow=ms, signal=mg,
                                       sl=sl, tp=tp, allow_short=allow_short)))
        for an in atr_n:
            for am in atr_ma:
                for amu in atr_mult:
                    for sl in SL_GRID:
                        for tp in (0.0, 0.04, 0.08):
                            specs.append(("ATR", f"ATR_{an}_{am}_{amu}_SL{sl}_TP{tp}{sfx}",
                                          dict(kind="atr", atr_n=an, ma=am, mult=amu,
                                               sl=sl, tp=tp, allow_short=allow_short)))
        for sn in stoch_n:
            for lo in st_lo:
                for hi in st_hi:
                    for sl in SL_GRID:
                        for tp in TP_GRID:
                            specs.append(("STOCH", f"STOCH_{sn}_{lo}_{hi}_SL{sl}_TP{tp}{sfx}",
                                          dict(kind="stoch", n=sn, lo=lo, hi=hi,
                                               sl=sl, tp=tp, allow_short=allow_short)))
    return specs


def make_signal(spec_params, o, h, l, c):
    """Build the causal {-1,0,1} target-position array for one spec."""
    p = spec_params
    ash = p["allow_short"]
    k = p["kind"]
    if k == "ma":
        line = sma if p["ma"] == "sma" else ema
        return signal_ma_cross(line(c, p["fast"]), line(c, p["slow"]), ash)
    if k == "rsi":
        return signal_rsi(rsi(c, p["n"]), p["lo"], p["hi"], ash)
    if k == "macd":
        return signal_macd(c, p["fast"], p["slow"], p["signal"], ash)
    if k == "atr":
        return signal_atr_channel(c, atr(h, l, c, p["atr_n"]), p["ma"], p["mult"], ash)
    if k == "stoch":
        return signal_stoch(stoch_k(h, l, c, p["n"]), p["lo"], p["hi"], ash)
    raise ValueError(f"unknown kind {k}")


# ---------------------------------------------------------------------------
# Realistic-friction array builders (per instrument, used by the driver).
# ---------------------------------------------------------------------------
def build_friction(market: str, symbol: str, index, close, n: int):
    """Build the per-bar friction arrays the sim kernel consumes for one
    instrument. `market` in {"equity","forex","crypto"}; `index` is the bar-close
    DatetimeIndex (tz-aware: ET for equity, UTC for forex); `close` is f64[n].

    Returns the 6-tuple of kernel arrays:
        (cost_frac, cost_abs, force_flat, rollover_flag, swap_mult,
         swap_px_night, borrow_daily, day_date_i8)  -- 8 items, see kernel.

    EQUITY  : cost_frac = time-of-day half-spread (bp) + min-ticket commission as
              a fraction of price; cost_abs = 0; force_flat handled by day-ids;
              short borrow accrued per ET calendar day on shorts.
    FOREX   : cost_abs = time-of-day half-spread in PRICE units; cost_frac = 0;
              force_flat at last open bar before weekend; triple-Wed swap on the
              daily rollover bar.
    CRYPTO  : unchanged-style flat cost (handled by caller, not here) -- this
              builder returns frictionless arrays so the caller keeps its own
              production-framework costs. (We never recost crypto here.)
    """
    z_f = np.zeros(n, np.float64)
    z_i8 = np.zeros(n, np.int8)
    one_f = np.ones(n, np.float64)
    z_i64 = np.zeros(n, np.int64)
    if market == "forex":
        cost_abs, force_flat, rollover, swap_mult, swap_night = \
            RZ.build_fx_friction_arrays(index, symbol)
        return (z_f, cost_abs, force_flat, rollover, swap_mult,
                swap_night, 0.0, z_i64)
    if market == "equity":
        cost_frac, borrow = RZ.build_equity_friction_arrays(index, close, symbol)
        # per-ET-calendar-day key for borrow accrual on shorts
        idx = index.tz_convert("America/New_York") if index.tz is not None \
            else index.tz_localize("America/New_York")
        day_key = np.ascontiguousarray(idx.normalize().asi8, np.int64)
        return (cost_frac, z_f, z_i8, z_i8, one_f, 0.0, borrow, day_key)
    # crypto / unknown: frictionless arrays (caller supplies its own costs)
    return (z_f, z_f, z_i8, z_i8, one_f, 0.0, 0.0, z_i64)


def sim_one(o, h, l, c, day_id, sig, friction, sl_frac, tp_frac, allow_short,
            ref=False):
    """Run ONE strategy through the realistic kernel (or the pure-Python ref if
    ref=True). `friction` is the 8-tuple from `build_friction`."""
    (cost_frac, cost_abs, force_flat, rollover, swap_mult,
     swap_night, borrow, day_key) = friction
    fn = _sim_one_ref if ref else _sim_grid_kernel
    return fn(o, h, l, c, day_id, sig, cost_frac, cost_abs,
              float(sl_frac), float(tp_frac), bool(allow_short),
              force_flat, rollover, swap_mult, float(swap_night),
              float(borrow), day_key)
