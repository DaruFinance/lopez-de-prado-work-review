"""
realism.py, single source of truth for REALISTIC, retail-realistic frictions
for US-EQUITY (ETF) and FOREX (spot) backtests in the LdP review program.

Design rules
------------
* CAUSAL: every friction uses only information known at or before bar t. Session
  calendars, time-of-day spread schedules, rollover/borrow accruals and pip-value
  are all known ex-ante (they are deterministic functions of the timestamp and the
  instrument), so applying them at bar t is not look-ahead.
* NO CLAMPING: we model frictions and weekend/overnight GAPS as real risk and let
  PnL fall where it falls. The Fri-close -> Sun-open FX gap and the equity
  overnight gap are genuine; we never floor/cap a trade's PnL.
* NUMBA-FRIENDLY: the per-bar arrays produced here (half-spread cost-rate, overnight
  swap accrual, borrow accrual, session/last-bar flags) are plain float/int numpy
  arrays. The hot sim kernel consumes them with no Python objects, so it stays njit
  and bit-identical against a pure-Python reference.
* VECTORIZABLE: all builders are pure numpy over a tz-aware DatetimeIndex.

What is MODELED vs DATA-DRIVEN
------------------------------
MODELED (calibrated from public typical retail values, documented below):
  - FX per-pair base half-spread (pips) and the UTC time-of-day spread multiplier.
  - FX overnight swap (negative-carry pips/night, triple on Wednesday).
  - Equity ETF base half-spread (bp) and the intraday open/close-wide schedule.
  - Equity commission ($/share with a min-ticket -> converted to a bp-of-notional
    rate per fill) and short-borrow (annualized %, accrued per calendar day held).
DATA-DRIVEN:
  - Bar OHLC, the real weekend gap (Fri close -> Sun open) and equity overnight
    gap come straight from the data; they are NOT modeled, just kept.
  - The FX cache on disk carries open/high/low/close/count only (the HistData
    fetch discarded the per-tick spread), so the FX half-spread here is the
    CALIBRATED schedule, not a measured per-bar spread. If a future fetch carries
    real mean(ask-bid)/bar we can swap `fx_halfspread_pips_schedule` for it.

Calibration references (typical RETAIL conditions, London-NY overlap baseline)
------------------------------------------------------------------------------
FX base half-spread (pips, i.e. HALF of the typical quoted spread at the tight
London-NY overlap; full spread ~= 2x these):
    EURUSD 0.10   GBPUSD 0.25   USDJPY 0.15   USDCHF 0.30
    USDCAD 0.30   AUDUSD 0.20   NZDUSD 0.35   EURGBP 0.30
(EURUSD is the tightest major ~0.1-0.2 pip; cable / commodity / cross pairs wider.)
FX time-of-day multiplier on the half-spread (UTC hour):
    London-NY overlap 12:00-16:00 -> 1.0 (tightest)
    London morning    07:00-12:00 -> 1.3
    NY afternoon       16:00-21:00 -> 1.6
    Rollover           21:00-23:00 -> 3.0 (thin book at 22:00 UTC value-date roll)
    Asian / thin       23:00-07:00 -> 2.2
FX swap: representative negative carry ~0.30 pip/night charged when a position is
held through the 21:00-22:00 UTC daily rollover; TRIPLE on Wednesday (Wed roll
books Sat+Sun+Mon value dates). Modeled as a conservative debit on BOTH long and
short (no per-pair funding-rate feed on disk).

Equity ETF base half-spread (bp of price, midday baseline):
    SPY 0.5  QQQ 0.5  IWM 1.0  XLK 1.5  XLF 1.5  XLE 2.0  XLV 2.0
    UVXY 12.0  VXX 8.0   (vol ETPs are wide)
Equity intraday spread multiplier (US/Eastern minutes from RTH open):
    open      09:30-10:00 -> 2.5x
    mid       10:00-15:45 -> 1.0x
    close     15:45-16:00 -> 1.8x
Equity commission: 0.0035 $/share, min ticket $0.35, charged per FILL, converted
to a per-fill rate of notional = max(min_ticket, per_share*shares)/notional. With
a fixed 1-unit (1-share-equivalent) position the per-fill cost is simply the
min-ticket vs per-share comparison on a 1-share notional == price. We model the
commission as a bp-of-notional floor-aware rate per fill (see equity_commission_rate_bp).
Equity short borrow (annualized): SPY/QQQ/IWM/sector SPDRs ~0.5%/yr; vol ETPs
UVXY/VXX ~5%/yr. Accrued per CALENDAR day a short is held. Cash longs: no margin
interest (we model long-only financing as zero).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ===========================================================================
# Pip-value handling (FX)
# ===========================================================================
def fx_pip_size(pair: str) -> float:
    """Price increment of ONE pip for `pair`. JPY-quoted pairs pip = 0.01,
    all other majors pip = 0.0001. (Pair is e.g. 'EURUSD', 'USDJPY'.)"""
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


# ===========================================================================
# FOREX session / weekend calendar
# ===========================================================================
# FX trades ~Sun 22:00 UTC -> Fri 22:00 UTC. We treat the market as CLOSED from
# Fri 22:00 UTC through Sun 22:00 UTC (no ability to exit), so a position must be
# flattened at the last open bar before the Friday close and re-entered Sunday.
FX_OPEN_WEEKDAY_SUN = 6        # python weekday(): Mon=0..Sun=6
FX_OPEN_HOUR_SUN = 22          # Sunday 22:00 UTC open
FX_CLOSE_WEEKDAY_FRI = 4       # Friday
FX_CLOSE_HOUR_FRI = 22         # Friday 22:00 UTC close


def fx_is_open(ts_utc) -> np.ndarray:
    """Boolean (array), is the FX market open at the given UTC timestamp(s)?

    Open from Sun 22:00 UTC to Fri 22:00 UTC. CAUSAL: depends only on the
    timestamp. Accepts a scalar Timestamp or a DatetimeIndex (UTC)."""
    idx = pd.DatetimeIndex([ts_utc]) if not isinstance(ts_utc, (pd.DatetimeIndex, np.ndarray)) else pd.DatetimeIndex(ts_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    wd = idx.weekday.to_numpy()
    hr = idx.hour.to_numpy()
    closed_sat = (wd == 5)
    closed_sun_early = (wd == 6) & (hr < FX_OPEN_HOUR_SUN)
    closed_fri_late = (wd == 4) & (hr >= FX_CLOSE_HOUR_FRI)
    return ~(closed_sat | closed_sun_early | closed_fri_late)


def fx_force_flat_flags(index_utc: pd.DatetimeIndex) -> np.ndarray:
    """int8[n], 1 at every bar that is the LAST OPEN BAR before a weekend close
    (force-flat here), else 0. A bar is a force-flat bar if it is open and the
    NEXT bar in the series is closed (or there is no next bar).

    Because the FX bar cache only contains open-session bars, the natural Fri->Sun
    discontinuity is detected as: open(t) and (t is last, or weekday/hour of t+1 is
    in the closed window). We flag the Friday bar so the sim closes there; the
    Fri-close->Sun-open price gap that remains across the boundary is REAL and kept.
    """
    idx = index_utc.tz_convert("UTC") if index_utc.tz is not None else index_utc.tz_localize("UTC")
    openf = fx_is_open(idx)
    n = len(idx)
    flat = np.zeros(n, np.int8)
    if n == 0:
        return flat
    # last open bar before a gap: open[t] & (t==last or not open[t+1])
    nxt_open = np.empty(n, bool)
    nxt_open[:-1] = openf[1:]
    nxt_open[-1] = False
    flat[(openf) & (~nxt_open)] = 1
    return flat


def fx_rollover_flags(index_utc: pd.DatetimeIndex) -> np.ndarray:
    """int8[n], 1 at the FIRST bar at/after the 21:00-22:00 UTC daily rollover on
    each calendar day (the bar at which an overnight-held position is charged a
    swap), with the multiplier encoded separately. We mark the first bar whose UTC
    hour is >= 21 on a given UTC date as that day's rollover bar."""
    idx = index_utc.tz_convert("UTC") if index_utc.tz is not None else index_utc.tz_localize("UTC")
    n = len(idx)
    flag = np.zeros(n, np.int8)
    if n == 0:
        return flag
    hr = idx.hour.to_numpy()
    date = idx.normalize().asi8                       # per-UTC-date key
    is_roll_window = hr >= 21
    seen = {}
    for i in range(n):
        if is_roll_window[i]:
            d = date[i]
            if d not in seen:
                seen[d] = True
                flag[i] = 1
    return flag


def fx_swap_multiplier(index_utc: pd.DatetimeIndex) -> np.ndarray:
    """float64[n], swap multiplier at each bar: 3.0 if the bar's UTC date is a
    Wednesday (triple swap for the weekend value date), else 1.0. Used together
    with `fx_rollover_flags` to charge swap only on the rollover bar."""
    idx = index_utc.tz_convert("UTC") if index_utc.tz is not None else index_utc.tz_localize("UTC")
    wd = idx.weekday.to_numpy()
    return np.where(wd == 2, 3.0, 1.0).astype(np.float64)   # Wed = weekday 2


# ===========================================================================
# FOREX time-of-day half-spread schedule
# ===========================================================================
FX_BASE_HALFSPREAD_PIPS = {
    "EURUSD": 0.10, "GBPUSD": 0.25, "USDJPY": 0.15, "USDCHF": 0.30,
    "USDCAD": 0.30, "AUDUSD": 0.20, "NZDUSD": 0.35, "EURGBP": 0.30,
}
FX_DEFAULT_HALFSPREAD_PIPS = 0.40        # unknown / exotic pair fallback (wide)
FX_SWAP_PIPS_PER_NIGHT = 0.30            # conservative negative carry per night


def _fx_tod_multiplier(hours: np.ndarray) -> np.ndarray:
    """UTC-hour -> half-spread multiplier (see module docstring)."""
    m = np.empty(hours.shape[0], np.float64)
    # default thin/Asian
    m[:] = 2.2
    m[(hours >= 7) & (hours < 12)] = 1.3        # London morning
    m[(hours >= 12) & (hours < 16)] = 1.0       # London-NY overlap (tightest)
    m[(hours >= 16) & (hours < 21)] = 1.6       # NY afternoon
    m[(hours >= 21) & (hours < 23)] = 3.0       # rollover (thin book)
    return m


def fx_halfspread_pips_schedule(index_utc: pd.DatetimeIndex, pair: str) -> np.ndarray:
    """float64[n], modeled HALF-spread in PIPS at each bar = base * tod_mult."""
    idx = index_utc.tz_convert("UTC") if index_utc.tz is not None else index_utc.tz_localize("UTC")
    base = FX_BASE_HALFSPREAD_PIPS.get(pair.upper(), FX_DEFAULT_HALFSPREAD_PIPS)
    hr = idx.hour.to_numpy()
    return base * _fx_tod_multiplier(hr)


def fx_per_fill_cost_price(index_utc: pd.DatetimeIndex, pair: str) -> np.ndarray:
    """float64[n], per-FILL FX cost in PRICE units at each bar = half-spread(pips)
    * pip_size. Charged on each entry/exit fill. (Crossing half the spread per
    fill is the realistic retail model.)"""
    return fx_halfspread_pips_schedule(index_utc, pair) * fx_pip_size(pair)


def fx_swap_cost_price_per_night(pair: str) -> float:
    """Per-NIGHT swap debit in PRICE units (before the Wed triple multiplier)."""
    return FX_SWAP_PIPS_PER_NIGHT * fx_pip_size(pair)


# ===========================================================================
# US-EQUITY session / time-of-day schedule
# ===========================================================================
RTH_OPEN_MIN = 9 * 60 + 30        # 09:30 ET
RTH_CLOSE_MIN = 16 * 60           # 16:00 ET
EQ_OPEN_WIDE_END = 10 * 60        # 10:00 ET
EQ_CLOSE_WIDE_START = 15 * 60 + 45  # 15:45 ET

EQ_BASE_HALFSPREAD_BP = {
    "SPY": 0.5, "QQQ": 0.5, "IWM": 1.0,
    "XLK": 1.5, "XLF": 1.5, "XLE": 2.0, "XLV": 2.0,
    "UVXY": 12.0, "VXX": 8.0,
}
EQ_DEFAULT_HALFSPREAD_BP = 3.0

# commission
EQ_COMMISSION_PER_SHARE = 0.0035   # $/share
EQ_COMMISSION_MIN_TICKET = 0.35    # $ per fill floor
EQ_ORDER_NOTIONAL = 25_000.0       # $ assumed order notional (commission is a fraction
                                   # of the ORDER, not of a single share, see below)

# short borrow (annualized fraction)
EQ_BORROW_ANNUAL = {
    "SPY": 0.005, "QQQ": 0.005, "IWM": 0.007,
    "XLK": 0.005, "XLF": 0.005, "XLE": 0.005, "XLV": 0.005,
    "UVXY": 0.05, "VXX": 0.05,
}
EQ_DEFAULT_BORROW_ANNUAL = 0.01


def _eq_tod_multiplier(min_of_day_et: np.ndarray) -> np.ndarray:
    """ET minute-of-day -> half-spread multiplier (open/close wide, midday tight)."""
    m = np.ones(min_of_day_et.shape[0], np.float64)
    m[(min_of_day_et >= RTH_OPEN_MIN) & (min_of_day_et < EQ_OPEN_WIDE_END)] = 2.5
    m[(min_of_day_et >= EQ_CLOSE_WIDE_START) & (min_of_day_et < RTH_CLOSE_MIN)] = 1.8
    return m


def equity_halfspread_bp_schedule(index_et: pd.DatetimeIndex, ticker: str) -> np.ndarray:
    """float64[n], modeled half-spread in BP at each bar = base_bp * tod_mult.
    Index must be America/New_York tz-aware (RTH bars)."""
    if index_et.tz is None:
        idx = index_et.tz_localize("America/New_York")
    else:
        idx = index_et.tz_convert("America/New_York")
    base = EQ_BASE_HALFSPREAD_BP.get(ticker.upper(), EQ_DEFAULT_HALFSPREAD_BP)
    mod = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
    return base * _eq_tod_multiplier(mod)


def equity_commission_rate_bp(price: np.ndarray, ticker: str = "") -> np.ndarray:
    """float64[n], per-FILL commission as a RATE (fraction of notional) for a
    realistic ORDER of EQ_ORDER_NOTIONAL dollars: shares = notional/price,
    cost$ = max(min_ticket, per_share*shares), rate = cost$/notional
          = max(per_share/price, min_ticket/notional).
    (The earlier version assumed a 1-SHARE position, which spread the $0.35
    min-ticket over a single share's price and inflated equity commission ~10-70x
   , e.g. 7 bp on a $500 SPY share, ~35 bp on a $10 ETP. Corrected here: for a
    $25k order, commission is sub-bp and spread dominates, as in reality.)
    Returns a FRACTION (not bp).
    """
    p = np.asarray(price, np.float64)
    per_share_rate = np.where(p > 0, EQ_COMMISSION_PER_SHARE / p, 0.0)
    floor_rate = EQ_COMMISSION_MIN_TICKET / EQ_ORDER_NOTIONAL
    return np.maximum(per_share_rate, floor_rate)


def equity_per_fill_cost_rate(index_et: pd.DatetimeIndex, price: np.ndarray,
                              ticker: str) -> np.ndarray:
    """float64[n], total per-FILL equity cost as a FRACTION of fill price =
    half-spread(bp)*1e-4 + commission_rate. Charged on each entry/exit fill."""
    hs = equity_halfspread_bp_schedule(index_et, ticker) * 1e-4
    comm = equity_commission_rate_bp(price, ticker)
    return hs + comm


def equity_borrow_rate_daily(ticker: str) -> float:
    """Per-CALENDAR-DAY short-borrow fraction (annual/365)."""
    ann = EQ_BORROW_ANNUAL.get(ticker.upper(), EQ_DEFAULT_BORROW_ANNUAL)
    return ann / 365.0


# ===========================================================================
# Convenience: build the full per-bar friction arrays the sim kernel needs.
# ===========================================================================
def build_fx_friction_arrays(index_utc: pd.DatetimeIndex, pair: str):
    """Return the numpy arrays the FX sim kernel consumes (all length n):
        per_fill_cost_px : f64  per-fill cost in PRICE units (half-spread)
        force_flat       : i8   1 at last open bar before a weekend gap
        rollover_flag    : i8   1 at the daily 21:00 UTC rollover bar
        swap_mult        : f64  3.0 on Wednesday else 1.0
        swap_px_night    : f64  scalar swap cost/night in price units (broadcast)
    """
    per_fill = np.ascontiguousarray(fx_per_fill_cost_price(index_utc, pair), np.float64)
    force_flat = np.ascontiguousarray(fx_force_flat_flags(index_utc), np.int8)
    rollover = np.ascontiguousarray(fx_rollover_flags(index_utc), np.int8)
    swap_mult = np.ascontiguousarray(fx_swap_multiplier(index_utc), np.float64)
    swap_night = float(fx_swap_cost_price_per_night(pair))
    return per_fill, force_flat, rollover, swap_mult, swap_night


def per_side_cost_fraction(market: str, symbol: str, index: pd.DatetimeIndex,
                           close: np.ndarray, crypto_fallback: float = 7.0e-4
                           ) -> np.ndarray:
    """float64[n], REALISTIC per-SIDE cost as a FRACTION of price at each bar, for
    the simple book/turnover-PnL retrofits in projects 05/07/08/09/11/12/13.

    This is the time-of-day half-spread (+commission for equity) expressed as a
    fraction, indexed by bar. A round trip charges this twice (entry + exit), or a
    book charges `per_side[t] * |Δposition_t|` on turnover. CAUSAL (time-of-day is
    ex-ante).

    CRYPTO IS LEFT AS-IS: the caller passes its own house crypto cost via
    `crypto_fallback` and this returns a flat array of that value (no recosting).

    market in {"equity"/"equities", "forex", "crypto"}; index tz-aware
    (ET for equity, UTC for forex); close f64[n] used to convert FX price-units to
    a fraction and to apply the equity commission floor.
    """
    n = len(close)
    c = np.asarray(close, np.float64)
    if market in ("equity", "equities"):
        return equity_per_fill_cost_rate(index, c, symbol)
    if market == "forex":
        abs_px = fx_per_fill_cost_price(index, symbol)
        return np.where(c > 0, abs_px / c, 0.0)
    # crypto / unknown: caller's own flat per-side cost (unchanged house default)
    return np.full(n, float(crypto_fallback), np.float64)


def build_equity_friction_arrays(index_et: pd.DatetimeIndex, price: np.ndarray,
                                 ticker: str):
    """Return the numpy arrays the equity sim kernel consumes (all length n):
        per_fill_cost_rate : f64  per-fill cost as FRACTION of fill price
        borrow_daily       : f64  scalar daily borrow fraction (broadcast)
    Force-flat at RTH close is handled by the existing session day-id logic in
    the kernel (last bar of each ET day), so no extra flag is needed.
    """
    per_fill = np.ascontiguousarray(equity_per_fill_cost_rate(index_et, price, ticker), np.float64)
    borrow = float(equity_borrow_rate_daily(ticker))
    return per_fill, borrow
