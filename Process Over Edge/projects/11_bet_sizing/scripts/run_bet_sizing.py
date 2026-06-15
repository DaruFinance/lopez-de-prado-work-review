#!/usr/bin/env python3
"""
run_bet_sizing.py, Bet Sizing from predicted probabilities (López de Prado, AFML Ch.10).

Idempotent driver. Reproduces LdP's bet-sizing recipe and tests, across
Crypto + US Equities + Forex, whether sizing a bet by the META-MODEL PROBABILITY
(and averaging concurrent bets / discretizing the size to curb overtrading)
beats a fixed-size book, judged by the program HEADLINE METRIC, the Deflated
Sharpe Ratio (not raw PF/Sharpe), net of realistic costs, with PBO + effective-N.

LdP Ch.10 recipe implemented here
---------------------------------
A SECONDARY (meta) classifier outputs p = P(the primary's bet is profitable).
The primary fixes the SIDE s in {+1,-1}; the meta only sizes / vetoes.

  1. From p, form a test statistic against p0 = 0.5 (two-outcome case):
        z = (p - 0.5) / sqrt(p*(1-p))
     and map to a signed bet size with the standard-normal CDF:
        m = 2*Phi(z) - 1            (m in (-1, 1)),  signed = s * m
     m=0 at p=0.5 (no edge), m->1 as p->1. This is LdP eq. (10.1)-(10.2).

  2. AVERAGE CONCURRENT / OVERLAPPING ACTIVE BETS.  A bet opened at t0 stays
     active until its triple-barrier exit. At any bar, several bets overlap.
     LdP averages the signed sizes of all bets active at that bar to get the
     book's net position, which both nets opposing bets and damps turnover.
     This per-bar averaging across overlapping holding windows is the HOT LOOP
     -> Numba kernel `_avg_active_kernel`, verified bit-identical vs a pure
     -Python reference `avg_active_reference`.

  3. DISCRETIZE the size to a grid of step `d` (LdP `discretizeSignal`):
        m_disc = round(m / d) * d, clipped to [-1, 1]
     to avoid over-trading on tiny size changes.

  4. Convert the book's per-bar net position into a costed P&L stream: P&L at
     bar t = position_t * bar_return_{t->t+1} - cost * |position_t - position_{t-1}|
     (cost charged on the *change* in book position = turnover), per market.

Sizing schemes compared (same events, same OOF probabilities, same costs):
  A. fixed        : signed unit size on every acted event (size = s*1)
  B. prob         : signed m = s*(2*Phi(z)-1), continuous
  C. prob_disc    : signed discretized m on a step-d grid
All three are run through the SAME active-bet averaging + costed-book engine, so
the only difference is the per-event target size. The grid of (meta-threshold,
discretization step, max_hold...) are the IS-tunable knobs = the DSR trials.

Everything is on REAL 1-minute data, costed, causal-only, purged-CV for the OOF
meta probabilities (reuses projects/03_meta_labeling/scripts/tbm.py).

Run:  python3 scripts/run_bet_sizing.py                 (full multi-market run)
      python3 scripts/run_bet_sizing.py --smoke          (one instrument per market, tiny subset)
      python3 scripts/run_bet_sizing.py --profile        (cProfile a single instrument)
      python3 scripts/run_bet_sizing.py --verify-kernel  (numba vs python bit-identical check)
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = ROOT = _d
sys.path.insert(0, REPO_ROOT)
import config as cfg
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
# reuse the meta-labeling library (triple-barrier + primary + features)
META_SCRIPTS = os.path.join(ROOT, "projects", "03_meta_labeling", "scripts")
sys.path.insert(0, META_SCRIPTS)
sys.path.insert(0, HERE)

from lib import bars as B
from lib import overfit as O
from lib import realism as RZ
import tbm

from scipy import stats as ss
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CRYPTO_DIR = cfg.CRYPTO_1M
FX_DIR = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

# per-turnover cost in bp of notional (charged on the CHANGE in book position).
COST_BP = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}
N_TARGET_BARS = 20000          # dollar/tick bars per instrument (LdP-style)
# Realistic per-trade EQUITY notional used to scale the broker min-ticket
# commission. lib.realism.equity_commission_rate_bp models a literal 1-SHARE
# position, so the $0.35 min-ticket floor becomes 8-70 bp/fill on a $50-$600 ETF
# share, an artifact of a degenerate 1-share book, NOT a real friction. A retail
# trader sizes a position in dollars, so we apply the SAME min-ticket schedule on
# a realistic $EQ_NOTIONAL position (shares = notional/price); the spread schedule
# from lib.realism is used unchanged. No clamping, costs still fall where they
# fall, we just size the commission on a non-degenerate book.
EQ_NOTIONAL = 10_000.0

# IS-tunable knob grid (the "trials"): structural shape is fixed (EMA-xover
# primary + triple-barrier + bagged-tree meta + Ch.10 sizing). Numeric knobs
# (barriers, holding, meta-threshold, discretization step) are tuned IN-SAMPLE.
FAST_SLOW = [(20, 60), (30, 90)]
PT_SL = [(1.0, 1.0), (1.5, 1.0)]              # (profit-take, stop) vol multiples
MAX_HOLD = [50, 100]
META_THRESH = [0.50, 0.52]                    # act when P(profit) >= this
DISC_STEP = [0.10, 0.20]                      # discretization grid step d
VOL_SPAN = 50
PARAM_GRID = [(fs, ps, mh, mt, ds)
              for fs in FAST_SLOW for ps in PT_SL for mh in MAX_HOLD
              for mt in META_THRESH for ds in DISC_STEP]
N_TRIALS = len(PARAM_GRID)                    # 32 trials -> fed to DSR / MinBTL

N_SPLITS = 6
EMBARGO = 0.01


# --------------------------------------------------------------------------- #
# Data -> bars
# --------------------------------------------------------------------------- #
def make_bars(market: str, path: str) -> pd.DataFrame:
    if market == "crypto":
        base = tbm.load_base_crypto(path)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "equities":
        base = B.load_base_equity_etf(path, rth=True)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "forex":
        base = tbm.load_base_fx(path)
        thr = base["count"].sum() / N_TARGET_BARS
        return B.threshold_bars(base, "count", thr)
    raise ValueError(market)


def equity_per_side_cost_realistic(index_et, price, ticker, notional=EQ_NOTIONAL):
    """Per-SIDE equity cost as a FRACTION of price, on a realistic $notional book.

    half-spread(bp) schedule from lib.realism (unchanged) + min-ticket commission
    applied to shares = notional/price (not a degenerate 1-share book). CAUSAL
    (time-of-day schedule is ex-ante); no clamping. Returns fraction-of-price[n]."""
    p = np.asarray(price, np.float64)
    hs = RZ.equity_halfspread_bp_schedule(index_et, ticker) * 1e-4
    shares = np.where(p > 0, notional / p, 0.0)
    per_fill_dollar = np.maximum(RZ.EQ_COMMISSION_MIN_TICKET,
                                 RZ.EQ_COMMISSION_PER_SHARE * shares)
    comm = np.where(notional > 0, per_fill_dollar / notional, 0.0)
    return hs + comm


def bars_per_year(bars: pd.DataFrame) -> float:
    """Annualisation factor = bars per calendar year, from the elapsed span."""
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    return len(bars) / years


# --------------------------------------------------------------------------- #
# LdP Ch.10 sizing math (vectorised, causal, p is an OOF probability)
# --------------------------------------------------------------------------- #
def prob_to_size(p: np.ndarray, p0: float = 0.5) -> np.ndarray:
    """LdP eq.(10.1)-(10.2): z=(p-p0)/sqrt(p(1-p)); m=2*Phi(z)-1, in (-1,1).

    Magnitude of the (unsigned) bet from the meta probability. Caller multiplies
    by the side s to get the signed size. p clipped off {0,1} for a finite z."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = (p - p0) / np.sqrt(p * (1.0 - p))
    return 2.0 * ss.norm.cdf(z) - 1.0


def discretize_size(m: np.ndarray, step: float) -> np.ndarray:
    """LdP discretizeSignal: snap to a grid of `step`, clip to [-1,1]."""
    if step <= 0:
        return np.clip(m, -1.0, 1.0)
    d = np.round(m / step) * step
    return np.clip(d, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# HOT LOOP, average concurrent/overlapping active bets per bar (Numba kernel)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _avg_active_kernel(ev_idx, hold, signed_size, n_bars):
    """Net book position per bar = MEAN signed size of all bets active at the bar.

    Bet k is active on bars [ev_idx[k], ev_idx[k]+hold[k]) (entry bar inclusive,
    exit bar exclusive, the position is held over those bars and earns the
    bar-forward return). At each bar we average the signed sizes of every bet
    active there (LdP `avgActiveSignals`); a bar with no active bet is flat (0).

    O(n_ev + n_bars) two-pointer over a difference array of (sum, count), so it
    scales to the full cross-section without an n_ev x n_bars materialisation.
    Returns the per-bar net position array of length n_bars."""
    dsum = np.zeros(n_bars + 1, np.float64)     # difference array for size sum
    dcnt = np.zeros(n_bars + 1, np.int64)       # difference array for active count
    for k in range(ev_idx.shape[0]):
        i0 = ev_idx[k]
        h = hold[k]
        if h < 1:
            h = 1
        j1 = i0 + h
        if j1 > n_bars:
            j1 = n_bars
        if i0 < 0:
            i0 = 0
        if i0 >= n_bars:
            continue
        dsum[i0] += signed_size[k]
        dsum[j1] -= signed_size[k]
        dcnt[i0] += 1
        dcnt[j1] -= 1
    pos = np.zeros(n_bars, np.float64)
    run_sum = 0.0
    run_cnt = 0
    for t in range(n_bars):
        run_sum += dsum[t]
        run_cnt += dcnt[t]
        if run_cnt > 0:
            pos[t] = run_sum / run_cnt
        else:
            pos[t] = 0.0
    return pos


def avg_active_reference(ev_idx, hold, signed_size, n_bars):
    """Pure-Python/NumPy reference for the active-bet averaging (no Numba).

    Independent implementation via NumPy difference arrays + cumsum (the kernel
    instead does an explicit scalar running-sum loop in compiled code). Both
    express LdP's `avgActiveSignals`: at each bar, the arithmetic mean of the
    signed sizes of every bet whose [entry, exit) window covers it. The two code
    paths share NO lines, so a bit-identical match is a genuine cross-check.

    Float summation order is fixed to MATCH the kernel (interval-endpoint
    contributions accumulated left-to-right) so the comparison is exact, not
    merely within tolerance: the difference-array endpoints are placed in the
    same order and the prefix sum advances bar-by-bar exactly as the kernel."""
    ev_idx = np.asarray(ev_idx, np.int64)
    hold = np.asarray(hold, np.int64)
    signed_size = np.asarray(signed_size, float)
    dsum = np.zeros(n_bars + 1, float)
    dcnt = np.zeros(n_bars + 1, np.int64)
    for k in range(len(ev_idx)):
        i0 = int(ev_idx[k])
        if i0 < 0:
            i0 = 0
        if i0 >= n_bars:
            continue
        h = max(1, int(hold[k]))
        j1 = min(n_bars, i0 + h)
        dsum[i0] += signed_size[k]
        dsum[j1] -= signed_size[k]
        dcnt[i0] += 1
        dcnt[j1] -= 1
    run_sum = np.cumsum(dsum[:n_bars])
    run_cnt = np.cumsum(dcnt[:n_bars])
    pos = np.zeros(n_bars, float)
    active = run_cnt > 0
    pos[active] = run_sum[active] / run_cnt[active]
    return pos


def costed_book_pnl(pos: np.ndarray, bar_logret: np.ndarray, cost) -> np.ndarray:
    """Per-bar net P&L of a book whose net position is `pos`.

    pnl_t = pos_t * r_{t->t+1} - cost_t * |pos_t - pos_{t-1}|
    (cost charged on the change in book position = realised turnover). The final
    bar has no forward return, so it earns 0 and only pays its closing turnover.
    Causal: pos_t is set at the close of bar t from info up to t; it earns the
    return from t to t+1. `cost` may be a scalar (crypto flat) OR a per-bar array
    (REALISTIC time-of-day half-spread + commission for equity/forex)."""
    n = len(pos)
    cost_arr = np.broadcast_to(np.asarray(cost, float), (n))
    pnl = np.zeros(n, float)
    prev = 0.0
    for t in range(n):
        fwd = bar_logret[t] if t + 1 < n else 0.0
        turn = abs(pos[t] - prev)
        pnl[t] = pos[t] * fwd - cost_arr[t] * turn
        prev = pos[t]
    return pnl


def score_series(r: np.ndarray, bpy: float) -> dict:
    r = r[np.isfinite(r)]
    nz = r[r != 0.0]
    sr = O.sharpe(r)
    sr_ann = sr * np.sqrt(bpy)
    sk = float(ss.skew(r)) if len(r) > 2 else 0.0
    ku = float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0
    gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else np.nan
    return dict(sr_per_bar=sr, sr_ann=sr_ann, pf=pf, skew=sk, kurt=ku,
                mean=float(r.mean()), n_obs=len(r))


# --------------------------------------------------------------------------- #
# Core: run one instrument
# --------------------------------------------------------------------------- #
def run_instrument(market: str, name: str, path: str, smoke=False):
    bars = make_bars(market, path)
    if smoke:                                   # tiny subset for 1-core smoke
        bars = bars.iloc[:4000]
    if len(bars) < 3000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    # REALISTIC per-bar per-side cost (time-of-day half-spread + commission) for
    # EQUITY/FOREX; crypto keeps the flat house default. Causal (ex-ante schedule).
    if market in ("equity", "equities"):
        # realistic-notional equity cost (avoids the 1-share min-ticket artifact)
        cost = equity_per_side_cost_realistic(bars.index, close, name)
    else:
        cost = RZ.per_side_cost_fraction(market, name, bars.index, close,
                                         crypto_fallback=COST_BP[market] / 1e4)
    # bar-forward log-returns r_{t->t+1}; last bar has no forward return.
    bar_logret = np.zeros(n, float)
    lp = np.log(close)
    bar_logret[:-1] = lp[1:] - lp[:-1]

    grid = PARAM_GRID[:2] if smoke else PARAM_GRID
    trials = []                                 # per-trial dict of the 3 schemes

    for (fast, slow), (pt, sl), mh, mt, ds in grid:
        vol = tbm.ewma_vol(close, VOL_SPAN)
        side_full = tbm.primary_ma_crossover(close, fast, slow)
        ev = tbm.crossover_events(side_full)
        warm = slow + 12
        ev = ev[(ev > warm) & (ev < n - 1)]
        if len(ev) < (30 if smoke else 200):
            continue
        side = side_full[ev]
        tb = tbm.triple_barrier(close, high, low, openp, ev, side, vol, pt, sl, mh)
        hold = tb["hold"].to_numpy()
        ret_gross = tb["ret_gross"].to_numpy()
        # realistic round-trip per-trade cost = entry-bar + exit-bar per-side cost
        ex_bar = np.minimum(ev + np.maximum(1, hold.astype(np.int64)), n - 1)
        rt_cost = cost[ev] + cost[ex_bar]
        pnl_trade = ret_gross - rt_cost         # net per-trade for the meta label
        y = (pnl_trade > 0).astype(int)
        if y.sum() < (5 if smoke else 20) or (1 - y).sum() < (5 if smoke else 20):
            continue

        # --- OOF meta probabilities via purged k-fold (leakage-free) ---
        feat = tbm.build_features(bars, side_full, vol, fast, slow)
        X = np.nan_to_num(feat.iloc[ev].to_numpy(np.float64), nan=0.0,
                          posinf=0.0, neginf=0.0)
        p_oof = np.full(len(ev), np.nan)
        for tr, te in O.purged_kfold_splits(len(ev), N_SPLITS, EMBARGO, label_span=3):
            if len(tr) < 50 or y[tr].sum() < 5 or (1 - y[tr]).sum() < 5:
                p_oof[te] = y[tr].mean() if len(tr) else 0.5
                continue
            clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
                n_estimators=(10 if smoke else 40), max_samples=0.8,
                max_features=0.8, bootstrap=True, n_jobs=1, random_state=0)
            clf.fit(X[tr], y[tr])
            if len(clf.classes_) == 1:
                p_oof[te] = float(clf.classes_[0])
            else:
                pi = list(clf.classes_).index(1)
                p_oof[te] = clf.predict_proba(X[te])[: pi]
        p_oof = np.nan_to_num(p_oof, nan=float(y.mean()))

        # --- act gate (meta veto) and the 3 target sizes per event ---
        act = p_oof >= mt
        mag = prob_to_size(p_oof)                       # unsigned magnitude in (-1,1)
        size_fixed = np.where(act, 1.0, 0.0) * side             # A
        size_prob = np.where(act, mag, 0.0) * side             # B
        size_disc = np.where(act, discretize_size(mag, ds), 0.0) * side  # C

        # --- HOT LOOP: average concurrent bets -> per-bar book position ---
        evi = np.ascontiguousarray(ev.astype(np.int64))
        hol = np.ascontiguousarray(hold.astype(np.int64))
        pos_fixed = _avg_active_kernel(evi, hol, np.ascontiguousarray(size_fixed), n)
        pos_prob = _avg_active_kernel(evi, hol, np.ascontiguousarray(size_prob), n)
        pos_disc = _avg_active_kernel(evi, hol, np.ascontiguousarray(size_disc), n)

        # --- costed book P&L streams and scores ---
        r_fixed = costed_book_pnl(pos_fixed, bar_logret, cost)
        r_prob = costed_book_pnl(pos_prob, bar_logret, cost)
        r_disc = costed_book_pnl(pos_disc, bar_logret, cost)
        sf, sp, sd = (score_series(r_fixed, bpy), score_series(r_prob, bpy),
                      score_series(r_disc, bpy))
        # turnover = total |position change| (overtrading proxy)
        def turnover(pos):
            return float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
        trials.append(dict(
            fast=fast, slow=slow, pt=pt, sl=sl, mh=mh, mt=mt, ds=ds,
            n_ev=len(ev), n_act=int(act.sum()), frac_act=float(act.mean()),
            r_fixed=r_fixed, r_prob=r_prob, r_disc=r_disc,
            sf=sf, sp=sp, sd=sd,
            turn_fixed=turnover(pos_fixed), turn_prob=turnover(pos_prob),
            turn_disc=turnover(pos_disc),
            p_oof=p_oof, size_prob=size_prob, size_disc=size_disc,
            pos_prob=pos_prob, ev=ev, hold=hold))

    if not trials:
        return None

    sr_fixed = np.array([t["sf"]["sr_per_bar"] for t in trials])
    sr_prob = np.array([t["sp"]["sr_per_bar"] for t in trials])
    sr_disc = np.array([t["sd"]["sr_per_bar"] for t in trials])

    # winner per scheme = best per-bar Sharpe across the IS-tunable grid
    bf = int(np.argmax(sr_fixed)); bp = int(np.argmax(sr_prob)); bd = int(np.argmax(sr_disc))

    def dsr(score, sr_trials):
        return O.deflated_sharpe_ratio(score["sr_per_bar"], score["n_obs"],
                                       score["skew"], score["kurt"], sr_trials)
    d_fixed = dsr(trials[bf]["sf"], sr_fixed)
    d_prob = dsr(trials[bp]["sp"], sr_prob)
    d_disc = dsr(trials[bd]["sd"], sr_disc)

    # PBO + effective-N on the prob-sized trial corpus (the headline scheme)
    Mp = np.column_stack([t["r_prob"] for t in trials])
    try:
        ns = min(10, max(2, 2 * (len(trials) // 2)))
        pbo_prob = O.pbo_cscv(Mp, n_splits=ns)["pbo"] if Mp.shape[1] >= 2 else np.nan
    except Exception:
        pbo_prob = np.nan
    try:
        eff = O.effective_n_trials(Mp)["effective_n"] if Mp.shape[1] >= 4 else len(trials)
    except Exception:
        eff = len(trials)

    Tp = trials[bp]                              # representative trial = best prob
    return dict(
        market=market, name=name, n_bars=n, bpy=bpy, n_trials=len(trials),
        best_prob=dict(fast=Tp["fast"], slow=Tp["slow"], pt=Tp["pt"], sl=Tp["sl"],
                       mh=Tp["mh"], mt=Tp["mt"], ds=Tp["ds"]),
        n_ev=Tp["n_ev"], frac_act=Tp["frac_act"],
        fixed_sr_ann=trials[bf]["sf"]["sr_ann"], fixed_pf=trials[bf]["sf"]["pf"],
        prob_sr_ann=Tp["sp"]["sr_ann"], prob_pf=Tp["sp"]["pf"],
        disc_sr_ann=trials[bd]["sd"]["sr_ann"], disc_pf=trials[bd]["sd"]["pf"],
        fixed_dsr=d_fixed["dsr"], prob_dsr=d_prob["dsr"], disc_dsr=d_disc["dsr"],
        fixed_turn=trials[bf]["turn_fixed"], prob_turn=Tp["turn_prob"],
        disc_turn=trials[bd]["turn_disc"],
        pbo_prob=pbo_prob, eff_n=eff,
        _T=Tp, _bars=bar_logret)


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def universe(smoke=False):
    u = []
    for p in sorted(glob.glob(os.path.join(CRYPTO_DIR, "*_1m.parquet"))):
        u.append(("crypto", os.path.basename(p).replace("_1m.parquet", ""), p))
    for s in ETF_SYMS:
        p = os.path.join(ETF_DIR, f"{s}.csv.gz")
        if os.path.exists(p):
            u.append(("equities", s, p))
    for p in sorted(glob.glob(os.path.join(FX_DIR, "*_fx1m.parquet"))):
        u.append(("forex", os.path.basename(p).replace("_fx1m.parquet", ""), p))
    if smoke:
        return [next(x for x in u if x[1] == "BTCUSDT"),
                next(x for x in u if x[1] == "SPY"),
                next(x for x in u if x[1] == "EURUSD")]
    return u


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, reps: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style(); P = style.PALETTE
    markets = ["crypto", "equities", "forex"]
    cmap = {"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}

    # FIG 1: DSR by market for the three schemes
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.4))
    xs = np.arange(len(markets)); w = 0.26
    for j, (sch, col, lab) in enumerate([("fixed_dsr", "#999999", "fixed"),
                                         ("prob_dsr", P["accent"], "prob"),
                                         ("disc_dsr", P["dollar"], "prob_disc")]):
        vals = [df[df.market == m][sch].median() for m in markets]
        ax.bar(xs + (j - 1) * w, vals, w, label=lab, color=col)
    ax.axhline(0.95, color="k", ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(markets)
    ax.set_ylabel("median DSR (OOS)"); ax.set_title("Deflated Sharpe by sizing scheme")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_dsr_by_scheme.png")); plt.close(fig)

    # FIG 2: turnover reduction (prob/disc vs fixed) by market
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.4))
    for j, (sch, col, lab) in enumerate([("fixed_turn", "#999999", "fixed"),
                                         ("prob_turn", P["accent"], "prob"),
                                         ("disc_turn", P["dollar"], "prob_disc")]):
        vals = [df[df.market == m][sch].median() for m in markets]
        ax.bar(xs + (j - 1) * w, vals, w, label=lab, color=col)
    ax.set_xticks(xs); ax.set_xticklabels(markets)
    ax.set_ylabel("median book turnover"); ax.set_title("Turnover by sizing scheme")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_turnover_by_scheme.png")); plt.close(fig)

    # FIG 3: representative book equity curves (fixed/prob/disc)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, m in enumerate(markets):
        if m not in reps:
            continue
        T = reps[m]["_T"]
        ax[i].plot(np.cumsum(T["r_fixed"]), color="#999999", label="fixed")
        ax[i].plot(np.cumsum(T["r_prob"]), color=P["accent"], label="prob")
        ax[i].plot(np.cumsum(T["r_disc"]), color=P["dollar"], label="prob_disc")
        ax[i].set_title(f"{m}: {reps[m]['name']}")
        ax[i].set_xlabel("bar"); ax[i].set_ylabel("cum log-return (net)"); ax[i].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_equity_by_scheme.png")); plt.close(fig)

    # FIG 4: bet-size distribution prob vs discretized (representative crypto)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, m in enumerate(markets):
        if m not in reps:
            continue
        T = reps[m]["_T"]
        ax[i].hist(T["size_prob"][T["size_prob"] != 0], bins=30, alpha=0.6,
                   color=P["accent"], label="prob")
        ax[i].hist(T["size_disc"][T["size_disc"] != 0], bins=30, alpha=0.6,
                   color=P["dollar"], label="prob_disc")
        ax[i].set_title(f"{m}: {reps[m]['name']}")
        ax[i].set_xlabel("signed bet size"); ax[i].set_ylabel("events"); ax[i].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_betsize_dist.png")); plt.close(fig)
    print("  figures written to", os.path.abspath(FIG))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def make_tables(df: pd.DataFrame):
    cols = ["market", "name", "n_ev", "frac_act",
            "fixed_pf", "prob_pf", "disc_pf",
            "fixed_sr_ann", "prob_sr_ann", "disc_sr_ann",
            "fixed_dsr", "prob_dsr", "disc_dsr",
            "fixed_turn", "prob_turn", "disc_turn", "pbo_prob", "eff_n", "best_prob"]
    t = df[cols].copy()
    t["best_prob"] = t["best_prob"].apply(
        lambda d: f"f{d['fast']}/s{d['slow']} pt{d['pt']}/sl{d['sl']} h{d['mh']} mt{d['mt']} d{d['ds']}")
    t.round(4).to_csv(os.path.join(TAB, "per_instrument.csv"), index=False)

    g = df.groupby("market")
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_fixed_pf": g["fixed_pf"].median(),
        "med_prob_pf": g["prob_pf"].median(),
        "med_disc_pf": g["disc_pf"].median(),
        "med_fixed_dsr": g["fixed_dsr"].median(),
        "med_prob_dsr": g["prob_dsr"].median(),
        "med_disc_dsr": g["disc_dsr"].median(),
        "n_prob_dsr_gt95": g.apply(lambda x: int((x["prob_dsr"] > 0.95).sum())),
        "n_disc_dsr_gt95": g.apply(lambda x: int((x["disc_dsr"] > 0.95).sum())),
        "med_fixed_turn": g["fixed_turn"].median(),
        "med_prob_turn": g["prob_turn"].median(),
        "med_disc_turn": g["disc_turn"].median(),
        "med_pbo_prob": g["pbo_prob"].median(),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "by_market_summary.csv"))
    md = ["# Bet Sizing (LdP AFML Ch.10), results\n",
          f"_{N_TRIALS} IS-tunable trials per instrument; DSR is the headline metric._\n",
          "\n## By-market summary\n", summ.to_markdown(),
          "\n\n## Per-instrument (head)\n", t.round(4).head(42).to_markdown(index=False)]
    with open(os.path.join(TAB, "results.md"), "w") as f:
        f.write("\n".join(md))
    print("  tables written to", os.path.abspath(TAB))
    return summ


# --------------------------------------------------------------------------- #
# Kernel verification, numba vs pure-python bit-identical
# --------------------------------------------------------------------------- #
def verify_kernel(seed=0, n_cases=200):
    rng = np.random.default_rng(seed)
    max_abs = 0.0
    for _ in range(n_cases):
        n_bars = int(rng.integers(50, 800))
        n_ev = int(rng.integers(5, 200))
        ev = np.sort(rng.integers(0, n_bars, n_ev)).astype(np.int64)
        hold = rng.integers(1, 120, n_ev).astype(np.int64)
        sz = (rng.uniform(-1, 1, n_ev)).astype(np.float64)
        a = _avg_active_kernel(np.ascontiguousarray(ev), np.ascontiguousarray(hold),
                               np.ascontiguousarray(sz), n_bars)
        b = avg_active_reference(ev, hold, sz, n_bars)
        max_abs = max(max_abs, float(np.max(np.abs(a - b))))
    print(f"avg_active kernel vs reference: {n_cases} random cases, "
          f"max|Δ| = {max_abs:.3e}  ->  {'BIT-IDENTICAL' if max_abs == 0.0 else 'MISMATCH'}")
    return max_abs


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile_one():
    import cProfile, pstats, io
    market, name, path = "crypto", "BTCUSDT", os.path.join(CRYPTO_DIR, "BTCUSDT_1m.parquet")
    # warm the numba kernel (exclude JIT compile from the profile)
    _avg_active_kernel(np.array([0, 1], np.int64), np.array([2, 3], np.int64),
                       np.array([0.5, -0.5]), 8)
    pr = cProfile.Profile(); pr.enable()
    run_instrument(market, name, path)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verify-kernel", action="store_true", dest="verify")
    args = ap.parse_args()

    if args.verify:
        verify_kernel(); return
    if args.profile:
        profile_one(); return

    u = universe(smoke=args.smoke)
    print(f"running {len(u)} instruments across "
          f"{len(set(m for m, _, _ in u))} markets; "
          f"{2 if args.smoke else N_TRIALS} trials each"
          f"{' [SMOKE]' if args.smoke else ''}")
    rows, reps = [], {}
    t0 = time.perf_counter()
    for market, name, path in u:
        try:
            tt = time.perf_counter()
            r = run_instrument(market, name, path, smoke=args.smoke)
            if r is None:
                print(f"  [skip] {market:9s} {name:10s} (insufficient data/events)")
                continue
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            want = {"crypto": "BTCUSDT", "equities": "SPY", "forex": "EURUSD"}
            if market not in reps or name == want.get(market):
                reps[market] = r
            print(f"  [ok]   {market:9s} {name:10s} "
                  f"fix_DSR={r['fixed_dsr']:.3f} prob_DSR={r['prob_dsr']:.3f} "
                  f"disc_DSR={r['disc_dsr']:.3f} | fix_PF={r['fixed_pf']:.3f} "
                  f"prob_PF={r['prob_pf']:.3f} disc_PF={r['disc_pf']:.3f} "
                  f"({time.perf_counter()-tt:.1f}s)")
        except Exception as e:
            print(f"  [ERR]  {market:9s} {name:10s}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no instruments produced results"); return
    df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore"
            ).to_parquet(os.path.join(TAB, "raw_results.parquet"))
    summ = make_tables(df)
    make_figures(df, reps)
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== BY-MARKET SUMMARY ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()
