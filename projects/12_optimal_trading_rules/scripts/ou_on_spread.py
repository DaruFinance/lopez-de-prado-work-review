#!/usr/bin/env python3
"""
ou_on_spread.py: OU optimal trading rule on a COINTEGRATED RESIDUAL SPREAD,
i.e. on the process for which the rule was actually derived.

Why this exists
---------------
The headline study fit the OU optimal-rule apparatus to a univariate rolling
z-score of log price and found it behaved like a coin-flip. The writeup's own
closing note flags the reason and the fix: a univariate z-score is not a genuine
Ornstein-Uhlenbeck process, so the rule's mesh has little to optimise (the stop
pins to the grid edge). The OU optimal profit-take / stop-loss rule is DERIVED
for a mean-reverting OU process, so the fair test is on a series where OU is the
true data-generating process: the residual SPREAD of a cointegrated pair.

This script does exactly that, end to end, with no look-ahead:
  1. From real crypto perpetual hourly OHLCV, screen a small set of candidate
     pairs for cointegration on TRAIN log-prices only (Engle-Granger: OLS hedge
     ratio beta on TRAIN, ADF on the TRAIN residual). Keep pairs that pass.
  2. Build the residual spread s_t = logA - beta*logB with beta estimated on
     TRAIN only. Fit an OU/AR(1) to the TRAIN spread (reusing the study's
     otr.fit_ou), derive the optimal (profit-take, stop-loss) from a Monte-Carlo
     mesh on the FITTED process (reusing the study's enter-at-deviation mesh
     ou_mesh_dev), and trade the spread OUT-OF-SAMPLE.
  3. Controls, on the same OOS spread and identical costs: a fixed-band /
     Bollinger rule (enter at +/- k stationary-std, exit at the mean) and a
     buy-hold-the-spread benchmark. Report OOS Sharpe / PF and the Deflated
     Sharpe Ratio for the OU rule vs the controls, per pair and pooled.

Honest question
---------------
On a genuinely mean-reverting spread (OU's true regime), does the OU optimal rule
beat the controls and clear deflation? Or is the study's coin-flip verdict a
wrong-regime artifact (raw price) rather than a property of the method?

No look-ahead anywhere
----------------------
The hedge ratio beta and the OU parameters (E0, phi, sigma) are estimated on
TRAIN only; OOS trading uses those frozen values. Intrabar exits use full OHLC of
BOTH legs: the spread's intrabar high/low are bounded by the leg OHLC extremes
(spread_high <= logA_high - beta*logB_low for beta > 0, and the mirror for the
low). Realistic per-fill costs are charged on BOTH legs at entry and exit. The
spread is two-sided on perps (both legs shortable).

RAM
---
Only the pairs we actually trade are loaded, one pair (two ~60k-row hourly
frames) at a time, aligned on the shared timestamp index. Peak working set is a
handful of float64 arrays of length ~60k plus the Monte-Carlo random stream
(n_paths x horizon float64). Well under 0.5 GB. No full-panel load.

Reuses the study's machinery unmodified: otr.fit_ou, otr.ou_mesh_dev (via the
deepen module), lib.overfit (Sharpe, DSR), lib.realism conventions, lib.style.
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, resource
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import config as cfg
from lib import overfit as O
import otr
import deepen                      # for the enter-at-deviation OU mesh (ou_mesh_dev)

from scipy import stats as ss

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
PERP_DIR = cfg.CRYPTO_1H

# Candidate pairs to SCREEN for cointegration. Chosen as economically related,
# liquid perps with long shared history (no full-panel load: a short curated list).
CANDIDATE_PAIRS = [
    # economically-related siblings (same sector / fork / theme) -- the natural
    # places to look for a stable long-run equilibrium between two perps.
    ("AAVEUSDT", "UNIUSDT"),    # two DeFi blue chips
    ("COMPUSDT", "AAVEUSDT"),   # two DeFi lending protocols
    ("SNXUSDT", "AAVEUSDT"),    # two DeFi protocols
    ("1INCHUSDT", "UNIUSDT"),   # two DEX tokens
    ("THETAUSDT", "FILUSDT"),   # two decentralised-infrastructure tokens
    ("GRTUSDT", "LINKUSDT"),    # two data/oracle middleware tokens
    ("GALAUSDT", "SANDUSDT"),   # two gaming/metaverse tokens
    ("ETCUSDT", "ETHUSDT"),     # ETH and its fork
    ("AVAXUSDT", "SOLUSDT"),    # two newer L1s
    ("LTCUSDT", "BCHUSDT"),     # two BTC forks
]
N_KEEP = 5                       # keep at most this many cointegrated pairs

# per-side cost in bp of notional PER LEG, charged on entry and exit. A spread
# round trip touches BOTH legs twice, so 4 fills total. Crypto taker-ish.
COST_BP_PER_LEG = 7.0

IS_FRAC = 0.6                    # chronological train fraction (beta + OU fit)
ADF_P_MAX = 0.05                 # Engle-Granger ADF p-value to call a pair cointegrated
MIN_OOS_TRADES = 25
MIN_SHARED_BARS = 8000           # require a decent shared history

# OU optimal-rule mesh (in spread / stationary-std units). The spread is OU; pt/sl
# are in units of the stationary std of the spread.
PT_GRID = np.round(np.arange(0.25, 3.01, 0.25), 4)
SL_GRID = np.round(np.arange(0.25, 3.01, 0.25), 4)
N_PATHS = 20000
MC_HORIZON = 500
MC_SEED = 12

ENTRY_K = 2.0                    # enter when |spread z| >= ENTRY_K (fresh extreme)
MAX_HOLD = 200                   # vertical-barrier cap (bars)
Z_SPAN = 500                     # causal EWMA span for the spread z-score (bars)
BARS_PER_YEAR = 24.0 * 365.25    # hourly bars

# smoke overrides
SMOKE_PT = np.array([0.5, 1.0, 1.5, 2.0])
SMOKE_SL = np.array([0.5, 1.0, 1.5, 2.0])
SMOKE_PATHS = 400
SMOKE_HORIZON = 120


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_perp(symbol: str) -> pd.DataFrame:
    """Real hourly perp OHLCV. open_time may be a column or the index."""
    f = os.path.join(PERP_DIR, f"{symbol}_1h.parquet")
    df = pd.read_parquet(f, columns=["open_time", "open", "high", "low", "close", "volume"])
    ot = pd.to_datetime(df["open_time"], utc=True)
    df = df.assign(open_time=ot).set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["open"] > 0)]
    return df


def align_pair(a: pd.DataFrame, b: pd.DataFrame):
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx], idx


# --------------------------------------------------------------------------- #
# Cointegration: Engle-Granger (OLS hedge ratio + ADF on residual), TRAIN only
# --------------------------------------------------------------------------- #
def _adf_pvalue(x: np.ndarray) -> float:
    """ADF test p-value via statsmodels if available, else a fallback that
    regresses dx_t on x_{t-1} and returns the t-stat mapped through an
    approximate Dickey-Fuller surface. The statsmodels path is exact."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return 1.0
    try:
        from statsmodels.tsa.stattools import adfuller
        # AIC lag-selection over a long maxlag is the dominant cost on a ~35k-row
        # residual; a short fixed augmentation captures the short-memory structure
        # of a residual spread without the multi-second autolag search.
        return float(adfuller(x, maxlag=5, regression="c", autolag=None)[1])
    except Exception:
        # Fallback: AR(1) on the level; map the slope t-stat to a coarse p-value.
        y = np.diff(x)
        xl = x[:-1]
        X = np.column_stack([np.ones_like(xl), xl])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        n = len(y)
        s2 = (resid @ resid) / max(n - 2, 1)
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(s2 * xtx_inv[1, 1])
        t = beta[1] / se if se > 0 else 0.0
        # crude DF surface: more-negative t -> smaller p
        return float(min(max(0.5 * np.exp(0.5 * (t + 2.0)), 1e-4), 1.0))


def engle_granger_train(la_tr: np.ndarray, lb_tr: np.ndarray):
    """OLS of log A on log B over TRAIN: la = alpha + beta*lb + resid.
    Returns (alpha, beta, adf_pvalue_of_resid). beta is the hedge ratio,
    estimated on TRAIN only (no look-ahead)."""
    X = np.column_stack([np.ones_like(lb_tr), lb_tr])
    coef, *_ = np.linalg.lstsq(X, la_tr, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = la_tr - (alpha + beta * lb_tr)
    return alpha, beta, _adf_pvalue(resid)


# --------------------------------------------------------------------------- #
# Spread construction + spread OHLC (intrabar extremes bounded by leg OHLC)
# --------------------------------------------------------------------------- #
def spread_ohlc(a: pd.DataFrame, b: pd.DataFrame, beta: float):
    """Residual spread s = logA - beta*logB at each bar, plus a CONSERVATIVE
    intrabar high/low for s bounded by the leg OHLC extremes:

        beta >= 0:  s_high = logA_high - beta*logB_low
                    s_low  = logA_low  - beta*logB_high
        beta <  0:  s_high = logA_high - beta*logB_high
                    s_low  = logA_low  - beta*logB_low

    These are outer bounds (the true intrabar spread extreme cannot exceed them
    because each leg's contribution is maximised/minimised independently), so a
    first-touch TP/SL on these bounds is the pessimistic, no-look-ahead reading.
    Returns (s_open, s_high, s_low, s_close) as float64 arrays."""
    laO, laH, laL, laC = (np.log(a[c].to_numpy(np.float64)) for c in ("open", "high", "low", "close"))
    lbO, lbH, lbL, lbC = (np.log(b[c].to_numpy(np.float64)) for c in ("open", "high", "low", "close"))
    s_open = laO - beta * lbO
    s_close = laC - beta * lbC
    if beta >= 0.0:
        s_high = laH - beta * lbL
        s_low = laL - beta * lbH
    else:
        s_high = laH - beta * lbH
        s_low = laL - beta * lbL
    # guard the open/close inside the [low, high] envelope (numerical safety)
    s_high = np.maximum.reduce([s_high, s_open, s_close])
    s_low = np.minimum.reduce([s_low, s_open, s_close])
    return s_open, s_high, s_low, s_close


# --------------------------------------------------------------------------- #
# Causal rolling z-score / local sigma of the spread
# --------------------------------------------------------------------------- #
def spread_rolling_stats(s_close, span):
    """Causal EWMA mean and std of the spread level. Output at t uses only data
    <= t (no look-ahead). A rolling z is the right signal for a slowly-drifting
    cointegrated spread: a frozen multi-year mean lets the spread wander out of
    band for years, whereas the OU optimal rule is about reversion to a LOCAL
    equilibrium at the measured half-life. Returns (z, local_sigma)."""
    s = pd.Series(np.asarray(s_close, np.float64))
    mu = s.ewm(span=span, adjust=True).mean()
    sd = s.ewm(span=span, adjust=True).std()
    z = ((s - mu) / sd.replace(0, np.nan)).to_numpy()
    z[~np.isfinite(z)] = 0.0
    sig = sd.to_numpy()
    sig[~np.isfinite(sig)] = 0.0
    return z, sig


def spread_entry_events(z, entry_k, warm):
    """Fade fresh |z| >= entry_k extremes of the (causal rolling) spread z. side =
    -sign(z): long the spread (expected to rise to its local mean) when z<0, short
    when z>0. Causal: uses z[t] only. Returns (event indices, side)."""
    z = np.asarray(z, np.float64)
    over = np.abs(z) >= entry_k
    fresh = over.copy()
    fresh[1:] &= ~over[:-1]
    idx = np.where(fresh)[0]
    idx = idx[idx > warm]
    side = -np.sign(z[idx]).astype(np.int64)
    keep = side != 0
    return idx[keep].astype(np.int64), side[keep]


# --------------------------------------------------------------------------- #
# OOS spread exit applier: additive barriers in spread units, intrabar OHLC,
# per-leg costs on both legs. Numba kernel + pure-Python reference.
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _spread_exit_kernel(ev_idx, side, sig_ev, s_open, s_high, s_low, s_close,
                        pt_mult, sl_mult, max_hold):
    """Apply a (pt_mult, sl_mult) ADDITIVE rule on real OOS bars using full
    intrabar spread high/low first-touch. Barriers are pt_mult / sl_mult LOCAL
    spread sigmas wide (sig_ev[k] is the causal rolling spread std at the event,
    so the barrier scaling carries no look-ahead). Entry at next bar open
    (s_open[i0+1]); placed in the direction of the bet (side>0 expects s to RISE
    toward its local mean). Pessimistic: the adverse barrier is checked first when
    both are touched in one bar. Returns side-signed gross spread P&L, exit bar,
    label, hold."""
    n_ev = ev_idx.shape[0]
    n = s_close.shape[0]
    out_pnl = np.empty(n_ev, np.float64)
    out_touch = np.empty(n_ev, np.int64)
    out_label = np.empty(n_ev, np.int64)
    out_hold = np.empty(n_ev, np.int64)
    for k in range(n_ev):
        i0 = ev_idx[k]
        s = side[k]
        sg = sig_ev[k]
        pt_abs = pt_mult * sg
        sl_abs = sl_mult * sg
        entry_bar = i0 + 1
        if entry_bar > n - 1:
            entry_bar = n - 1
        entry = s_open[entry_bar]
        if s > 0:                      # long spread: TP above entry, SL below
            up = entry + pt_abs        # profit target (reversion up)
            dn = entry - sl_abs        # stop (further divergence down)
        else:                          # short spread: TP below entry, SL above
            dn = entry - pt_abs        # profit target (reversion down)
            up = entry + sl_abs        # stop (further divergence up)
        j_end = entry_bar + max_hold
        if j_end > n - 1:
            j_end = n - 1
        touched = -1
        label = 0
        exit_lvl = s_close[j_end]
        for j in range(entry_bar, j_end + 1):
            hi = s_high[j]
            lo = s_low[j]
            if s > 0:
                if lo <= dn:           # adverse (stop) first
                    touched = j; label = -1; exit_lvl = dn; break
                if hi >= up:
                    touched = j; label = 1; exit_lvl = up; break
            else:
                if hi >= up:           # adverse (stop) first
                    touched = j; label = -1; exit_lvl = up; break
                if lo <= dn:
                    touched = j; label = 1; exit_lvl = dn; break
        if touched < 0:
            touched = j_end; label = 0; exit_lvl = s_close[j_end]
        out_pnl[k] = s * (exit_lvl - entry)     # signed spread P&L (gross, log units)
        out_touch[k] = touched
        out_label[k] = label
        out_hold[k] = touched - entry_bar
    return out_pnl, out_touch, out_label, out_hold


def _spread_exit_reference(ev_idx, side, sig_ev, s_open, s_high, s_low, s_close,
                           pt_mult, sl_mult, max_hold):
    """Independent pure-Python reference for the spread exit kernel."""
    n = len(s_close)
    out_pnl = np.empty(len(ev_idx)); out_lab = np.empty(len(ev_idx), int)
    out_hold = np.empty(len(ev_idx), int)
    for k in range(len(ev_idx)):
        i0 = int(ev_idx[k]); s = int(side[k])
        pt_abs = pt_mult * sig_ev[k]; sl_abs = sl_mult * sig_ev[k]
        eb = min(i0 + 1, n - 1); entry = s_open[eb]
        if s > 0:
            up = entry + pt_abs; dn = entry - sl_abs
        else:
            dn = entry - pt_abs; up = entry + sl_abs
        j_end = min(eb + max_hold, n - 1)
        touched = -1; lab = 0; ex = s_close[j_end]
        for j in range(eb, j_end + 1):
            if s > 0:
                if s_low[j] <= dn: touched, lab, ex = j, -1, dn; break
                if s_high[j] >= up: touched, lab, ex = j, 1, up; break
            else:
                if s_high[j] >= up: touched, lab, ex = j, -1, up; break
                if s_low[j] <= dn: touched, lab, ex = j, 1, dn; break
        if touched < 0: touched, lab, ex = j_end, 0, s_close[j_end]
        out_pnl[k] = s * (ex - entry); out_lab[k] = lab; out_hold[k] = touched - eb
    return out_pnl, out_lab, out_hold


def apply_spread_rule(s_open, s_high, s_low, s_close, sig_ev, ev_idx, side,
                      pt_mult, sl_mult, max_hold):
    f = lambda x: np.ascontiguousarray(np.asarray(x, np.float64))
    pnl, touch, lab, hold = _spread_exit_kernel(
        np.ascontiguousarray(np.asarray(ev_idx, np.int64)),
        np.ascontiguousarray(np.asarray(side, np.int64)),
        f(sig_ev), f(s_open), f(s_high), f(s_low), f(s_close),
        float(pt_mult), float(sl_mult), int(max_hold))
    return pd.DataFrame({"ev_idx": ev_idx, "side": side, "touch": touch,
                         "label": lab, "pnl_gross": pnl, "hold": hold})


# --------------------------------------------------------------------------- #
# P&L bookkeeping
# --------------------------------------------------------------------------- #
def trade_returns_to_series(events_idx, hold, pnl_net, n_bars):
    r = np.zeros(n_bars)
    for k in range(len(events_idx)):
        i0 = int(events_idx[k]); h = max(1, int(hold[k]))
        per = pnl_net[k] / h
        j1 = min(i0 + h, n_bars)
        r[i0:j1] += per
    return r


def score_series(r, bpy=BARS_PER_YEAR):
    r = np.asarray(r, np.float64); r = r[np.isfinite(r)]
    nz = r[r != 0.0]
    sr = O.sharpe(r); sr_ann = sr * np.sqrt(bpy)
    sk = float(ss.skew(r)) if len(r) > 2 else 0.0
    ku = float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0
    gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else np.nan
    return dict(sr_per_bar=sr, sr_ann=sr_ann, pf=pf, skew=sk, kurt=ku,
                mean=float(r.mean()), n_obs=len(r))


# --------------------------------------------------------------------------- #
# Core: one pair (cointegration -> OU fit -> OU rule vs controls OOS)
# --------------------------------------------------------------------------- #
def run_pair(sym_a, sym_b, smoke=False):
    pt_grid = SMOKE_PT if smoke else PT_GRID
    sl_grid = SMOKE_SL if smoke else SL_GRID
    n_paths = SMOKE_PATHS if smoke else N_PATHS
    horizon = SMOKE_HORIZON if smoke else MC_HORIZON

    a0 = load_perp(sym_a); b0 = load_perp(sym_b)
    a, b, idx = align_pair(a0, b0)
    del a0, b0
    n = len(idx)
    if n < MIN_SHARED_BARS:
        return None

    la = np.log(a["close"].to_numpy(np.float64))
    lb = np.log(b["close"].to_numpy(np.float64))
    n_is = int(n * IS_FRAC)

    # --- Engle-Granger on TRAIN only (hedge ratio + cointegration test) ---
    alpha, beta, adf_p = engle_granger_train(la[:n_is], lb[:n_is])
    cointegrated = (adf_p <= ADF_P_MAX) and np.isfinite(beta)

    # --- spread + spread OHLC with the FROZEN train beta ---
    s_open, s_high, s_low, s_close = spread_ohlc(a, b, beta)

    # --- causal rolling z + local sigma of the spread (no look-ahead) ---
    z, local_sig = spread_rolling_stats(s_close, Z_SPAN)

    # --- OU fit on the TRAIN demeaned spread (causal). Fitting on the rolling
    # RESIDUAL (spread minus its causal rolling mean) isolates the local
    # mean-reversion the rule actually trades, rather than the slow multi-year
    # drift of the level. half-life = -ln2/ln(phi). ---
    s_ser = pd.Series(s_close)
    roll_mu = s_ser.ewm(span=Z_SPAN, adjust=True).mean().to_numpy()
    resid = s_close - roll_mu
    fit = otr.fit_ou(resid[:n_is][np.isfinite(resid[:n_is])])
    if not fit["ok"]:
        return None
    half_life = fit["half_life"]

    # --- per-leg cost: a spread RT charges per-leg cost on BOTH legs, entry+exit.
    # 2 legs * 2 fills = 4 * (per-leg bp). The spread P&L is in log units, so the
    # cost is also a log-return fraction; charge the sum across legs/fills. ---
    cpl = COST_BP_PER_LEG / 1e4
    rt_cost = 4.0 * cpl                # both legs, entry + exit

    warm = max(Z_SPAN + 5, n_is // 50)

    # --- entry events on the causal rolling spread z ---
    ev_all, side_all = spread_entry_events(z, ENTRY_K, warm)
    oos_mask = ev_all >= n_is
    ev_oos, side_oos = ev_all[oos_mask], side_all[oos_mask]
    if len(ev_oos) < MIN_OOS_TRADES:
        return None
    n_oos = n - n_is
    sig_oos = local_sig[ev_oos]        # causal local spread std at each OOS event

    # stationary std of the fitted OU residual (sigma / sqrt(1-phi^2)); the mesh
    # pt/sl come out in OU-LEVEL units, so dividing by stat_sd expresses them in
    # stationary-std units, which we then re-scale by the LOCAL spread sigma at
    # each OOS event. This keeps the rule unit-consistent and fully causal.
    stat_sd = fit["sigma"] / max(np.sqrt(1.0 - fit["phi"] ** 2), 1e-9)

    # ====================================================================== #
    # ARM 1: OU optimal rule. Enter at the deviation (entry_k away from the
    # mean), profit-take toward the mean. Reuse the study's ou_mesh_dev.
    # x0 in OU-level units sits ENTRY_K stationary-std away from E0.
    # ====================================================================== #
    x0_dev = fit["E0"] + ENTRY_K * stat_sd
    sh_mesh, _ = deepen.ou_mesh_dev(fit["E0"], fit["phi"], fit["sigma"], x0=x0_dev,
                                    pt_grid=pt_grid, sl_grid=sl_grid,
                                    n_paths=n_paths, max_horizon=horizon, seed=MC_SEED)
    pt_star, sl_star, sh_star, _ = otr.optimal_rule(sh_mesh, pt_grid, sl_grid)
    # mesh pt/sl (OU-level units) -> stationary-std multipliers
    pt_mult = pt_star / stat_sd
    sl_mult = sl_star / stat_sd
    tb_ou = apply_spread_rule(s_open, s_high, s_low, s_close, sig_oos,
                              ev_oos, side_oos, pt_mult, sl_mult, MAX_HOLD)
    pnl_ou = tb_ou["pnl_gross"].to_numpy() - rt_cost
    r_ou = trade_returns_to_series(ev_oos - n_is, tb_ou["hold"].to_numpy(), pnl_ou, n_oos)
    s_ou = score_series(r_ou)

    # ====================================================================== #
    # ARM 2 (control A): fixed-band / Bollinger on the spread. Enter at the
    # same +/- ENTRY_K extreme (shared entry), but take profit at the local mean
    # (z -> 0), i.e. the textbook band rule, with a symmetric further-band stop,
    # same max-hold and cost. Entry at z=+/-ENTRY_K means reverting to the local
    # mean covers ENTRY_K local sigmas.
    # ====================================================================== #
    band_pt = ENTRY_K                  # profit at the local mean (in local sigmas)
    band_sl = ENTRY_K                  # symmetric stop a further band away
    tb_bb = apply_spread_rule(s_open, s_high, s_low, s_close, sig_oos,
                              ev_oos, side_oos, band_pt, band_sl, MAX_HOLD)
    pnl_bb = tb_bb["pnl_gross"].to_numpy() - rt_cost
    r_bb = trade_returns_to_series(ev_oos - n_is, tb_bb["hold"].to_numpy(), pnl_bb, n_oos)
    s_bb = score_series(r_bb)

    # ====================================================================== #
    # ARM 3 (control B): buy-hold-the-spread benchmark over OOS. Hold a unit
    # long spread (logA - beta*logB) from the start of OOS to its end; per-bar
    # return = delta spread. One entry + one exit cost. This is the passive
    # "is there a trend in the spread" benchmark.
    # ====================================================================== #
    s_oos = s_close[n_is:]
    r_bh = np.zeros(n_oos)
    if n_oos > 1:
        r_bh[1:] = np.diff(s_oos)
        r_bh[1] -= rt_cost / 2.0       # entry cost amortised at start
        r_bh[-1] -= rt_cost / 2.0      # exit cost at end
    s_bh = score_series(r_bh)

    return dict(
        sym_a=sym_a, sym_b=sym_b, n_shared=n, n_is=n_is, n_oos=n_oos,
        beta=beta, adf_p=adf_p, cointegrated=bool(cointegrated),
        ou_phi=fit["phi"], ou_half_life=half_life, ou_sigma=fit["sigma"],
        stat_sd=stat_sd, pt_star=pt_star, sl_star=sl_star, mesh_sharpe=sh_star,
        n_oos_trades=int(len(ev_oos)),
        ou_sr_ann=s_ou["sr_ann"], ou_pf=s_ou["pf"], ou_n_obs=s_ou["n_obs"],
        ou_skew=s_ou["skew"], ou_kurt=s_ou["kurt"], ou_sr_pb=s_ou["sr_per_bar"],
        bb_sr_ann=s_bb["sr_ann"], bb_pf=s_bb["pf"], bb_sr_pb=s_bb["sr_per_bar"],
        bh_sr_ann=s_bh["sr_ann"], bh_pf=s_bh["pf"], bh_sr_pb=s_bh["sr_per_bar"],
        _r_ou=r_ou, _r_bb=r_bb, _r_bh=r_bh, _z=z, _s_close=s_close,
        _roll_mu=roll_mu, _local_sig=local_sig,
        _n_is=n_is, _ev_oos=ev_oos)


def add_dsr(rows):
    """Deflated Sharpe per pair. The OU mesh searches a 12x12 (pt,sl) grid on the
    fitted process -> 144 in-sample trials; that grid is the OU rule's trial set.
    The band control searches no grid (fixed rule) -> benchmark at SR0=0. We use
    the per-pair OOS Sharpe with the OU mesh trial count for the OU arm."""
    n_trials_ou = len(PT_GRID) * len(SL_GRID)
    for r in rows:
        # OU arm: deflate against the dispersion of the mesh's own trial Sharpes.
        # We approximate var[SR] across the mesh by the spread of cell Sharpes is
        # unavailable post-hoc; use the conservative var_sr = 1/n_obs proxy with
        # the full mesh trial count (False Strategy Theorem default).
        var_sr = 1.0 / max(r["ou_n_obs"], 2)
        sr0 = O.expected_max_sharpe(n_trials_ou, var_sr)
        r["ou_dsr"] = O.prob_sharpe_ratio(r["ou_sr_pb"], r["ou_n_obs"],
                                          r["ou_skew"], r["ou_kurt"], sr_benchmark=sr0)
        r["ou_sr0"] = sr0
        # band control: single fixed rule, deflate against SR0=0 (PSR vs zero).
        r["bb_dsr"] = O.prob_sharpe_ratio(r["bb_sr_pb"], r["ou_n_obs"],
                                          0.0, 3.0, sr_benchmark=0.0)
    return rows


# --------------------------------------------------------------------------- #
# Tables & figure
# --------------------------------------------------------------------------- #
def make_tables(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    cols = ["sym_a", "sym_b", "cointegrated", "adf_p", "beta",
            "ou_half_life", "ou_phi", "n_oos_trades",
            "pt_star", "sl_star",
            "ou_sr_ann", "bb_sr_ann", "bh_sr_ann",
            "ou_pf", "bb_pf", "bh_pf",
            "ou_dsr", "bb_dsr"]
    t = df[cols].copy().round(4)
    t.to_csv(os.path.join(TAB, "ou_on_spread.csv"), index=False)

    # pooled medians
    pooled = dict(
        n_pairs=len(df),
        med_ou_sr_ann=float(df["ou_sr_ann"].median()),
        med_bb_sr_ann=float(df["bb_sr_ann"].median()),
        med_bh_sr_ann=float(df["bh_sr_ann"].median()),
        med_ou_pf=float(df["ou_pf"].median()),
        med_bb_pf=float(df["bb_pf"].median()),
        med_ou_dsr=float(df["ou_dsr"].median()),
        med_bb_dsr=float(df["bb_dsr"].median()),
        ou_beats_band_sr=int((df["ou_sr_ann"] > df["bb_sr_ann"]).sum()),
        ou_beats_bh_sr=int((df["ou_sr_ann"] > df["bh_sr_ann"]).sum()),
        n_ou_dsr_gt95=int((df["ou_dsr"] > 0.95).sum()),
        n_bb_dsr_gt95=int((df["bb_dsr"] > 0.95).sum()),
        med_half_life=float(df["ou_half_life"].median()),
    )

    md = ["# OU optimal trading rule on a cointegrated residual spread\n",
          "_OU's true regime: the residual spread of a cointegrated crypto perp "
          "pair. Hedge ratio (beta) and OU parameters fit on TRAIN only; OOS "
          "trading with full intrabar OHLC exits (spread extremes bounded by leg "
          "OHLC) and per-leg costs on both legs. DSR is the headline._\n",
          "\n## Per pair\n", t.to_markdown(index=False),
          "\n\n## Pooled\n",
          pd.Series(pooled).to_frame("value").round(4).to_markdown()]
    with open(os.path.join(TAB, "ou_on_spread.md"), "w") as f:
        f.write("\n".join(md))
    print("  tables ->", os.path.abspath(TAB))
    return df, pooled


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style()
    P = style.PALETTE

    # representative pair = the genuinely cointegrated pair with the best OU OOS
    # Sharpe (this is OU's true regime); fall back to best-Sharpe if none pass.
    coint_rows = [r for r in rows if r["cointegrated"]]
    rep = max(coint_rows or rows, key=lambda r: r["ou_sr_ann"])
    n_is = rep["_n_is"]; s = rep["_s_close"]
    rmu = rep["_roll_mu"]; rsig = rep["_local_sig"]
    oos = slice(n_is, len(s))
    x = np.arange(len(s) - n_is)
    s_oos = s[oos]; mu_oos = rmu[oos]; sig_oos = rsig[oos]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    # left: OOS spread with the causal rolling mean + entry bands
    ax[0].plot(x, s_oos, color="#444444", lw=0.7, label="spread (OOS)")
    ax[0].plot(x, mu_oos, color=P["dollar"], lw=1.0, label="rolling mean")
    ax[0].plot(x, mu_oos + ENTRY_K * sig_oos, color=P["accent"], ls="--", lw=0.8,
               label=f"+/-{ENTRY_K:g} sigma entry band")
    ax[0].plot(x, mu_oos - ENTRY_K * sig_oos, color=P["accent"], ls="--", lw=0.8)
    ax[0].set_title(f"{rep['sym_a']} - {rep['beta']:.2f}*{rep['sym_b']}\n"
                    f"OU half-life={rep['ou_half_life']:.0f} bars  ADF p={rep['adf_p']:.3f}")
    ax[0].set_xlabel("OOS bar (hourly)"); ax[0].set_ylabel("residual spread (log units)")
    ax[0].legend(fontsize=8)

    # right: OOS equity, OU vs band vs buy-hold (pooled-representative)
    ax[1].plot(np.cumsum(rep["_r_bh"]), color="#bbbbbb", lw=1.0, label="buy-hold spread")
    ax[1].plot(np.cumsum(rep["_r_bb"]), color=P["tick"], lw=1.1, label="fixed-band ctrl")
    ax[1].plot(np.cumsum(rep["_r_ou"]), color=P["accent"], lw=1.3, label="OU optimal rule")
    ax[1].set_title(f"OOS equity (net): OU rule vs controls\n"
                    f"OU SR {rep['ou_sr_ann']:.2f}  band SR {rep['bb_sr_ann']:.2f}  "
                    f"OU DSR {rep['ou_dsr']:.2f}")
    ax[1].set_xlabel("OOS bar (hourly)"); ax[1].set_ylabel("cum spread P&L (net, log units)")
    ax[1].legend(fontsize=8)
    fig.suptitle("OU optimal trading rule on a cointegrated residual spread (OU's true regime)")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_ou_on_spread.png")
    fig.savefig(out); plt.close(fig)
    print("  figure ->", os.path.abspath(out))


# --------------------------------------------------------------------------- #
# Verification: spread-exit kernel bit-identical check
# --------------------------------------------------------------------------- #
def verify():
    print("=== spread-exit kernel: Numba vs pure-Python reference ===")
    rng = np.random.default_rng(0)
    s = np.cumsum(rng.standard_normal(2000) * 0.01)
    s_close = s
    s_high = s + np.abs(rng.standard_normal(2000)) * 0.003
    s_low = s - np.abs(rng.standard_normal(2000)) * 0.003
    s_open = s + rng.standard_normal(2000) * 0.001
    s_high = np.maximum.reduce([s_high, s_open, s_close])
    s_low = np.minimum.reduce([s_low, s_open, s_close])
    ev = np.arange(60, 1900, 17, dtype=np.int64)
    side = np.where(rng.standard_normal(len(ev)) > 0, 1, -1).astype(np.int64)
    sig_ev = 0.005 + 0.002 * np.abs(rng.standard_normal(len(ev)))
    tb = apply_spread_rule(s_open, s_high, s_low, s_close, sig_ev, ev, side, 1.5, 1.0, 100)
    ref_pnl, ref_lab, ref_hold = _spread_exit_reference(
        ev, side, sig_ev, s_open, s_high, s_low, s_close, 1.5, 1.0, 100)
    dpnl = float(np.max(np.abs(tb["pnl_gross"].to_numpy() - ref_pnl)))
    dlab = int(np.max(np.abs(tb["label"].to_numpy() - ref_lab)))
    dhold = int(np.max(np.abs(tb["hold"].to_numpy() - ref_hold)))
    print(f"  pnl_gross max|d| = {dpnl:.3e}   label max|d| = {dlab}   hold max|d| = {dhold}")
    print("verification done." if dpnl < 1e-12 and dlab == 0 and dhold == 0 else "MISMATCH")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny MC, quick pass")
    ap.add_argument("--verify", action="store_true", help="kernel bit-identical check only")
    args = ap.parse_args()

    if args.verify:
        verify(); return

    t0 = time.perf_counter()
    print(f"screening {len(CANDIDATE_PAIRS)} candidate pairs for cointegration "
          f"(Engle-Granger on TRAIN log-prices, ADF p<={ADF_P_MAX}) ...")
    screened = []
    for sym_a, sym_b in CANDIDATE_PAIRS:
        try:
            r = run_pair(sym_a, sym_b, smoke=args.smoke)
        except Exception as e:
            print(f"  [ERR]  {sym_a:10s}/{sym_b:10s}: {e}")
            continue
        if r is None:
            print(f"  [skip] {sym_a:10s}/{sym_b:10s} (insufficient shared history / OU fit / trades)")
            continue
        tag = "COINT" if r["cointegrated"] else "no-coint"
        print(f"  [{tag:8s}] {sym_a:10s}/{sym_b:10s} ADF p={r['adf_p']:.3f} "
              f"beta={r['beta']:.3f} HL={r['ou_half_life']:.0f} "
              f"OU_SR={r['ou_sr_ann']:.2f} band_SR={r['bb_sr_ann']:.2f} "
              f"OU_PF={r['ou_pf']:.3f}")
        screened.append(r)

    # keep cointegrated pairs (fall back to lowest-ADF if too few pass)
    coint = [r for r in screened if r["cointegrated"]]
    if len(coint) < 2:
        coint = sorted(screened, key=lambda r: r["adf_p"])[:max(2, N_KEEP)]
        print(f"\n  (only {sum(r['cointegrated'] for r in screened)} pairs passed ADF; "
              f"keeping the {len(coint)} lowest-ADF pairs for the OU test)")
    coint = sorted(coint, key=lambda r: r["adf_p"])[:N_KEEP]
    print(f"\nkeeping {len(coint)} pairs for the OU-on-spread headline")

    if not coint:
        print("no usable pairs"); return

    coint = add_dsr(coint)
    df, pooled = make_tables(coint)
    try:
        make_figure(coint)
    except Exception as e:
        print(f"  [figure skipped] {e}")

    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== POOLED ===")
    for k, v in pooled.items():
        print(f"  {k:22s} {v}")
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\nRAM peak (max RSS): {peak_kb/1e6:.2f} GB")


if __name__ == "__main__":
    main()
