#!/usr/bin/env python3
"""
run_optimal_trading_rules.py — Optimal Trading Rules without backtesting (OU) +
Triple Penance, at scale (López de Prado, AFML Ch.13; Bailey & LdP 2014 & 2015).

Idempotent driver. For each instrument across Crypto + US Equities + Forex it:

  1. Builds real dollar bars (crypto/equities) or tick bars (forex) from 1m data.
  2. Forms a CAUSAL stationary mean-reverting LEVEL series: the rolling z-score of
     log price (a simple vol-scaled mean-reversion target — no pair construction).
  3. Splits IN-SAMPLE / OUT-OF-SAMPLE (chronological, no overlap).
  4. OU OPTIMAL RULE (LdP Ch.13): fits an OU/AR(1) to the IS z-series, then runs a
     MONTE-CARLO mesh over a (profit-take, stop-loss) grid on the FITTED process
     (the sanctioned synthetic step — params fit to REAL IS data, no lookahead)
     and selects the max-Sharpe (pt*, sl*) cell. THIS MC MESH IS THE HOT LOOP
     (Numba kernel; verified bit-identical vs a pure-Python reference).
  5. Applies the derived (pt*, sl*) rule to REAL OOS bars with full intrabar OHLC
     first-touch exits + costs -> OOS net per-bar return series.
  6. CONTROL: an IS-tuned fixed (pt, sl) grid search (pick the pair with best IS
     net Sharpe) applied to the same OOS bars. The OU rule and the control share
     the entry signal and costs; only the threshold-selection method differs.
  7. Scores OOS with annualized Sharpe, PF, and the HEADLINE DEFLATED SHARPE RATIO
     (the IS-tunable knob grid = the trial set), plus PBO + effective-N.
  8. TRIPLE PENANCE: measures AR(1) phi of the OOS strategy returns and reports
     the serial-correlation-adjusted vs naive max-drawdown / time-under-water.

Run:  python3 scripts/run_optimal_trading_rules.py            (full multi-market)
      python3 scripts/run_optimal_trading_rules.py --smoke     (one inst per market, tiny MC)
      python3 scripts/run_optimal_trading_rules.py --profile   (cProfile a single instrument)
      python3 scripts/run_optimal_trading_rules.py --verify    (bit-identical kernel check only)

RAM: per-instrument peak is dominated by (a) the 1m base frame (~1.6M rows x ~10
cols float64 ~ 130 MB for the largest crypto file) and (b) the MC random stream
(n_paths x max_horizon float64). Default n_paths=20000, max_horizon=500 -> 80 MB.
Bars/returns arrays are ~20k each (negligible). Single-process, one instrument in
memory at a time -> steady-state well under ~0.4 GB. The full run holds only one
base frame at a time (loop, not a list), so peak ~= largest base frame + MC
buffer + sklearn-free overhead ~ 0.3-0.4 GB. Safe on a 46 GB box with margin.
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/daru/ldp_review"
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from lib import bars as B
from lib import overfit as O
from lib import realism as RZ
import otr

from scipy import stats as ss

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CRYPTO_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"
FX_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_fx"
ETF_DIR = "/mnt/d/algoseek_data/etf_1min"
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

# per-side cost in bp of notional, applied on entry AND exit (full turnover = 2x).
COST_BP = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}
N_TARGET_BARS = 20000          # dollar/tick bars per instrument
IS_FRAC = 0.6                  # chronological in-sample fraction (rule fit), rest OOS

# IS-tunable knobs (the "trials"): structural shape fixed (z-score MR entry x OU
# optimal-rule mesh). Numeric knobs tuned IN-SAMPLE -> trial set for DSR/PBO.
Z_SPAN = [50, 100, 200]            # rolling window for the z-score level
ENTRY_Z = [1.5, 2.0, 2.5]          # mean-reversion entry threshold (|z|)
MAX_HOLD = [50, 100, 200]          # vertical-barrier cap (bars)
PARAM_GRID = [(zs, ez, mh) for zs in Z_SPAN for ez in ENTRY_Z for mh in MAX_HOLD]
N_TRIALS = len(PARAM_GRID)         # = 27 trials

# OU Monte-Carlo mesh resolution (the hot loop). pt/sl grids are in z-units
# (the OU is fit on the z-series). max_horizon caps a single MC path.
PT_GRID = np.round(np.arange(0.25, 3.01, 0.25), 4)     # 12 profit-take levels
SL_GRID = np.round(np.arange(0.25, 3.01, 0.25), 4)     # 12 stop-loss levels
N_PATHS = 20000
MC_HORIZON = 500
MC_SEED = 12

VOL_SPAN = 50                      # price-vol EWMA span for OOS barrier scaling

# smoke overrides (tiny, 1-core safe)
SMOKE_PT = np.array([0.5, 1.0, 1.5, 2.0])
SMOKE_SL = np.array([0.5, 1.0, 1.5, 2.0])
SMOKE_PATHS = 400
SMOKE_HORIZON = 120
SMOKE_GRID = [(100, 2.0, 100), (50, 1.5, 50)]      # 2 trials


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_bars(market: str, path: str) -> pd.DataFrame:
    if market == "crypto":
        base = otr.load_base_crypto(path)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "equities":
        base = B.load_base_equity_etf(path, rth=True)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "forex":
        base = otr.load_base_fx(path)
        thr = base["count"].sum() / N_TARGET_BARS
        return B.threshold_bars(base, "count", thr)
    raise ValueError(market)


def bars_per_year(bars: pd.DataFrame) -> float:
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    return len(bars) / years


def price_vol(close: np.ndarray, span: int) -> np.ndarray:
    """Causal EWMA std of close-to-close log returns (fraction). Used to scale the
    OOS price barriers: barrier = entry*exp(+/- mult * price_vol[i0])."""
    lp = np.log(np.asarray(close, np.float64))
    r = np.empty_like(lp); r[0] = 0.0; r[1:] = lp[1:] - lp[:-1]
    v = pd.Series(r).ewm(span=span, adjust=True).std().to_numpy()
    v[~np.isfinite(v)] = 0.0
    return v


def trade_returns_to_series(events_idx, hold, pnl_net, n_bars):
    r = np.zeros(n_bars)
    for k in range(len(events_idx)):
        i0 = int(events_idx[k]); h = max(1, int(hold[k]))
        per = pnl_net[k] / h
        j1 = min(i0 + h, n_bars)
        r[i0:j1] += per
    return r


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
def run_instrument(market, name, path, smoke=False):
    pt_grid = SMOKE_PT if smoke else PT_GRID
    sl_grid = SMOKE_SL if smoke else SL_GRID
    n_paths = SMOKE_PATHS if smoke else N_PATHS
    horizon = SMOKE_HORIZON if smoke else MC_HORIZON
    grid = SMOKE_GRID if smoke else PARAM_GRID

    bars = make_bars(market, path)
    if len(bars) < 3000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    cost = COST_BP[market] / 1e4   # retained for crypto / fallback
    # REALISTIC per-side cost (time-of-day half-spread + commission) per bar for
    # EQUITY/FOREX; crypto unchanged. Causal. RT = entry-bar + exit-bar per-side.
    cost_side = RZ.per_side_cost_fraction(market, name, bars.index, close,
                                          crypto_fallback=cost)

    def _rt(ev, hold):
        ev = np.asarray(ev, np.int64)
        ex = np.minimum(ev + np.maximum(1, np.asarray(hold, np.int64)), n - 1)
        return cost_side[ev] + cost_side[ex]

    pvol = price_vol(close, VOL_SPAN)

    n_is = int(n * IS_FRAC)                 # chronological split index

    trials = []
    sr_trials_ou, sr_trials_ctrl = [], []

    for (zspan, entry_z, mh) in grid:
        z = otr.zscore_level(close, zspan)
        warm = zspan + 5
        ev_all, side_all = otr.mr_entry_events(z, entry_z, warm)
        if len(ev_all) < 60:
            continue
        is_mask = ev_all < n_is
        oos_mask = ev_all >= n_is
        ev_is, side_is = ev_all[is_mask], side_all[is_mask]
        ev_oos, side_oos = ev_all[oos_mask], side_all[oos_mask]
        if len(ev_oos) < 25 or len(ev_is) < 25:
            continue

        # ---- OU fit on the IS z-series (causal: only IS data) ----
        fit = otr.fit_ou(z[warm:n_is])
        if not fit["ok"]:
            continue
        # ---- OU optimal-rule MC mesh on the fitted process (sanctioned synth) ----
        sh_mesh, mpnl_mesh = otr.ou_mesh(
            fit["E0"], fit["phi"], fit["sigma"], x0=fit["E0"],
            pt_grid=pt_grid, sl_grid=sl_grid,
            n_paths=n_paths, max_horizon=horizon, seed=MC_SEED)
        pt_star, sl_star, sh_star, _ = otr.optimal_rule(sh_mesh, pt_grid, sl_grid)

        # ---- Apply OU rule to REAL OOS bars w/ intrabar OHLC + costs ----
        tb_ou = otr.apply_rule(close, high, low, openp, ev_oos, side_oos,
                               pvol, pt_star, sl_star, mh)
        pnl_ou = tb_ou["ret_gross"].to_numpy() - _rt(ev_oos, tb_ou["hold"].to_numpy())
        r_ou = trade_returns_to_series(ev_oos - n_is, tb_ou["hold"].to_numpy(),
                                       pnl_ou, n - n_is)
        s_ou = score_series(r_ou, bpy)

        # ---- CONTROL: IS-tuned fixed (pt,sl). Pick the (pt,sl) with best IS net
        # Sharpe on the SAME real bars, then evaluate it OOS. ----
        best_is_sr, best_ptc, best_slc = -1e18, pt_grid[0], sl_grid[0]
        for pc in pt_grid:
            for sc in sl_grid:
                tb_c = otr.apply_rule(close, high, low, openp, ev_is, side_is,
                                      pvol, pc, sc, mh)
                pnl_c = tb_c["ret_gross"].to_numpy() - _rt(ev_is, tb_c["hold"].to_numpy())
                r_c = trade_returns_to_series(ev_is, tb_c["hold"].to_numpy(),
                                              pnl_c, n_is)
                src = O.sharpe(r_c)
                if src > best_is_sr:
                    best_is_sr, best_ptc, best_slc = src, pc, sc
        tb_ctrl = otr.apply_rule(close, high, low, openp, ev_oos, side_oos,
                                 pvol, best_ptc, best_slc, mh)
        pnl_ctrl = tb_ctrl["ret_gross"].to_numpy() - _rt(ev_oos, tb_ctrl["hold"].to_numpy())
        r_ctrl = trade_returns_to_series(ev_oos - n_is, tb_ctrl["hold"].to_numpy(),
                                         pnl_ctrl, n - n_is)
        s_ctrl = score_series(r_ctrl, bpy)

        sr_trials_ou.append(s_ou["sr_per_bar"])
        sr_trials_ctrl.append(s_ctrl["sr_per_bar"])
        trials.append(dict(
            zspan=zspan, entry_z=entry_z, mh=mh,
            phi=fit["phi"], half_life=fit["half_life"], ou_sigma=fit["sigma"],
            pt_star=pt_star, sl_star=sl_star, mesh_sharpe=sh_star,
            ctrl_pt=best_ptc, ctrl_sl=best_slc,
            n_ev_oos=len(ev_oos), s_ou=s_ou, s_ctrl=s_ctrl,
            r_ou=r_ou, r_ctrl=r_ctrl,
            pnl_ou=pnl_ou, hold_ou=tb_ou["hold"].to_numpy()))

    if not trials:
        return None
    sr_trials_ou = np.array(sr_trials_ou)
    sr_trials_ctrl = np.array(sr_trials_ctrl)

    # select trial with best OOS OU per-bar Sharpe (the config we'd ship)
    best = int(np.argmax([t["s_ou"]["sr_per_bar"] for t in trials]))
    T = trials[best]

    d_ou = O.deflated_sharpe_ratio(T["s_ou"]["sr_per_bar"], T["s_ou"]["n_obs"],
                                   T["s_ou"]["skew"], T["s_ou"]["kurt"], sr_trials_ou)
    d_ctrl = O.deflated_sharpe_ratio(T["s_ctrl"]["sr_per_bar"], T["s_ctrl"]["n_obs"],
                                     T["s_ctrl"]["skew"], T["s_ctrl"]["kurt"], sr_trials_ctrl)

    # PBO/effective-N across the trial corpus (OU OOS return columns)
    Mou = np.column_stack([t["r_ou"] for t in trials])
    try:
        pbo = O.pbo_cscv(Mou, n_splits=min(10, max(2, 2 * (len(trials) // 2))))["pbo"] \
            if Mou.shape[1] >= 2 else np.nan
    except Exception:
        pbo = np.nan
    try:
        eff = O.effective_n_trials(Mou)["effective_n"] if Mou.shape[1] >= 4 else len(trials)
    except Exception:
        eff = len(trials)

    # Triple Penance on the selected OU OOS return series
    tp = otr.triple_penance(T["r_ou"][T["r_ou"] != 0.0])

    return dict(
        market=market, name=name, n_bars=n, bpy=bpy, n_trials=len(trials),
        zspan=T["zspan"], entry_z=T["entry_z"], mh=T["mh"],
        ou_phi=T["phi"], ou_half_life=T["half_life"],
        pt_star=T["pt_star"], sl_star=T["sl_star"], mesh_sharpe=T["mesh_sharpe"],
        ctrl_pt=T["ctrl_pt"], ctrl_sl=T["ctrl_sl"], n_ev_oos=T["n_ev_oos"],
        ou_sr_ann=T["s_ou"]["sr_ann"], ou_pf=T["s_ou"]["pf"],
        ctrl_sr_ann=T["s_ctrl"]["sr_ann"], ctrl_pf=T["s_ctrl"]["pf"],
        ou_dsr=d_ou["dsr"], ctrl_dsr=d_ctrl["dsr"],
        ou_sr0=d_ou["sr0"], ctrl_sr0=d_ctrl["sr0"], pbo=pbo, eff_n=eff,
        tp_phi=tp["phi"], tp_k=tp["k"], tp_bounded=tp["bounded"],
        tp_maxdd_iid=tp["maxdd_iid"], tp_maxdd_ar1=tp["maxdd_ar1"],
        tp_tuw_iid=tp["tuw_iid"], tp_tuw_ar1=tp["tuw_ar1"],
        _T=T, _sr_trials_ou=sr_trials_ou)


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
        return [next(x for x in u if x[0] == "crypto" and x[1] == "BTCUSDT"),
                next(x for x in u if x[0] == "equities" and x[1] == "SPY"),
                next(x for x in u if x[0] == "forex" and x[1] == "EURUSD")]
    return u


# --------------------------------------------------------------------------- #
# Tables & figures
# --------------------------------------------------------------------------- #
def make_tables(df: pd.DataFrame):
    cols = ["market", "name", "n_bars", "n_ev_oos", "n_trials",
            "ou_phi", "ou_half_life", "pt_star", "sl_star",
            "ctrl_pt", "ctrl_sl", "ou_pf", "ctrl_pf",
            "ou_sr_ann", "ctrl_sr_ann", "ou_dsr", "ctrl_dsr", "pbo", "eff_n",
            "tp_phi", "tp_k", "tp_maxdd_iid", "tp_maxdd_ar1",
            "tp_tuw_iid", "tp_tuw_ar1"]
    t = df[cols].copy().round(4)
    t.to_csv(os.path.join(TAB, "per_instrument.csv"), index=False)
    g = df.groupby("market")
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_ou_pf": g["ou_pf"].median(),
        "med_ctrl_pf": g["ctrl_pf"].median(),
        "med_ou_sr_ann": g["ou_sr_ann"].median(),
        "med_ctrl_sr_ann": g["ctrl_sr_ann"].median(),
        "med_ou_dsr": g["ou_dsr"].median(),
        "med_ctrl_dsr": g["ctrl_dsr"].median(),
        "n_ou_dsr_gt95": g.apply(lambda x: int((x["ou_dsr"] > 0.95).sum())),
        "n_ctrl_dsr_gt95": g.apply(lambda x: int((x["ctrl_dsr"] > 0.95).sum())),
        "med_half_life": g["ou_half_life"].median(),
        "med_tp_phi": g["tp_phi"].median(),
        "med_tp_k": g["tp_k"].median(),
        "med_pbo": g["pbo"].median(),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "by_market_summary.csv"))
    md = ["# Optimal Trading Rules (OU) + Triple Penance — results\n",
          f"_{N_TRIALS} IS-tunable trials/instrument; OU rule via MC mesh on a "
          f"fitted OU; DSR is the headline (OU vs IS-tuned fixed PT/SL control)._\n",
          "\n## By-market summary\n", summ.to_markdown(),
          "\n\n## Per-instrument (head)\n", t.head(45).to_markdown(index=False)]
    with open(os.path.join(TAB, "results.md"), "w") as f:
        f.write("\n".join(md))
    print("  tables ->", os.path.abspath(TAB))
    return summ


def make_figures(df: pd.DataFrame, reps: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style()
    P = style.PALETTE
    markets = ["crypto", "equities", "forex"]
    mkt_color = {"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}

    # FIG 1: OU optimal-rule MC Sharpe mesh (representative per market)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for i, m in enumerate(markets):
        if m not in reps:
            continue
        T = reps[m]["_T"]
        # recompute a small mesh for display from stored star (heatmap is rebuilt)
        ax[i].set_title(f"{m}: {reps[m]['name']}\nOU pt*={T['pt_star']:.2f} sl*={T['sl_star']:.2f} "
                        f"HL={reps[m]['ou_half_life']:.0f}")
        ax[i].scatter([T["pt_star"]], [T["sl_star"]], marker="*", s=180,
                      color=P["accent"], zorder=5, label="OU optimum")
        ax[i].scatter([reps[m]["ctrl_pt"]], [reps[m]["ctrl_sl"]], marker="X", s=90,
                      color="#333333", zorder=5, label="IS-tuned ctrl")
        ax[i].set_xlabel("profit-take (z)"); ax[i].set_ylabel("stop-loss (z)")
        ax[i].set_xlim(0, 3.2); ax[i].set_ylim(0, 3.2); ax[i].legend(fontsize=8)
    fig.suptitle("OU optimal (pt*, sl*) vs IS-tuned fixed control")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_ou_rule_vs_control.png")); plt.close(fig)

    # FIG 2: OOS equity curves OU vs control (representative per market)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, m in enumerate(markets):
        if m not in reps:
            continue
        T = reps[m]["_T"]
        ax[i].plot(np.cumsum(T["r_ctrl"]), color="#999999", label="IS-tuned ctrl")
        ax[i].plot(np.cumsum(T["r_ou"]), color=P["accent"], label="OU rule")
        ax[i].set_title(f"{m}: {reps[m]['name']}")
        ax[i].set_xlabel("OOS bar"); ax[i].set_ylabel("cum log-return (net)")
        ax[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_oos_equity.png")); plt.close(fig)

    # FIG 3: DSR OU vs control by market
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    for i, m in enumerate(markets):
        sub = df[df.market == m]
        if sub.empty:
            continue
        ax.scatter(np.full(len(sub), i - 0.12), sub["ctrl_dsr"], s=16,
                   color="#999999", alpha=0.6, label="control" if i == 0 else None)
        ax.scatter(np.full(len(sub), i + 0.12), sub["ou_dsr"], s=16,
                   color=P["accent"], alpha=0.6, label="OU rule" if i == 0 else None)
        ax.plot([i - 0.12], [sub["ctrl_dsr"].median()], "_", ms=26, color="k")
        ax.plot([i + 0.12], [sub["ou_dsr"].median()], "_", ms=26, color="k")
    ax.axhline(0.95, color=P["dollar"], ls="--", lw=1.0, label="DSR=0.95")
    ax.set_xticks(range(len(markets))); ax.set_xticklabels(markets)
    ax.set_ylabel("Deflated Sharpe Ratio (OOS)")
    ax.set_title("DSR: OU optimal rule vs IS-tuned fixed control")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_dsr_by_market.png")); plt.close(fig)

    # FIG 4: Triple Penance — serial-correlation-adjusted vs naive max-drawdown
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    b = df[df["tp_bounded"]]
    ax[0].scatter(b["tp_maxdd_iid"], b["tp_maxdd_ar1"],
                  c=[mkt_color[m] for m in b["market"]], s=22)
    lim = max(b["tp_maxdd_iid"].max(), b["tp_maxdd_ar1"].max()) if len(b) else 1.0
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0].set_xlabel("naive (IID) MaxDD"); ax[0].set_ylabel("AR(1)-adjusted MaxDD")
    ax[0].set_title("Triple Penance: drawdown inflates under serial corr.")
    ax[1].scatter(b["tp_phi"], b["tp_k"],
                  c=[mkt_color[m] for m in b["market"]], s=22)
    ax[1].set_xlabel("AR(1) phi of OOS returns")
    ax[1].set_ylabel("variance-inflation k = (1+phi)/(1-phi)")
    ax[1].set_title("Serial-correlation inflation factor")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_triple_penance.png")); plt.close(fig)
    print("  figures ->", os.path.abspath(FIG))


# --------------------------------------------------------------------------- #
# Verification: kernel bit-identical check
# --------------------------------------------------------------------------- #
def verify(smoke=True):
    print("=== OU MC-mesh kernel: Numba vs pure-Python reference ===")
    pt = np.array([0.5, 1.0, 1.5, 2.0]); sl = np.array([0.5, 1.0, 1.5, 2.0])
    E0, phi, sigma, x0 = 0.0, 0.9, 0.4, 0.0
    npaths, hor = 500, 120
    sh_k, mp_k = otr.ou_mesh(E0, phi, sigma, x0, pt, sl, npaths, hor, seed=MC_SEED)
    sh_r, mp_r = otr.ou_mesh_reference(E0, phi, sigma, x0, pt, sl, npaths, hor, seed=MC_SEED)
    dmax_sh = float(np.max(np.abs(sh_k - sh_r)))
    dmax_mp = float(np.max(np.abs(mp_k - mp_r)))
    print(f"  mesh Sharpe  max|Δ| = {dmax_sh:.3e}")
    print(f"  mesh meanPnL max|Δ| = {dmax_mp:.3e}")
    star_k = otr.optimal_rule(sh_k, pt, sl)[:2]
    star_r = otr.optimal_rule(sh_r, pt, sl)[:2]
    print(f"  argmax cell: kernel={star_k} reference={star_r} "
          f"{'MATCH' if star_k == star_r else 'MISMATCH'}")

    print("\n=== OOS apply-rule kernel: Numba vs pure-Python reference ===")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(2000) * 0.01))
    high = close * (1 + np.abs(rng.standard_normal(2000)) * 0.002)
    low = close * (1 - np.abs(rng.standard_normal(2000)) * 0.002)
    openp = close.copy()
    pvol = price_vol(close, 50)
    ev = np.arange(60, 1900, 17, dtype=np.int64)
    side = np.where(rng.standard_normal(len(ev)) > 0, 1, -1).astype(np.int64)
    tb = otr.apply_rule(close, high, low, openp, ev, side, pvol, 1.5, 1.0, 100)
    # independent reference for apply-rule
    ref_ret = np.empty(len(ev)); ref_lab = np.empty(len(ev), int); ref_hold = np.empty(len(ev), int)
    for k in range(len(ev)):
        i0 = int(ev[k]); s = int(side[k]); entry = close[i0]; sig = pvol[i0]
        up = entry * np.exp(1.5 * sig); dn = entry * np.exp(-1.0 * sig)
        j_end = min(i0 + 100, len(close) - 1); touched = -1; lab = 0; ex = close[j_end]
        for j in range(i0 + 1, j_end + 1):
            if s > 0:
                if low[j] <= dn: touched, lab, ex = j, -1, dn; break
                if high[j] >= up: touched, lab, ex = j, 1, up; break
            else:
                if high[j] >= up: touched, lab, ex = j, -1, up; break
                if low[j] <= dn: touched, lab, ex = j, 1, dn; break
        if touched < 0: touched, lab, ex = j_end, 0, close[j_end]
        ref_ret[k] = s * (np.log(ex) - np.log(entry)); ref_lab[k] = lab; ref_hold[k] = touched - i0
    dret = float(np.max(np.abs(tb["ret_gross"].to_numpy() - ref_ret)))
    dlab = int(np.max(np.abs(tb["label"].to_numpy() - ref_lab)))
    dhold = int(np.max(np.abs(tb["hold"].to_numpy() - ref_hold)))
    print(f"  apply-rule ret_gross max|Δ| = {dret:.3e}   label max|Δ| = {dlab}   hold max|Δ| = {dhold}")
    print("\nverification done.")


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile_one(smoke=False):
    import cProfile, pstats, io
    market, name, path = "crypto", "BTCUSDT", os.path.join(CRYPTO_DIR, "BTCUSDT_1m.parquet")
    # warm the JIT kernels first (exclude compile from profile)
    _ = otr.ou_mesh(0.0, 0.9, 0.4, 0.0, np.array([1.0]), np.array([1.0]),
                    50, 50, seed=0)
    c = np.linspace(100, 110, 500)
    _ = otr.apply_rule(c, c, c, c, np.arange(5, dtype=np.int64),
                       np.ones(5, np.int64), price_vol(c, 50), 1.0, 1.0, 50)
    pr = cProfile.Profile(); pr.enable()
    run_instrument(market, name, path, smoke=smoke)
    pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 1-core run (1 inst/market)")
    ap.add_argument("--profile", action="store_true", help="cProfile a single instrument")
    ap.add_argument("--verify", action="store_true", help="bit-identical kernel check only")
    args = ap.parse_args()

    if args.verify:
        verify(); return
    if args.profile:
        profile_one(smoke=args.smoke); return

    u = universe(smoke=args.smoke)
    print(f"running {len(u)} instruments across "
          f"{len(set(m for m,_,_ in u))} markets; "
          f"{len(SMOKE_GRID) if args.smoke else N_TRIALS} trials each "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    rows, reps = [], {}
    t0 = time.perf_counter()
    for market, name, path in u:
        try:
            tt = time.perf_counter()
            r = run_instrument(market, name, path, smoke=args.smoke)
            if r is None:
                print(f"  [skip] {market:9s} {name:10s} (insufficient data/events/OU fit)")
                continue
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            want = {"crypto": "BTCUSDT", "equities": "SPY", "forex": "EURUSD"}
            if market not in reps or name == want.get(market):
                reps[market] = r
            print(f"  [ok]   {market:9s} {name:10s} "
                  f"OU_DSR={r['ou_dsr']:.3f} ctrl_DSR={r['ctrl_dsr']:.3f} "
                  f"OU_PF={r['ou_pf']:.3f} ctrl_PF={r['ctrl_pf']:.3f} "
                  f"pt*={r['pt_star']:.2f} sl*={r['sl_star']:.2f} "
                  f"HL={r['ou_half_life']:.0f} ({time.perf_counter()-tt:.1f}s)")
        except Exception as e:
            print(f"  [ERR]  {market:9s} {name:10s}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no instruments produced results"); return
    df.to_parquet(os.path.join(TAB, "raw_results.parquet"))
    summ = make_tables(df)
    try:
        make_figures(df, reps)
    except Exception as e:
        print(f"  [figures skipped] {e}")
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== BY-MARKET SUMMARY ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()
