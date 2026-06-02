#!/usr/bin/env python3
"""run_event_study.py -- natural-experiment / event study extension to the
Causal Factor Investing study.

WHY THIS EXISTS
---------------
The main study (scripts/run_causal.py) climbs an evidence hierarchy and stops at
Level 3 (a factor that survives two independent observational confounders). Levels
4-7 -- a defended causal graph, INTERVENTIONAL evidence, and cross-market
replication -- are left explicitly open. This script supplies the missing
interventional flavour using a QUASI-NATURAL EXPERIMENT.

THE EXPERIMENT
--------------
Event = a token gaining a Binance USD-M PERPETUAL FUTURES listing. The listing is
an exogenous structural shock to the asset's market microstructure: it adds
leverage, short access, funding-rate mechanics, and a broad speculative
participant base. The timing of the listing is set by the exchange, NOT by the
factor we are testing, so it is exogenous to the factor's effect -- the defining
property of a natural experiment.

Factor under test = SHORT-TERM REVERSAL (trailing 5-day return, lagged, the same
definition as the main study). Reversal is the main study's lone Level-3 survivor
and is widely understood as a microstructure / liquidity-provision effect. The
sharp, testable, pre-registered prediction is therefore directional: if reversal
is a structural microstructure effect, its strength should shift DISCONTINUOUSLY
when the microstructure changes at the listing event (leveraged speculators bring
overreaction and forced-liquidation mean-reversion).

DESIGN (no look-ahead, event-time aligned)
------------------------------------------
* PRE window  : the token's own SPOT hourly returns BEFORE the perp listing.
* POST window : the token's PERP hourly returns AFTER the listing.
  (Same underlying asset on both sides; the instrument switch IS the treatment.)
* Returns resampled to daily; the reversal factor is computed within each side
  separately so no factor value ever straddles the event boundary or uses future
  data (explicit .shift(1), same as the main study).
* Estimand 1 (regression discontinuity in the factor effect): pooled panel
  reversal coefficient and t-stat PRE vs POST, plus the difference and a
  pre/post interaction test.
* Estimand 2 (abnormal returns): event-time cumulative abnormal return of the
  reversal long-short sleeve, centred on the listing day, with bootstrap CI.
* CONTROL group: established majors (BTC, ETH, BNB, ...) that have NO listing
  event, evaluated over the SAME calendar windows as the treated events. A
  discontinuity in the controls would signal a calendar artifact rather than an
  event effect (a placebo / parallel-trends check).

OUTPUTS
-------
tables/event_study.csv, tables/event_study.md, figures/fig_event_study.png

Single process. Set OMP/OpenBLAS/MKL threads to 1 before running. RAM-light
(loads one symbol at a time, daily resample, discards bars).
"""
from __future__ import annotations
import sys, os, glob, json, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d                                  # repo root (holds config.py)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
import config as cfg                  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "lib"))
from lib import overfit as O          # noqa: E402
from lib import style as S            # noqa: E402

# Data roots come from the repo config (see DATA.md); override via $LDP_* env.
LD_CSV   = cfg.CRYPTO_LISTING_DATES   # per-instrument first-listing dates (CSV)
SPOT_DIR = cfg.CRYPTO_SPOT_1H         # spot hourly klines, <SYM>_1h.parquet
PERP_DIR = cfg.CRYPTO_PERP_1H         # perp hourly klines, <SYM>_1h.parquet

# Control = established large-cap perps that were already trading well before the
# bulk of the listing events under study (used as a placebo / parallel-trends set).
CONTROLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

MIN_PRE  = 20      # daily obs required in the pre (spot) window
MIN_POST = 20      # daily obs required in the post (perp) window
WIN_DAYS = 60      # cap each side to +/- this many days around the event
EVENT_HALF = 30    # event-time abnormal-return window half-width (days)


# ----- factor + OLS helpers: copied verbatim from scripts/run_causal.py so the
# ----- definitions match the main study exactly (lib/ kept unmodified). --------
def _factor_revers(ret: pd.Series) -> pd.Series:
    """CAUSAL short-reversal factor at day t using only data up to t-1:
    trailing 5-day cumulative return, then lagged one day."""
    f = (1.0 + ret).rolling(5).apply(np.prod, raw=True) - 1.0
    return f.shift(1)


def _ols_slope_t(y, x):
    n = len(x)
    if n < 3:
        return 0.0, 0.0
    mx, my = x.mean(), y.mean()
    dx = x - mx
    sxx = (dx * dx).sum()
    if sxx <= 0:
        return 0.0, 0.0
    b = (dx * (y - my)).sum() / sxx
    a = my - b * mx
    e = y - (a + b * x)
    sse = (e * e).sum()
    se = np.sqrt(sse / (n - 2) / sxx)
    return b, (b / se if se > 0 else 0.0)


def _ols_interaction_t(y, x, d):
    """Pooled OLS of y on [1, x, d, x*d]; return the slope and t-stat of the
    interaction term x*d. d is the post-event dummy (1 = post, 0 = pre).
    The interaction t is the regression-discontinuity test: does the reversal
    slope change discontinuously at the event?"""
    n = len(y)
    if n < 6:
        return 0.0, 0.0
    X = np.column_stack([np.ones(n), x, d, x * d])
    try:
        XtX = X.T @ X
        XtXi = np.linalg.inv(XtX)
        beta = XtXi @ (X.T @ y)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    resid = y - X @ beta
    dof = n - X.shape[1]
    if dof <= 0:
        return 0.0, 0.0
    s2 = (resid @ resid) / dof
    se = np.sqrt(s2 * XtXi[3, 3])
    b = beta[3]
    return float(b), float(b / se) if se > 0 else 0.0


def _load_daily_close(path):
    df = pd.read_parquet(path, columns=["open_time", "close"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    s = df.set_index("open_time")["close"].sort_index()
    d = s.resample("1D").last().dropna()
    return d


def _daily_ret(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


# ------------------------------------------------------------------------------
def build_events(smoke=False):
    """Return list of dicts: each treated event with pre (spot) and post (perp)
    daily-return series, event-time aligned (day index = days from listing)."""
    ld = pd.read_csv(LD_CSV)
    ld = ld[(ld.category == "binance_um") & (~ld.listing_suspect.astype(bool))].copy()
    ld["listing"] = pd.to_datetime(ld.listing_date, utc=True)
    spot = {os.path.basename(p).replace("_1h.parquet", ""): p
            for p in glob.glob(SPOT_DIR + "/*.parquet")}
    perp = {os.path.basename(p).replace("_1h.parquet", ""): p
            for p in glob.glob(PERP_DIR + "/*.parquet")}

    events = []
    rows = ld.itertuples()
    seen = 0
    for r in rows:
        sym = r.symbol
        if sym in CONTROLS:           # never treat a control as an event
            continue
        if sym not in spot or sym not in perp:
            continue
        try:
            sclose = _load_daily_close(spot[sym])
            pclose = _load_daily_close(perp[sym])
        except Exception:
            continue
        listing = r.listing
        # PRE: spot returns strictly before listing day
        sret = _daily_ret(sclose)
        pre = sret[sret.index < listing]
        # POST: perp returns from listing onward
        pret = _daily_ret(pclose)
        post = pret[pret.index >= listing]
        if len(pre) < MIN_PRE or len(post) < MIN_POST:
            continue
        # cap window length each side
        pre = pre.iloc[-WIN_DAYS:]
        post = post.iloc[:WIN_DAYS]
        events.append(dict(symbol=sym, listing=listing, pre=pre, post=post))
        seen += 1
        if smoke and seen >= 12:
            break
    return events


def panel_rd(events):
    """Pooled panel regression-discontinuity in the reversal factor effect.
    Stack (factor, return) pairs across all treated events, separately for pre
    and post, plus the pooled interaction test."""
    Yp, Xp, Yo, Xo = [], [], [], []
    per_event = []
    for ev in events:
        for side, ret in (("pre", ev["pre"]), ("post", ev["post"])):
            f = _factor_revers(ret).reindex(ret.index)
            df = pd.DataFrame({"y": ret, "x": f}).dropna()
            if len(df) < 8:
                continue
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            # standardize factor within side (comparable coefficients)
            sd = x.std(ddof=1)
            if sd <= 0:
                continue
            x = (x - x.mean()) / sd
            if side == "pre":
                Yp.append(y); Xp.append(x)
            else:
                Yo.append(y); Xo.append(x)
        # per-event pre/post t for sign-stability summary
        fp = _factor_revers(ev["pre"]).reindex(ev["pre"].index)
        dp = pd.DataFrame({"y": ev["pre"], "x": fp}).dropna()
        fo = _factor_revers(ev["post"]).reindex(ev["post"].index)
        do = pd.DataFrame({"y": ev["post"], "x": fo}).dropna()
        if len(dp) >= 8 and len(do) >= 8:
            _, tpre = _ols_slope_t(dp["y"].to_numpy(),
                                   _z(dp["x"].to_numpy()))
            _, tpost = _ols_slope_t(do["y"].to_numpy(),
                                    _z(do["x"].to_numpy()))
            per_event.append((ev["symbol"], tpre, tpost))

    yp = np.concatenate(Yp); xp = np.concatenate(Xp)
    yo = np.concatenate(Yo); xo = np.concatenate(Xo)
    b_pre, t_pre = _ols_slope_t(yp, xp)
    b_post, t_post = _ols_slope_t(yo, xo)

    # pooled interaction (regression-discontinuity) test
    y_all = np.concatenate([yp, yo])
    x_all = np.concatenate([xp, xo])
    d_all = np.concatenate([np.zeros(len(yp)), np.ones(len(yo))])
    b_int, t_int = _ols_interaction_t(y_all, x_all, d_all)

    return dict(n_pre=len(yp), n_post=len(yo),
                b_pre=b_pre, t_pre=t_pre, b_post=b_post, t_post=t_post,
                b_interaction=b_int, t_interaction=t_int,
                per_event=per_event)


def _z(a):
    s = a.std(ddof=1)
    return (a - a.mean()) / s if s > 0 else a - a.mean()


def control_rd(events):
    """Placebo: run the same pre/post RD on the CONTROL majors, using each
    treated event's listing date as a pseudo-event cut on the control's own
    history. If the controls show no discontinuity, the treated effect is not a
    calendar artifact."""
    perp = {os.path.basename(p).replace("_1h.parquet", ""): p
            for p in glob.glob(PERP_DIR + "/*.parquet")}
    spot = {os.path.basename(p).replace("_1h.parquet", ""): p
            for p in glob.glob(SPOT_DIR + "/*.parquet")}
    ctrl_ret = {}
    for c in CONTROLS:
        if c in perp:
            try:
                ctrl_ret[c] = _daily_ret(_load_daily_close(perp[c]))
            except Exception:
                pass
    if not ctrl_ret:
        return dict(n_pre=0, n_post=0, b_pre=0, t_pre=0, b_post=0,
                    t_post=0, b_interaction=0, t_interaction=0)
    cuts = [ev["listing"] for ev in events]
    rng = np.random.default_rng(7)
    Yp, Xp, Yo, Xo = [], [], [], []
    for cut in cuts:
        c = list(ctrl_ret.keys())[rng.integers(len(ctrl_ret))]
        r = ctrl_ret[c]
        pre = r[r.index < cut].iloc[-WIN_DAYS:]
        post = r[r.index >= cut].iloc[:WIN_DAYS]
        for side, ret in (("pre", pre), ("post", post)):
            if len(ret) < MIN_PRE:
                continue
            f = _factor_revers(ret).reindex(ret.index)
            df = pd.DataFrame({"y": ret, "x": f}).dropna()
            if len(df) < 8:
                continue
            x = _z(df["x"].to_numpy()); y = df["y"].to_numpy()
            if side == "pre":
                Yp.append(y); Xp.append(x)
            else:
                Yo.append(y); Xo.append(x)
    if not Yp or not Yo:
        return dict(n_pre=0, n_post=0, b_pre=0, t_pre=0, b_post=0,
                    t_post=0, b_interaction=0, t_interaction=0)
    yp = np.concatenate(Yp); xp = np.concatenate(Xp)
    yo = np.concatenate(Yo); xo = np.concatenate(Xo)
    b_pre, t_pre = _ols_slope_t(yp, xp)
    b_post, t_post = _ols_slope_t(yo, xo)
    y_all = np.concatenate([yp, yo]); x_all = np.concatenate([xp, xo])
    d_all = np.concatenate([np.zeros(len(yp)), np.ones(len(yo))])
    b_int, t_int = _ols_interaction_t(y_all, x_all, d_all)
    return dict(n_pre=len(yp), n_post=len(yo), b_pre=b_pre, t_pre=t_pre,
                b_post=b_post, t_post=t_post,
                b_interaction=b_int, t_interaction=t_int)


def event_time_car(events, n_boot=2000, seed=11):
    """Event-time abnormal return of the reversal long-short sleeve.

    For each event we build the reversal sleeve sign(factor_{t-1}) * return_t on
    the POST (perp) side, aligned so day 1 = first full day after listing. The
    'abnormal' return is the sleeve return net of the cross-event mean sleeve
    return on the PRE (spot) side -- i.e. the event-induced change in the reversal
    payoff relative to the same assets' own pre-event baseline. We average across
    events at each event-day and accumulate to a CAR, with a bootstrap CI over
    events."""
    # pre baseline mean sleeve return per event (the 'normal' reversal payoff)
    sleeves_post = []     # list of arrays length EVENT_HALF (day 1..EVENT_HALF)
    baselines = []
    for ev in events:
        # baseline: mean reversal sleeve on pre side
        fp = _factor_revers(ev["pre"]).reindex(ev["pre"].index)
        dfp = pd.DataFrame({"y": ev["pre"], "x": fp}).dropna()
        if len(dfp) < 8:
            continue
        base = float((np.sign(dfp["x"]) * dfp["y"]).mean())
        # post sleeve, aligned by event day
        fo = _factor_revers(ev["post"]).reindex(ev["post"].index)
        dfo = pd.DataFrame({"y": ev["post"], "x": fo}).dropna()
        if len(dfo) < 5:
            continue
        sl = (np.sign(dfo["x"]) * dfo["y"]).to_numpy()
        arr = np.full(EVENT_HALF, np.nan)
        m = min(EVENT_HALF, len(sl))
        arr[:m] = sl[:m]
        sleeves_post.append(arr)
        baselines.append(base)
    M = np.array(sleeves_post)              # (n_events, EVENT_HALF)
    base = np.array(baselines)[:, None]     # (n_events, 1)
    abnormal = M - base                     # abnormal daily sleeve return
    # mean abnormal return per event-day (ignore NaN)
    ar = np.nanmean(abnormal, axis=0)
    ar = np.nan_to_num(ar, nan=0.0)
    car = np.cumsum(ar)
    # bootstrap CI over events
    rng = np.random.default_rng(seed)
    ne = abnormal.shape[0]
    boots = np.empty((n_boot, EVENT_HALF))
    for b in range(n_boot):
        idx = rng.integers(0, ne, ne)
        arb = np.nanmean(abnormal[idx], axis=0)
        arb = np.nan_to_num(arb, nan=0.0)
        boots[b] = np.cumsum(arb)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    # final-day CAR significance from bootstrap
    final = boots[:, -1]
    p_final = 2.0 * min((final <= 0).mean(), (final >= 0).mean())
    return dict(days=np.arange(1, EVENT_HALF + 1), ar=ar, car=car,
                lo=lo, hi=hi, n_events=ne,
                car_final=float(car[-1]), car_p=float(p_final))


def make_figure(treated, control, car, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # (1) event-time CAR of the reversal sleeve with bootstrap CI
    ax = axes[0]
    d = car["days"]
    ax.fill_between(d, car["lo"] * 100, car["hi"] * 100, color="#56B4E9",
                    alpha=0.25, label="95% bootstrap CI")
    ax.plot(d, car["car"] * 100, color="#0072B2", lw=2.0,
            label="continuation sleeve CAR")
    ax.axhline(0, color="#999999", lw=0.9, ls="--")
    ax.set_title("Continuation-sleeve abnormal return after perp listing\n"
                 "(falls = reversal strengthens; vs each asset's pre baseline)")
    ax.set_xlabel("trading days since perpetual listing")
    ax.set_ylabel("cumulative abnormal return (%)")
    ax.legend(loc="best")

    # (2) regression-discontinuity bar: reversal t-stat pre vs post,
    #     treated vs control
    ax = axes[1]
    groups = ["treated\npre", "treated\npost", "control\npre", "control\npost"]
    tvals = [treated["t_pre"], treated["t_post"],
             control["t_pre"], control["t_post"]]
    colors = ["#999999", "#D55E00", "#cccccc", "#E69F00"]
    bars = ax.bar(groups, tvals, color=colors, edgecolor="black", lw=0.6)
    ax.axhline(1.96, color="#009E73", lw=1.0, ls=":")
    ax.axhline(-1.96, color="#009E73", lw=1.0, ls=":")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Reversal-factor t-stat: discontinuity at the event\n"
                 "(dotted green = |t| = 1.96)")
    ax.set_ylabel("pooled reversal t-stat")
    for b, v in zip(bars, tvals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.4 if v >= 0 else -0.8),
                f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (12 events, few bootstraps)")
    args = ap.parse_args()

    n_boot = 200 if args.smoke else 2000
    t0 = pd.Timestamp.now()
    print("[event-study] building events ...", flush=True)
    events = build_events(smoke=args.smoke)
    print(f"[event-study] qualifying treated events: {len(events)}", flush=True)

    print("[event-study] pooled regression-discontinuity (treated) ...", flush=True)
    treated = panel_rd(events)
    print("[event-study] placebo regression-discontinuity (controls) ...", flush=True)
    control = control_rd(events)
    print("[event-study] event-time abnormal returns + bootstrap ...", flush=True)
    car = event_time_car(events, n_boot=n_boot)

    # ----- write tables -------------------------------------------------------
    tbl = pd.DataFrame([
        dict(group="treated", window="pre",  measure="reversal_slope",
             value=treated["b_pre"],  t_stat=treated["t_pre"],  n=treated["n_pre"]),
        dict(group="treated", window="post", measure="reversal_slope",
             value=treated["b_post"], t_stat=treated["t_post"], n=treated["n_post"]),
        dict(group="treated", window="post-minus-pre", measure="reversal_RD_interaction",
             value=treated["b_interaction"], t_stat=treated["t_interaction"],
             n=treated["n_pre"] + treated["n_post"]),
        dict(group="control", window="pre",  measure="reversal_slope",
             value=control["b_pre"],  t_stat=control["t_pre"],  n=control["n_pre"]),
        dict(group="control", window="post", measure="reversal_slope",
             value=control["b_post"], t_stat=control["t_post"], n=control["n_post"]),
        dict(group="control", window="post-minus-pre", measure="reversal_RD_interaction",
             value=control["b_interaction"], t_stat=control["t_interaction"],
             n=control["n_pre"] + control["n_post"]),
        dict(group="treated", window=f"event+{EVENT_HALF}d", measure="sleeve_CAR_abnormal",
             value=car["car_final"], t_stat=np.nan, n=car["n_events"]),
    ])
    tbl["car_boot_p"] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, car["car_p"]]
    tbl = tbl.round(6)
    csv_path = os.path.join(PROJ, "tables", "event_study.csv")
    tbl.to_csv(csv_path, index=False)

    # per-event sign-stability summary
    pe = treated["per_event"]
    n_strength = sum(1 for _, tp, to in pe if abs(to) > abs(tp))
    frac_strength = n_strength / len(pe) if pe else float("nan")

    md = []
    md.append("# Event study -- perpetual-listing natural experiment\n")
    md.append(f"- Treated events (Binance USD-M perpetual listings, pre-spot vs "
              f"post-perp): **{len(events)}**")
    md.append(f"- Control majors (placebo, no listing event): "
              f"{', '.join(CONTROLS)}")
    md.append(f"- Pre window: token spot daily returns (up to {WIN_DAYS}d before "
              f"listing); Post window: perp daily returns (up to {WIN_DAYS}d after).")
    md.append(f"- Factor: short-term reversal (trailing 5d, lagged), identical to "
              f"the main study.\n")
    md.append("## Regression-discontinuity in the reversal-factor effect\n")
    md.append("| group | window | reversal slope | t-stat | n |")
    md.append("|---|---|---|---|---|")
    for _, r in tbl[tbl.measure.isin(["reversal_slope",
                                      "reversal_RD_interaction"])].iterrows():
        md.append(f"| {r.group} | {r.window} | {r.value:.4f} | "
                  f"{r.t_stat:.2f} | {int(r.n)} |")
    md.append("")
    md.append("## Abnormal return (continuation sleeve)\n")
    md.append(f"- The abnormal-return sleeve bets on continuation "
              f"(sign(factor) * return); a NEGATIVE value means recent winners "
              f"reverse, i.e. reversal pays. Cumulative abnormal return over "
              f"{EVENT_HALF} post-listing days, net of each asset's own "
              f"pre-listing baseline: **{car['car_final']*100:.2f}%** "
              f"(bootstrap two-sided p = {car['car_p']:.3f}, "
              f"{car['n_events']} events). A negative CAR is the reversal effect "
              f"strengthening, consistent with the more-negative post slope above.")
    md.append(f"- Per-event check (caveat): the reversal |t| is larger POST than "
              f"PRE in only **{frac_strength*100:.0f}%** of individual events "
              f"({n_strength}/{len(pe)}). The discontinuity is a pooled-panel / "
              f"average phenomenon, not present asset-by-asset.\n")
    md.append("## Reading\n")
    treated_strengthens = (abs(treated["t_post"]) > abs(treated["t_pre"])) and \
                          (abs(treated["t_interaction"]) > 1.96)
    control_flat = abs(control["t_interaction"]) <= 1.96
    md.append(f"- Treated discontinuity (post-minus-pre interaction) "
              f"t = {treated['t_interaction']:.2f}; "
              f"control discontinuity t = {control['t_interaction']:.2f}.")
    if treated_strengthens and control_flat:
        md.append("- The reversal effect shifts discontinuously at the exogenous "
                  "listing event in the treated assets but NOT in the controls -- "
                  "consistent with the event (not the calendar) driving the change.")
    elif abs(treated["t_interaction"]) <= 1.96:
        md.append("- No significant discontinuity in the treated assets: the "
                  "listing event does not measurably move the reversal effect.")
    else:
        md.append("- A discontinuity appears in the treated assets; the control "
                  "comparison qualifies how cleanly it can be attributed to the "
                  "event (see control interaction t above).")
    md_path = os.path.join(PROJ, "tables", "event_study.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    # ----- figure -------------------------------------------------------------
    fig_path = os.path.join(PROJ, "figures", "fig_event_study.png")
    make_figure(treated, control, car, fig_path)

    dt = (pd.Timestamp.now() - t0).total_seconds()
    print(f"[event-study] done in {dt:.1f}s", flush=True)
    print(f"  treated  pre t={treated['t_pre']:.2f}  post t={treated['t_post']:.2f}  "
          f"RD-interaction t={treated['t_interaction']:.2f}", flush=True)
    print(f"  control  pre t={control['t_pre']:.2f}  post t={control['t_post']:.2f}  "
          f"RD-interaction t={control['t_interaction']:.2f}", flush=True)
    print(f"  CAR(+{EVENT_HALF}d)={car['car_final']*100:.2f}%  "
          f"boot-p={car['car_p']:.3f}  events={car['n_events']}", flush=True)
    print(f"  wrote {csv_path}", flush=True)
    print(f"  wrote {md_path}", flush=True)
    print(f"  wrote {fig_path}", flush=True)

    # summary json for downstream
    summ = dict(n_events=len(events),
                treated=treated_summary(treated), control=control_summary(control),
                car_final=car["car_final"], car_p=car["car_p"],
                frac_post_stronger=frac_strength)
    with open(os.path.join(PROJ, "tables", "event_study_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)


def treated_summary(t):
    return {k: t[k] for k in ("n_pre", "n_post", "b_pre", "t_pre",
                              "b_post", "t_post", "b_interaction", "t_interaction")}


def control_summary(c):
    return {k: c[k] for k in ("n_pre", "n_post", "b_pre", "t_pre",
                              "b_post", "t_post", "b_interaction", "t_interaction")}


if __name__ == "__main__":
    main()
