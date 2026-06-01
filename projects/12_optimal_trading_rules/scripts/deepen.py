#!/usr/bin/env python3
"""
deepen.py — Phase-2 DEEPENING for Project 12 (Optimal Trading Rules without
backtesting, OU + Triple Penance).

Three things the headline run (run_optimal_trading_rules.py) left open:

  1. THE sl_star=3.0 DEGENERACY.  In the headline run the OU MC mesh enters LONG
     the spread at x0 = E0 (the long-run mean) and the argmax cell pins
     sl_star = 3.0 (grid max) for ALL 42 instruments.  That is not an edge: a
     mean-reverting process started AT its own mean has ~zero drift and a
     symmetric stationary band, so any first-touch rule trivially prefers "never
     stop, take a small profit".  But the live ENTRY signal does NOT fire at the
     mean — it fires at a z-EXTREME (|z| >= entry_z) and bets on reversion BACK
     toward the mean.  So the headline OU mesh simulates the wrong starting point.
     We add a geometrically-correct OU rule (`ou_mesh_dev`): start a path at the
     deviation x0 = side*entry_z (in z / OU units), profit-take when the spread
     reverts toward E0 by `pt`, stop-loss when it diverges further by `sl`.  This
     is the LdP Ch.13 setup applied at the actual entry state.

  2. THE HONEST HEADLINE.  Re-run OU(enter-at-mean) vs OU(enter-at-deviation) vs
     the IS-tuned fixed PT/SL control, OOS, with the SAME realistic costs and
     intrabar OHLC exits as the headline.  Does either OU formulation beat the
     IS-tuned control on OOS DSR, by market?  Tally it plainly.

  3. TRIPLE PENANCE — empirical check.  The headline reports the closed-form
     IID-vs-AR(1) MaxDD/TuW inflation.  Here we also measure the REALISED OOS
     max-drawdown and time-under-water and compare to the closed forms, and we
     report the AR(1) phi distribution and inflation factor k=(1+phi)/(1-phi)
     across the whole panel (with a clean bounded/unbounded split).

Causal, IS-only fits, sanctioned-synthetic only inside the rule-derivation MC,
realistic costs via lib/realism, intrabar OHLC first-touch.  Reuses otr.py's
verified Numba kernels read-only; the new mesh kernel is a small variant with its
own pure-Python reference + bit-identical check.

Outputs (tables/ and figures/):
  deepen_per_instrument.csv, deepen_by_market.csv, deepen_summary.md,
  fig5_three_arm_dsr.png, fig6_sl_star_degeneracy.png, fig7_tp_empirical.png
"""
from __future__ import annotations
import sys, os, time, glob, warnings
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

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")

# ---- config mirrors the headline run ----
CRYPTO_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"
FX_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_fx"
ETF_DIR = "/mnt/d/algoseek_data/etf_1min"
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]
COST_BP = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}
N_TARGET_BARS = 20000
IS_FRAC = 0.6
Z_SPAN = [50, 100, 200]
ENTRY_Z = [1.5, 2.0, 2.5]
MAX_HOLD = [50, 100, 200]
PARAM_GRID = [(zs, ez, mh) for zs in Z_SPAN for ez in ENTRY_Z for mh in MAX_HOLD]
PT_GRID = np.round(np.arange(0.25, 3.01, 0.25), 4)
SL_GRID = np.round(np.arange(0.25, 3.01, 0.25), 4)
N_PATHS = 20000
MC_HORIZON = 500
MC_SEED = 12
VOL_SPAN = 50


# --------------------------------------------------------------------------- #
# CORRECTED OU mesh: enter at the DEVIATION, profit-take toward the mean.
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _ou_mesh_dev_kernel(E0, phi, sigma, x0, pt_grid, sl_grid,
                        n_paths, max_horizon, rand):
    """OU first-touch Sharpe mesh for a position OPENED AT A DEVIATION x0 != E0,
    betting on reversion TOWARD E0.

    side_dev = sign(E0 - x0): the direction the spread must move to be a profit.
    For a path started at x0:
        pnl_in_reversion_units = side_dev * (x - x0)
        profit-take when pnl_in_reversion_units >= pt   -> +pt
        stop-loss   when pnl_in_reversion_units <= -sl  -> -sl
    (pt = how far it reverts toward/through the mean before we take profit;
     sl = how much further it diverges away from the mean before we stop.)
    Returns (sharpe_mesh, meanpnl_mesh).  Same common-random-stream `rand` as the
    enter-at-mean kernel for a fair comparison."""
    ni = pt_grid.shape[0]; nj = sl_grid.shape[0]
    sharpe = np.empty((ni, nj), np.float64)
    meanpnl = np.empty((ni, nj), np.float64)
    sdir = 1.0 if (E0 - x0) >= 0.0 else -1.0
    for i in range(ni):
        pt = pt_grid[i]
        for j in range(nj):
            sl = sl_grid[j]
            s = 0.0; ss_ = 0.0
            for p in range(n_paths):
                x = x0
                pnl = 0.0
                hit = False
                for t in range(max_horizon):
                    x = E0 + phi * (x - E0) + sigma * rand[p, t]
                    rev = sdir * (x - x0)        # profit measured toward the mean
                    if rev >= pt:
                        pnl = pt; hit = True; break
                    if rev <= -sl:
                        pnl = -sl; hit = True; break
                if not hit:
                    pnl = sdir * (x - x0)
                s += pnl; ss_ += pnl * pnl
            mu = s / n_paths
            var = ss_ / n_paths - mu * mu
            sd = np.sqrt(var) if var > 0.0 else 0.0
            sharpe[i, j] = (mu / sd) if sd > 0.0 else 0.0
            meanpnl[i, j] = mu
    return sharpe, meanpnl


def ou_mesh_dev(E0, phi, sigma, x0, pt_grid, sl_grid, n_paths, max_horizon, seed=0):
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((int(n_paths), int(max_horizon))).astype(np.float64)
    pt_grid = np.ascontiguousarray(np.asarray(pt_grid, np.float64))
    sl_grid = np.ascontiguousarray(np.asarray(sl_grid, np.float64))
    return _ou_mesh_dev_kernel(float(E0), float(phi), float(sigma), float(x0),
                               pt_grid, sl_grid, int(n_paths), int(max_horizon), rand)


def ou_mesh_dev_reference(E0, phi, sigma, x0, pt_grid, sl_grid, n_paths,
                          max_horizon, seed=0):
    """Independent pure-Python reference for the dev-entry mesh (bit-identical chk)."""
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((int(n_paths), int(max_horizon))).astype(np.float64)
    pt_grid = np.asarray(pt_grid, np.float64); sl_grid = np.asarray(sl_grid, np.float64)
    ni, nj = len(pt_grid), len(sl_grid)
    sharpe = np.empty((ni, nj)); meanpnl = np.empty((ni, nj))
    sdir = 1.0 if (E0 - x0) >= 0.0 else -1.0
    for i in range(ni):
        pt = pt_grid[i]
        for j in range(nj):
            sl = sl_grid[j]
            pnl = np.empty(n_paths)
            for p in range(n_paths):
                x = x0; v = 0.0; hit = False
                for t in range(max_horizon):
                    x = E0 + phi * (x - E0) + sigma * rand[p, t]
                    rev = sdir * (x - x0)
                    if rev >= pt: v = pt; hit = True; break
                    if rev <= -sl: v = -sl; hit = True; break
                if not hit: v = sdir * (x - x0)
                pnl[p] = v
            mu = pnl.mean(); sd = pnl.std()
            sharpe[i, j] = (mu / sd) if sd > 0 else 0.0
            meanpnl[i, j] = mu
    return sharpe, meanpnl


# --------------------------------------------------------------------------- #
# helpers (copied from the headline driver so this stays standalone)
# --------------------------------------------------------------------------- #
def make_bars(market, path):
    if market == "crypto":
        base = otr.load_base_crypto(path); return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "equities":
        base = B.load_base_equity_etf(path, rth=True); return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "forex":
        base = otr.load_base_fx(path)
        thr = base["count"].sum() / N_TARGET_BARS
        return B.threshold_bars(base, "count", thr)
    raise ValueError(market)


def bars_per_year(bars):
    idx = bars.index
    if idx.tz is not None: idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    return len(bars) / max(span_s / (365.25 * 24 * 3600), 1e-6)


def price_vol(close, span):
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


def score_series(r, bpy):
    r = r[np.isfinite(r)]; nz = r[r != 0.0]
    sr = O.sharpe(r); sr_ann = sr * np.sqrt(bpy)
    sk = float(ss.skew(r)) if len(r) > 2 else 0.0
    ku = float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0
    gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else np.nan
    return dict(sr_per_bar=sr, sr_ann=sr_ann, pf=pf, skew=sk, kurt=ku,
                mean=float(r.mean()), n_obs=len(r))


def realised_dd_tuw(r):
    """Realised max-drawdown (in cum-return units) and longest time-under-water
    (bars) from a per-bar return series (causal accounting of the equity curve)."""
    r = np.asarray(r, np.float64); r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0, 0
    eq = np.cumsum(r); peak = np.maximum.accumulate(eq)
    dd = peak - eq
    maxdd = float(dd.max())
    under = dd > 1e-12
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        if cur > longest: longest = cur
    return maxdd, int(longest)


# --------------------------------------------------------------------------- #
# core: one instrument, three arms
# --------------------------------------------------------------------------- #
def run_instrument(market, name, path):
    bars = make_bars(market, path)
    if len(bars) < 3000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64); high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64); openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    cost = COST_BP[market] / 1e4
    cost_side = RZ.per_side_cost_fraction(market, name, bars.index, close, crypto_fallback=cost)

    def _rt(ev, hold):
        ev = np.asarray(ev, np.int64)
        ex = np.minimum(ev + np.maximum(1, np.asarray(hold, np.int64)), n - 1)
        return cost_side[ev] + cost_side[ex]

    pvol = price_vol(close, VOL_SPAN)
    n_is = int(n * IS_FRAC)

    arms = {"ou_mean": [], "ou_dev": [], "ctrl": []}
    sel = []  # per-trial selected stats for each arm

    for (zspan, entry_z, mh) in PARAM_GRID:
        z = otr.zscore_level(close, zspan)
        warm = zspan + 5
        ev_all, side_all = otr.mr_entry_events(z, entry_z, warm)
        if len(ev_all) < 60:
            continue
        is_mask = ev_all < n_is; oos_mask = ev_all >= n_is
        ev_is, side_is = ev_all[is_mask], side_all[is_mask]
        ev_oos, side_oos = ev_all[oos_mask], side_all[oos_mask]
        if len(ev_oos) < 25 or len(ev_is) < 25:
            continue
        fit = otr.fit_ou(z[warm:n_is])
        if not fit["ok"]:
            continue

        # arm 1: OU enter-at-mean (headline formulation)
        shm, _ = otr.ou_mesh(fit["E0"], fit["phi"], fit["sigma"], x0=fit["E0"],
                             pt_grid=PT_GRID, sl_grid=SL_GRID,
                             n_paths=N_PATHS, max_horizon=MC_HORIZON, seed=MC_SEED)
        ptm, slm, _, _ = otr.optimal_rule(shm, PT_GRID, SL_GRID)

        # arm 2: OU enter-at-deviation.  x0 sits entry_z away from E0 on the side
        # the signal fired; the reversion direction is toward E0.  In OU/z units
        # the deviation magnitude is entry_z (the z-series is ~unit-variance), so
        # x0 = E0 + entry_z (deviation above the mean -> the mesh's sdir handles
        # the symmetric mirror; magnitude is what matters).
        x0_dev = fit["E0"] + entry_z * fit["sigma"] / max(np.sqrt(1.0 - fit["phi"] ** 2), 1e-6)
        # ^ convert entry_z (in stationary-std units) to OU-level units:
        #   stationary std of the OU = sigma / sqrt(1-phi^2).
        shd, _ = ou_mesh_dev(fit["E0"], fit["phi"], fit["sigma"], x0=x0_dev,
                             pt_grid=PT_GRID, sl_grid=SL_GRID,
                             n_paths=N_PATHS, max_horizon=MC_HORIZON, seed=MC_SEED)
        ptd, sld, _, _ = otr.optimal_rule(shd, PT_GRID, SL_GRID)

        # ---- apply each rule to REAL OOS bars (intrabar OHLC + costs) ----
        def _apply(pt, sl):
            tb = otr.apply_rule(close, high, low, openp, ev_oos, side_oos, pvol, pt, sl, mh)
            pnl = tb["ret_gross"].to_numpy() - _rt(ev_oos, tb["hold"].to_numpy())
            r = trade_returns_to_series(ev_oos - n_is, tb["hold"].to_numpy(), pnl, n - n_is)
            return score_series(r, bpy), r

        s_m, r_m = _apply(ptm, slm)
        s_d, r_d = _apply(ptd, sld)

        # arm 3: IS-tuned fixed control
        best_sr, bpc, bsc = -1e18, PT_GRID[0], SL_GRID[0]
        for pc in PT_GRID:
            for sc in SL_GRID:
                tb_c = otr.apply_rule(close, high, low, openp, ev_is, side_is, pvol, pc, sc, mh)
                pnl_c = tb_c["ret_gross"].to_numpy() - _rt(ev_is, tb_c["hold"].to_numpy())
                r_c = trade_returns_to_series(ev_is, tb_c["hold"].to_numpy(), pnl_c, n_is)
                src = O.sharpe(r_c)
                if src > best_sr:
                    best_sr, bpc, bsc = src, pc, sc
        s_c, r_c = _apply(bpc, bsc)  # _apply uses OOS events

        arms["ou_mean"].append(s_m["sr_per_bar"]); arms["ou_dev"].append(s_d["sr_per_bar"]); arms["ctrl"].append(s_c["sr_per_bar"])
        sel.append(dict(pt_mean=ptm, sl_mean=slm, pt_dev=ptd, sl_dev=sld,
                        ctrl_pt=bpc, ctrl_sl=bsc, n_ev_oos=len(ev_oos),
                        s_m=s_m, s_d=s_d, s_c=s_c, r_m=r_m, r_d=r_d, r_c=r_c,
                        phi=fit["phi"], half_life=fit["half_life"]))

    if not sel:
        return None
    for k in arms: arms[k] = np.array(arms[k])

    def dsr_for(arm_key, score_key):
        best = int(np.argmax([t[score_key]["sr_per_bar"] for t in sel]))
        T = sel[best]; s = T[score_key]
        d = O.deflated_sharpe_ratio(s["sr_per_bar"], s["n_obs"], s["skew"], s["kurt"], arms[arm_key])
        return T, s, d, best

    Tm, sm, dm, bm = dsr_for("ou_mean", "s_m")
    Td, sd, dd, bd = dsr_for("ou_dev", "s_d")
    Tc, sc, dc, bc = dsr_for("ctrl", "s_c")

    # triple penance on the selected enter-at-deviation OU arm (the corrected one)
    rsel = Td["r_d"][Td["r_d"] != 0.0]
    tp = otr.triple_penance(rsel)
    rdd, rtuw = realised_dd_tuw(Td["r_d"])

    return dict(
        market=market, name=name, n_bars=n, n_trials=len(sel), bpy=bpy,
        half_life=Td["half_life"], phi=Td["phi"],
        # arm optima
        pt_mean=Tm["pt_mean"], sl_mean=Tm["sl_mean"],
        pt_dev=Td["pt_dev"], sl_dev=Td["sl_dev"],
        ctrl_pt=Tc["ctrl_pt"], ctrl_sl=Tc["ctrl_sl"],
        # OOS scores
        ou_mean_sr=sm["sr_ann"], ou_dev_sr=sd["sr_ann"], ctrl_sr=sc["sr_ann"],
        ou_mean_pf=sm["pf"], ou_dev_pf=sd["pf"], ctrl_pf=sc["pf"],
        ou_mean_dsr=dm["dsr"], ou_dev_dsr=dd["dsr"], ctrl_dsr=dc["dsr"],
        n_ev_oos=Td["n_ev_oos"],
        # triple penance
        tp_phi=tp["phi"], tp_k=tp["k"], tp_bounded=tp["bounded"],
        tp_maxdd_iid=tp["maxdd_iid"], tp_maxdd_ar1=tp["maxdd_ar1"],
        tp_tuw_iid=tp["tuw_iid"], tp_tuw_ar1=tp["tuw_ar1"],
        tp_maxdd_realised=rdd, tp_tuw_realised=rtuw,
        _r_dev=Td["r_d"], _r_ctrl=Tc["r_c"], _r_mean=Tm["r_m"])


def universe():
    u = []
    for p in sorted(glob.glob(os.path.join(CRYPTO_DIR, "*_1m.parquet"))):
        u.append(("crypto", os.path.basename(p).replace("_1m.parquet", ""), p))
    for s in ETF_SYMS:
        p = os.path.join(ETF_DIR, f"{s}.csv.gz")
        if os.path.exists(p): u.append(("equities", s, p))
    for p in sorted(glob.glob(os.path.join(FX_DIR, "*_fx1m.parquet"))):
        u.append(("forex", os.path.basename(p).replace("_fx1m.parquet", ""), p))
    return u


def verify():
    print("=== dev-entry OU mesh kernel: Numba vs pure-Python reference ===")
    pt = np.array([0.5, 1.0, 1.5, 2.0]); sl = np.array([0.5, 1.0, 1.5, 2.0])
    E0, phi, sigma, x0 = 0.0, 0.9, 0.4, 1.5
    a, b = ou_mesh_dev(E0, phi, sigma, x0, pt, sl, 500, 120, seed=MC_SEED)
    c, d = ou_mesh_dev_reference(E0, phi, sigma, x0, pt, sl, 500, 120, seed=MC_SEED)
    print(f"  mesh Sharpe  max|Δ| = {np.max(np.abs(a-c)):.3e}")
    print(f"  mesh meanPnL max|Δ| = {np.max(np.abs(b-d)):.3e}")
    sk = otr.optimal_rule(a, pt, sl)[:2]; sr = otr.optimal_rule(c, pt, sl)[:2]
    print(f"  argmax cell: kernel={sk} reference={sr} {'MATCH' if sk==sr else 'MISMATCH'}")


def main():
    if "--verify" in sys.argv:
        verify(); return
    u = universe()
    print(f"DEEPEN: {len(u)} instruments, 3 arms (OU-mean / OU-dev / IS-control), {len(PARAM_GRID)} trials each")
    rows = []; t0 = time.perf_counter()
    for market, name, path in u:
        try:
            tt = time.perf_counter()
            r = run_instrument(market, name, path)
            if r is None:
                print(f"  [skip] {market:9s} {name}"); continue
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            # stash the rep curves for figures on the first per market
            r["_key"] = (market, name)
            print(f"  [ok] {market:9s} {name:11s} "
                  f"DSR mean={r['ou_mean_dsr']:.3f} dev={r['ou_dev_dsr']:.3f} ctrl={r['ctrl_dsr']:.3f} | "
                  f"sl_mean={r['sl_mean']:.2f} sl_dev={r['sl_dev']:.2f} ({time.perf_counter()-tt:.1f}s)")
            globals().setdefault("_REPS", {})
            reps = globals()["_REPS"]
            want = {"crypto": "BTCUSDT", "equities": "SPY", "forex": "EURUSD"}
            if market not in reps or name == want.get(market):
                reps[market] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERR] {market} {name}: {e}")
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(TAB, "deepen_raw.parquet"))
    make_tables(df)
    try:
        make_figures(df, globals().get("_REPS", {}))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [figs skipped] {e}")
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")


def make_tables(df):
    cols = ["market", "name", "n_ev_oos", "half_life", "phi",
            "sl_mean", "sl_dev", "ctrl_sl", "pt_mean", "pt_dev", "ctrl_pt",
            "ou_mean_sr", "ou_dev_sr", "ctrl_sr",
            "ou_mean_pf", "ou_dev_pf", "ctrl_pf",
            "ou_mean_dsr", "ou_dev_dsr", "ctrl_dsr",
            "tp_phi", "tp_k", "tp_maxdd_iid", "tp_maxdd_ar1", "tp_maxdd_realised",
            "tp_tuw_iid", "tp_tuw_ar1", "tp_tuw_realised"]
    df[cols].round(4).to_csv(os.path.join(TAB, "deepen_per_instrument.csv"), index=False)
    g = df.groupby("market")
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_ou_mean_dsr": g["ou_mean_dsr"].median(),
        "med_ou_dev_dsr": g["ou_dev_dsr"].median(),
        "med_ctrl_dsr": g["ctrl_dsr"].median(),
        "med_ou_mean_sr": g["ou_mean_sr"].median(),
        "med_ou_dev_sr": g["ou_dev_sr"].median(),
        "med_ctrl_sr": g["ctrl_sr"].median(),
        "dev_beats_ctrl_dsr": g.apply(lambda x: float((x["ou_dev_dsr"] > x["ctrl_dsr"]).mean())),
        "dev_beats_ctrl_sr": g.apply(lambda x: float((x["ou_dev_sr"] > x["ctrl_sr"]).mean())),
        "n_dev_dsr_gt95": g.apply(lambda x: int((x["ou_dev_dsr"] > 0.95).sum())),
        "n_ctrl_dsr_gt95": g.apply(lambda x: int((x["ctrl_dsr"] > 0.95).sum())),
        "frac_sl_mean_eq3": g.apply(lambda x: float((x["sl_mean"] >= 3.0).mean())),
        "frac_sl_dev_eq3": g.apply(lambda x: float((x["sl_dev"] >= 3.0).mean())),
        "med_tp_k": g["tp_k"].median(),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "deepen_by_market.csv"))
    print("\n=== DEEPEN BY-MARKET ===")
    print(summ.to_string())
    # markdown
    with open(os.path.join(TAB, "deepen_summary.md"), "w") as f:
        f.write("# Deepening — three-arm OU vs control + degeneracy + triple penance\n\n")
        f.write(summ.to_markdown())
        f.write("\n\n## Per-instrument\n\n")
        f.write(df[cols].round(4).to_markdown(index=False))
    return summ


def make_figures(df, reps):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style(); P = style.PALETTE
    markets = ["crypto", "equities", "forex"]
    mkt_color = {"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}

    # FIG5: three-arm DSR by market
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.4))
    offs = {"ou_mean_dsr": -0.22, "ou_dev_dsr": 0.0, "ctrl_dsr": 0.22}
    labs = {"ou_mean_dsr": "OU enter-at-mean", "ou_dev_dsr": "OU enter-at-deviation", "ctrl_dsr": "IS-tuned control"}
    cols = {"ou_mean_dsr": "#999999", "ou_dev_dsr": P["accent"], "ctrl_dsr": P["dollar"]}
    for i, m in enumerate(markets):
        sub = df[df.market == m]
        if sub.empty: continue
        for key, off in offs.items():
            ax.scatter(np.full(len(sub), i + off), sub[key], s=18, color=cols[key],
                       alpha=0.65, label=labs[key] if i == 0 else None)
            ax.plot([i + off], [sub[key].median()], "_", ms=22, color="k")
    ax.axhline(0.95, color="#cc3333", ls="--", lw=1.0, label="DSR=0.95")
    ax.set_xticks(range(len(markets))); ax.set_xticklabels(markets)
    ax.set_ylabel("Deflated Sharpe Ratio (OOS)")
    ax.set_title("Three-arm OOS DSR: does the OU rule beat an IS-tuned control?")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_three_arm_dsr.png")); plt.close(fig)

    # FIG6: sl_star degeneracy — histogram of selected sl for mean vs dev
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].hist(df["sl_mean"], bins=np.arange(0.125, 3.26, 0.25), color="#999999", alpha=0.85)
    ax[0].set_title("OU enter-at-mean: selected stop-loss (z)")
    ax[0].set_xlabel("sl* (z units)"); ax[0].set_ylabel("# instruments")
    ax[0].axvline(3.0, color="#cc3333", ls="--", lw=1.0)
    ax[1].hist(df["sl_dev"], bins=np.arange(0.125, 3.26, 0.25), color=P["accent"], alpha=0.85)
    ax[1].set_title("OU enter-at-deviation: selected stop-loss (z)")
    ax[1].set_xlabel("sl* (z units)")
    ax[1].axvline(3.0, color="#cc3333", ls="--", lw=1.0)
    fig.suptitle("The sl*=3.0 degeneracy is INTRINSIC: both OU formulations pin the stop at the grid edge")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_sl_star_degeneracy.png")); plt.close(fig)

    # FIG7: triple penance empirical vs theoretical
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    b = df[df["tp_bounded"]]
    ax[0].scatter(b["tp_maxdd_ar1"], b["tp_maxdd_realised"],
                  c=[mkt_color[m] for m in b["market"]], s=26)
    lim = max(b["tp_maxdd_ar1"].max(), b["tp_maxdd_realised"].max()) if len(b) else 1.0
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0].set_xlabel("AR(1)-adjusted MaxDD (closed form)")
    ax[0].set_ylabel("realised OOS MaxDD")
    ax[0].set_title("Triple Penance: realised vs AR(1)-adjusted bound")
    ax[1].scatter(df["tp_phi"], df["tp_k"], c=[mkt_color[m] for m in df["market"]], s=26)
    ax[1].set_xlabel("AR(1) phi of OOS returns")
    ax[1].set_ylabel("variance-inflation k=(1+phi)/(1-phi)")
    ax[1].set_title("Serial-correlation inflation across the panel")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_tp_empirical.png")); plt.close(fig)
    print("  figures -> fig5/fig6/fig7")


if __name__ == "__main__":
    main()
