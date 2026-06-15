"""T_XS walk-forward — the cross-sectional WFO loop.

Per window [ws,ie) IS / [ie,oe) OOS (disjoint forward):
  1. Build cross-sectional features (done once, cached at panel level) and the
     forward cross-sectional return-RANK label, PURGED so a training row's label
     horizon H does not cross the IS boundary (lbl_end < boundary).
  2. Split IS into fit [ws,sel_start) and selection [sel_start,ie). Train the model
     (one hparam draw, NS_MODEL=1) on the purged fit rows; score the selection slice.
  3. Fan out NS_KNOB portfolio-knob configs over the cached score via the BATCHED sim
     (q / long_only / tilt / reb). Keep the knob with best IS-selection Sharpe.
  4. Refit on full purged IS; score IS+OOS; simulate the winning knob → IS pnl (phase 0)
     and OOS pnl (phase 1). The OOS path is what we report; per-bar pnl is the ledger.

Label: forward H-bar return, ranked cross-sectionally at the ENTRY bar (so the model
learns relative winners). lbl_end[t] = t+H (purge horizon). Causal: features are shift1,
label uses only [t, t+H] returns and is never used past its boundary in training.

A "combo" here = (family, feature variant, H). Knobs are IS-tuned per window (NOT
enumerated) per the structural-vs-tunable rule. The pair-switch knobs (q/reb/tilt/
long_only) are the bet-sizing layer.
"""
import gc
import numpy as np
from numpy.random import default_rng

import xs_common as xc
import xs_features as xf
import xs_sim as xsim
import xs_models as xm


def forward_rank_label(ret, valid, H):
    """y[t,i] = cross-sectional rank in [-1,1] of pair i's forward H-bar return,
    among pairs alive over [t, t+H]. lbl_end[t]=t+H. Causal as a TARGET (used only in
    training rows with lbl_end < boundary)."""
    T, N = ret.shape
    r = np.nan_to_num(ret)
    # forward cumulative H-bar return
    csum = np.cumsum(r, axis=0)
    fwd = np.full((T, N), np.nan, np.float32)
    fwd[:T - H] = (csum[H:] - csum[:T - H]).astype(np.float32)
    aliveH = np.zeros((T, N), bool)
    aliveH[:T - H] = valid[:T - H] & valid[H:]   # alive at entry AND exit
    y = np.full((T, N), np.nan, np.float32)
    for t in range(T - H):
        m = aliveH[t] & np.isfinite(fwd[t])
        n = int(m.sum())
        if n < 4:
            continue
        v = fwd[t][m]
        order = np.argsort(v)
        ranks = np.empty(n); ranks[order] = np.arange(n)
        yr = 2.0 * ranks / max(n - 1, 1) - 1.0
        row = np.full(N, np.nan, np.float32); row[np.where(m)[0]] = yr.astype(np.float32)
        y[t] = row
    lbl_end = np.full(T, -1, np.int64)
    lbl_end[:T - H] = np.arange(H, T)
    return y, lbl_end


def sample_knobs(rng, k):
    out = []
    for _ in range(k):
        out.append(dict(
            q=float(rng.choice([0.05, 0.1, 0.15, 0.2, 0.3])),
            long_only=bool(rng.integers(2)),
            tilt=float(rng.choice([0.0, 0.5, 1.0])),
            reb=int(rng.choice([1, 2, 4, 8, 24])),
        ))
    return out


def run_combo(panel, X, names, spec, ctx=None):
    """spec: family, H, feat_variant(unused stub for now). Returns (summary|None, records).
    records: (combo_id, window_id, phase, bar_idx, pnl_float). combo_id filled by caller."""
    family = spec["family"]; H = int(spec["H"]); cid = spec.get("combo_id", 0)
    ret = panel["ret"]; valid = panel["valid"]; funding = panel["funding"]
    T, N, F = X.shape
    y, lbl_end = forward_rank_label(ret, valid, H)
    seed0 = xc.combo_seed("xs", tuple(sorted((k, str(v)) for k, v in spec.items())))
    rng = default_rng(seed0)

    records = []; is_pnl_all = []; oos_pnl_all = []; nreb = 0; any_ok = False

    for wid, ws, ie, oe in xc.wfo_windows(T):
        sel_start = ie - xc.IS_SEL
        if sel_start <= ws + 500:
            continue
        # purged fit mask: rows in [ws,sel_start) with lbl_end < sel_start
        fit_t = np.arange(ws, sel_start)
        fit_t = fit_t[lbl_end[fit_t] >= 0]
        fit_t = fit_t[lbl_end[fit_t] < sel_start]
        if fit_t.size < 50:
            continue
        Xf = X[fit_t]; yf = y[fit_t]; vf = valid[fit_t]
        try:
            hp = xm.sample_hp(rng, family)
            model = xm.make_model(family, hp, seed0 + wid, F).fit(Xf, yf, vf)
            sc_sel = model.score(X[sel_start:ie], valid[sel_start:ie])
        except Exception:
            continue
        # knob search on selection slice
        knobs = sample_knobs(rng, xc.NS_KNOB)
        sel_pnl = xsim.simulate(sc_sel, ret[sel_start:ie], funding[sel_start:ie],
                                valid[sel_start:ie], knobs)
        best = None; bs = -1e18
        for ki in range(len(knobs)):
            s = xc.sharpe(sel_pnl[ki])
            if np.isfinite(s) and s > bs:
                bs = s; best = ki
        if best is None:
            continue
        knob = knobs[best]
        # refit on full purged IS
        ref_t = np.arange(ws, ie); ref_t = ref_t[lbl_end[ref_t] >= 0]; ref_t = ref_t[lbl_end[ref_t] < ie]
        if ref_t.size < 50:
            continue
        try:
            model = xm.make_model(family, hp, seed0 + wid, F).fit(X[ref_t], y[ref_t], valid[ref_t])
            sc_is = model.score(X[ws:ie], valid[ws:ie])
            sc_oos = model.score(X[ie:oe], valid[ie:oe])
        except Exception:
            continue
        is_pnl = xsim.simulate(sc_is, ret[ws:ie], funding[ws:ie], valid[ws:ie], [knob])[0]
        oos_pnl = xsim.simulate(sc_oos, ret[ie:oe], funding[ie:oe], valid[ie:oe], [knob])[0]
        for k in range(is_pnl.shape[0]):
            if is_pnl[k] != 0.0:
                records.append((cid, wid, 0, ws + k, float(is_pnl[k])))
        for k in range(oos_pnl.shape[0]):
            if oos_pnl[k] != 0.0:
                records.append((cid, wid, 1, ie + k, float(oos_pnl[k])))
        is_pnl_all.append(is_pnl); oos_pnl_all.append(oos_pnl); nreb += 1; any_ok = True

    # free the big per-combo arrays before returning so a respawn-on-next-task worker
    # (maxtasksperchild=1) never overlaps two combos' [T,N] working sets in RAM.
    del y, lbl_end
    try:
        del model
    except Exception:
        pass
    if not any_ok or nreb < xc.MIN_OOS_REB:
        gc.collect()
        return None, records
    oos = np.concatenate(oos_pnl_all)
    summ = dict(combo_id=cid, family=family, H=H,
                windows=nreb, oos_bars=int(oos.size),
                oos_sharpe=xc.sharpe(oos), oos_pf=xc.profit_factor(oos),
                oos_total_bp=float(oos.sum() * 1e4), oos_mean_bp=float(oos.mean() * 1e4))
    del is_pnl_all, oos_pnl_all, oos
    gc.collect()
    return summ, records
