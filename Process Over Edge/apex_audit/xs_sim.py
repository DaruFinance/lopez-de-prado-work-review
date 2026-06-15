"""T_XS portfolio sim — the pair-switch IS the strategy.

Given a per-bar per-pair causal SCORE [T,N] (decided at t-1, applied at t), build
cross-sectional weights and simulate a long-short rotation with per-fill costs +
funding. Batched over K knob configs on GPU (one big tensor sim), which is where the
GPU earns its keep — K simultaneous portfolio paths over [T,N].

Mechanics per bar t (all causal — score/valid are shift1-safe from upstream):
  1. eligible = valid[t] & finite(score[t]).
  2. select: long the top `q` fraction by score, short the bottom `q` (q a knob).
     (q=top/bottom quantile; long_only knob disables shorts.)
  3. weight: equal-weight within each leg, gross-normalized to 1 (so |w| sums to 1
     per side); optional score-tilt (knob). market-neutral (sum w = 0) unless long_only.
  4. pnl[t] = sum_i w[t,i]*ret[t,i]  - turnover_cost - funding_cost.
     turnover = sum |w[t]-w[t-1]| ; cost = turnover * TAKER_FILL (rotations are taker).
     funding = sum w[t,i]*funding[t,i]*FUNDING_PER_BAR_SCALE  (long pays positive funding).
  5. hold/rebalance throttle: only rebalance every `reb` bars (knob) → controls turnover/fees.

Returns per-knob OOS/IS pnl series so the WFO can pick the best IS-Sharpe knob and
report the OOS path. NumPy CPU fallback mirrors the torch path bit-for-bit in logic.
"""
import numpy as np
import xs_common as xc


def _weights_from_score(score, valid, q, long_only, tilt):
    """Vectorized per-bar weights for ONE knob, numpy [T,N]->[T,N]. (CPU reference.)

    Robust: select longs/shorts ONLY from the valid (alive, finite-score) pairs each
    bar, rank within that valid subset, and guard every weight-sum denominator. This
    fixes the earlier bug where masked pairs (set to -inf) were picked up as shorts."""
    T, N = score.shape
    W = np.zeros((T, N), np.float64)
    for t in range(T):
        m = valid[t] & np.isfinite(score[t])
        vidx = np.where(m)[0]              # absolute indices of VALID pairs only
        n = vidx.size
        if n < 4:
            continue
        sv = score[t][vidx].astype(np.float64)
        k = int(np.floor(q * n))
        k = min(max(k, 1), n // 2 if not long_only else n)  # leave room for both legs
        order = np.argsort(sv)            # asc within valid subset
        longs = vidx[order[-k:]]          # highest scores
        w = np.zeros(N)
        lt = 1.0 + tilt * (score[t][longs] - score[t][longs].mean())
        lt = np.clip(lt, 0.1, None)
        ls = lt.sum()
        if ls <= 1e-12:
            continue
        w[longs] = lt / ls
        if not long_only:
            shorts = vidx[order[:k]]      # lowest scores among VALID pairs
            st = 1.0 + tilt * (score[t][shorts].mean() - score[t][shorts])
            st = np.clip(st, 0.1, None)
            ss = st.sum()
            if ss > 1e-12:
                w[shorts] = -st / ss
        W[t] = w
    return W


def simulate_cpu(score, ret, funding, valid, knobs):
    """Reference CPU sim over a list of knob dicts. Returns pnl [K,T]."""
    K = len(knobs); T, N = score.shape
    out = np.zeros((K, T), np.float64)
    for ki, kn in enumerate(knobs):
        reb = int(kn["reb"]); q = float(kn["q"])
        lo = bool(kn["long_only"]); tilt = float(kn["tilt"])
        W = _weights_from_score(score, valid, q, lo, tilt)
        if reb > 1:                      # hold weights between rebalances
            for t in range(1, T):
                if t % reb != 0:
                    W[t] = W[t - 1]
        wprev = np.zeros(N)
        for t in range(T):
            r = np.nan_to_num(ret[t]); f = np.nan_to_num(funding[t])
            gross = float(W[t] @ r)
            turn = float(np.abs(W[t] - wprev).sum())
            fund = float((W[t] * f).sum()) * xc.FUNDING_PER_BAR_SCALE
            out[ki, t] = gross - turn * xc.TAKER_FILL - fund
            wprev = W[t]
    return out


def simulate_gpu(score, ret, funding, valid, knobs, tile_T=8192):
    """Batched GPU sim. All K knobs share the rank computation; weights differ by
    (q, long_only, tilt); turnover handled with a per-knob rebalance throttle.
    Returns pnl [K,T] float64. Falls back to CPU if torch/GPU absent."""
    if not xc.gpu_enabled():
        return simulate_cpu(score, ret, funding, valid, knobs)
    import torch
    dev = torch.device("cuda")
    T, N = score.shape
    K = len(knobs)
    f64 = torch.float64
    qs = torch.tensor([k["q"] for k in knobs], device=dev, dtype=f64)
    los = torch.tensor([1.0 if k["long_only"] else 0.0 for k in knobs], device=dev, dtype=f64)
    tilts = torch.tensor([k["tilt"] for k in knobs], device=dev, dtype=f64)
    rebs = [int(k["reb"]) for k in knobs]
    pnl = torch.zeros((K, T), dtype=f64, device=dev)
    wprev = torch.zeros((K, N), dtype=f64, device=dev)
    sc = torch.from_numpy(score).to(dev).double()
    rt = torch.from_numpy(np.nan_to_num(ret)).to(dev).double()
    fd = torch.from_numpy(np.nan_to_num(funding)).to(dev).double()
    vt = torch.from_numpy(valid).to(dev)
    for t in range(T):
        m = vt[t] & torch.isfinite(sc[t])
        n = int(m.sum())
        if n < 4:
            pnl[:, t] = -(wprev.abs().sum(1) * 0)  # carry zero; weights unchanged
            continue
        s = torch.where(m, sc[t], torch.tensor(float("-inf"), device=dev, dtype=f64))  # [N]
        order = torch.argsort(s)                    # asc
        ranks = torch.empty(N, device=dev, dtype=f64); ranks[order] = torch.arange(N, device=dev, dtype=f64)
        # per-knob k = floor(q*n)
        kk = torch.clamp((qs * n).floor(), min=1)   # [K]
        long_thr = (n - kk).view(K, 1)              # rank >= this => long
        short_thr = kk.view(K, 1)                   # rank < this => short
        rk = ranks.view(1, N)
        sval = torch.where(m, sc[t], torch.zeros_like(sc[t])).double().view(1, N)
        long_mask = (rk >= long_thr) & m.view(1, N)
        short_mask = (rk < short_thr) & m.view(1, N) & (los.view(K, 1) < 0.5)
        wl = long_mask.double() * (1.0 + tilts.view(K, 1) * sval)
        wl = wl.clamp(min=0.0)
        wl = wl / wl.sum(1, keepdim=True).clamp(min=1e-12)
        ws = short_mask.double() * (1.0 + tilts.view(K, 1) * (-sval))
        ws = ws.clamp(min=0.0)
        ws = ws / ws.sum(1, keepdim=True).clamp(min=1e-12)
        W = wl - ws                                  # [K,N]
        # rebalance throttle: per-knob, if t % reb != 0 keep previous weights
        keep = torch.tensor([(t % r != 0) and t > 0 for r in rebs], device=dev).view(K, 1)
        W = torch.where(keep, wprev, W)
        gross = (W * rt[t].view(1, N)).sum(1)
        turn = (W - wprev).abs().sum(1)
        fund = (W * fd[t].view(1, N)).sum(1) * xc.FUNDING_PER_BAR_SCALE
        pnl[:, t] = gross - turn * xc.TAKER_FILL - fund
        wprev = W
    return pnl.cpu().numpy()


def simulate(score, ret, funding, valid, knobs):
    return simulate_gpu(score, ret, funding, valid, knobs)
