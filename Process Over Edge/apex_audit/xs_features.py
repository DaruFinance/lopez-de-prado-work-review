"""T_XS cross-sectional features — GPU-batched, causal.

Every feature is computed per-pair causally (all inputs already shift(1) in xs_data),
then CROSS-SECTIONALLY normalized at each bar t (rank in [-1,1] and z-score across the
alive universe at t). The cross-section is the whole point: a pair's feature is its
standing RELATIVE to the universe that bar, which is what drives the pair-switch.

Returns a feature tensor X of shape [T, N, F] (float32) + a valid mask [T, N].
NaNs (dead/missing pairs) are excluded from each bar's cross-sectional stats and
filled to 0 (= neutral / median rank) so the model/sim never see NaN.

Runs on GPU if available (XS_GPU=1) — the rank/zscore over [T,N] is a big batched
tensor op; on an 8GB card we tile over time to bound VRAM.

AUDIT FIX (LOOKAHEAD): The original engine fed panel['ret'] (the contemporaneous
bar-t return, ret[t]=log(close[t])-log(close[t-1])) directly as a feature. The sim
simultaneously earns pnl[t] = W[t] @ ret[t]. This is a 1-bar lookahead: the model
uses the same bar's return as a feature that it then earns. Cross-sectional 1-bar
reversal is a known strong crypto signal, so this inflates PF artificially.

FIX: 'ret' is REMOVED from BASE_FIELDS and replaced with 'ret_lag1' = ret shifted by
1 bar, so the model at bar t uses ret[t-1] (the PRIOR bar's return). This is
implemented in build_raw_features() by an explicit np.roll/nan-fill shift of the
panel ret array. The pollute test: poisoning close[t] changes ret[t] but does NOT
change ret_lag1[t] (which is ret[t-1]), so a position at t is unaffected.
"""
import numpy as np
import xs_common as xc

# raw per-pair series used as cross-sectional feature inputs (all already causal in xs_data)
# NOTE: 'ret' REMOVED — it was the contemporaneous bar-t return, causing 1-bar lookahead.
# Replaced by 'ret_lag1' (ret[t-1]), built explicitly in build_raw_features().
BASE_FIELDS = [
    "oi_chg", "funding", "toptrader_ls", "global_ls", "taker_ls",
    "of_buy", "of_delta", "basis",
]


def _mom(close, k):
    """k-bar log momentum, shift1 (causal). close is [T,N]."""
    lc = np.log(np.where(close > 0, close, np.nan))
    m = np.full_like(lc, np.nan, dtype=np.float64)
    m[k:] = lc[k:] - lc[:-k]
    out = np.full_like(m, np.nan)
    out[1:] = m[:-1]               # shift1
    return out.astype(np.float32)


def _vol(ret, k):
    """trailing k-bar realized vol of ret, shift1."""
    import pandas as pd
    r = pd.DataFrame(ret)
    v = r.rolling(k, min_periods=max(5, k // 4)).std().shift(1).to_numpy(np.float32)
    return v


def build_raw_features(panel):
    """Assemble per-pair raw feature stack [T,N,F0] (pre cross-sectional norm).

    LOOKAHEAD FIX: ret[t] (contemporaneous bar-t return) is NOT used as a feature.
    Instead we add 'ret_lag1' = ret shifted by 1 bar (ret[t-1]), so the model at
    bar t only sees the return that ALREADY happened at t-1. This is the correct
    causal representation and avoids the 1-bar reversal lookahead.
    """
    close = panel["close"].astype(np.float64)
    ret = panel["ret"].astype(np.float64)
    feats = {}
    for k in (6, 24, 72, 168):
        feats[f"mom{k}"] = _mom(close, k)
    for k in (24, 72):
        feats[f"vol{k}"] = _vol(panel["ret"], k)
    # LOOKAHEAD FIX: ret_lag1 = ret[t-1], causal at t. Replace raw ret.
    T, N = ret.shape
    ret_lag1 = np.full((T, N), np.nan, np.float32)
    ret_lag1[1:] = ret[:-1].astype(np.float32)   # shift by 1: ret_lag1[t] = ret[t-1]
    feats["ret_lag1"] = ret_lag1
    # ABLATION HOOK: drop any feature named in $XS_ABLATE (comma-sep) so the WFO can
    # measure the economic magnitude carried by a single feature. The forward-filled
    # order-flow leak is reproduced with XS_ABLATE="of_buy,of_delta". Empty => baseline.
    import os as _os
    _ablate = {x.strip() for x in _os.environ.get("XS_ABLATE", "").split(",") if x.strip()}
    for f in BASE_FIELDS:
        if f in _ablate:
            continue
        feats[f] = panel[f].astype(np.float32)
    if _ablate:
        print(f"[ABLATE] dropped features: {sorted(_ablate)} | remaining raw: {list(feats.keys())}", flush=True)
    names = list(feats.keys())
    F0 = len(names)
    T, N = close.shape
    X = np.full((T, N, F0), np.nan, np.float32)
    for i, nm in enumerate(names):
        X[:, :, i] = feats[nm]
    return X, names


def cross_sectional_norm(X, valid, use_gpu=None, tile=4096):
    """Per-bar cross-sectional rank(->[-1,1]) and z-score across alive pairs.
    Input X [T,N,F0]; output Xn [T,N,2*F0] (rank block || zscore block), NaN->0.
    valid [T,N] bool gates which pairs enter each bar's cross-section.
    GPU-batched over time tiles to bound VRAM on the 8GB card."""
    T, N, F0 = X.shape
    use_gpu = xc.gpu_enabled() if use_gpu is None else use_gpu
    out = np.zeros((T, N, 2 * F0), np.float32)
    if use_gpu:
        import torch
        dev = torch.device("cuda")
        for a in range(0, T, tile):
            b = min(a + tile, T)
            xt = torch.from_numpy(X[a:b]).to(dev)              # [t,N,F0]
            vt = torch.from_numpy(valid[a:b]).to(dev).unsqueeze(-1)  # [t,N,1]
            m = vt & torch.isfinite(xt)
            xm = torch.where(m, xt, torch.nan)
            # rank within each bar across N (dim=1), ignoring NaN
            # argsort of argsort gives ranks; NaN pushed to end then masked
            big = torch.where(m, xm, torch.tensor(float("inf"), device=dev))
            order = big.argsort(dim=1)
            ranks = torch.empty_like(order, dtype=torch.float32)
            ar = torch.arange(N, device=dev, dtype=torch.float32).view(1, N, 1)
            ranks.scatter_(1, order, ar.expand_as(order))
            cnt = m.sum(dim=1, keepdim=True).clamp(min=1)       # alive count per bar
            rank01 = (2.0 * ranks / (cnt - 1).clamp(min=1) - 1.0)  # ~[-1,1]
            rank01 = torch.where(m, rank01, torch.zeros_like(rank01))
            # zscore across alive pairs
            mean = torch.where(m, xm, torch.zeros_like(xm)).sum(1, keepdim=True) / cnt
            var = torch.where(m, (xm - mean) ** 2, torch.zeros_like(xm)).sum(1, keepdim=True) / cnt
            z = (xm - mean) / torch.sqrt(var + 1e-12)
            z = torch.where(m, z, torch.zeros_like(z)).clamp(-5, 5)
            out[a:b, :, :F0] = rank01.cpu().numpy()
            out[a:b, :, F0:] = z.cpu().numpy()
            del xt, vt, m, xm, big, order, ranks, rank01, z
        return out, [f"rank_{i}" for i in range(F0)] + [f"z_{i}" for i in range(F0)]
    # CPU fallback (numpy, vectorized per tile)
    for a in range(0, T, tile):
        b = min(a + tile, T)
        xb = X[a:b].astype(np.float64); vb = valid[a:b][:, :, None] & np.isfinite(xb)
        xm = np.where(vb, xb, np.nan)
        cnt = vb.sum(1, keepdims=True).clip(min=1)
        order = np.argsort(np.where(vb, xm, np.inf), axis=1)
        ranks = np.empty_like(order, np.float64)
        ar = np.arange(N)[None, :, None]
        np.put_along_axis(ranks, order, np.broadcast_to(ar, order.shape).copy(), axis=1)
        rank01 = np.where(vb, 2.0 * ranks / np.clip(cnt - 1, 1, None) - 1.0, 0.0)
        mean = np.nansum(np.where(vb, xm, 0.0), 1, keepdims=True) / cnt
        var = np.nansum(np.where(vb, (xm - mean) ** 2, 0.0), 1, keepdims=True) / cnt
        z = np.clip(np.where(vb, (xm - mean) / np.sqrt(var + 1e-12), 0.0), -5, 5)
        out[a:b, :, :F0] = rank01.astype(np.float32)
        out[a:b, :, F0:] = z.astype(np.float32)
    return out, [f"rank_{i}" for i in range(F0)] + [f"z_{i}" for i in range(F0)]


def build_features(panel, use_gpu=None):
    X0, names0 = build_raw_features(panel)
    Xn, names = cross_sectional_norm(X0, panel["valid"], use_gpu=use_gpu)
    return Xn, names, names0
