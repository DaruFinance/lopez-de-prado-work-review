"""T_XS panel loader — build a survivorship-bias-free, point-in-time, causal panel
over the Binance UM perp universe on a common 1h UTC grid.

Outputs aligned float32 arrays of shape [T, N] (T bars × N pairs):
  close, high, low, ret(1-bar log), dvol(dollar volume),
  funding (per-bar rate, ffilled from 8h, shift1),
  oi, oi_chg, toptrader_ls, global_ls, taker_ls (shift1),
  of_buyratio, of_delta (orderflow, ffilled from native 4h/8h, shift1),
  basis (perp/spot - 1, shift1),
  valid mask (bool: pair alive & has data at t, per listing_dates + actual bars).

Causality: ALL feature-grade arrays are shift(1) (a value at bar t uses data ≤ t-1).
close/high/low/ret are NOT shifted (they are the tradable price path for the sim and
the label source); the sim only ever uses ret[t] for a position decided at t-1.

Survivorship: includes delisted pairs from crypto_perp_delisted_1h; validity ends at
delisting_date. Listing-date gating prevents look-ahead on new listings.

RAM: float32; ~568 pairs × union-grid. A universe filter (history/dvol/limit) trims N.
Cache: one .npz per (universe-hash, span) under _cache/.

AUDIT FIX (SURVIVORSHIP PHANTOM BARS): The original 'valid' mask used only
isfinite(close) & (close > 0). Because listing_dates.csv has delisting_date=NaN for
ALL 1250 entries, the delisting gate never fired. ~280 delisted pairs fed phantom
stale-price bars (volume=0, repeated close price) indefinitely after delisting.

FIX: 'valid' now additionally requires quote_volume > 0. A bar with zero volume is a
stale phantom bar — the pair is no longer trading. Gate: isfinite(c) & (c > 0) &
(vol > 0). This correctly excludes all phantom delisted bars from the cross-section.
"""
import os
import glob
import hashlib
import numpy as np
import pandas as pd

import xs_common as xc


def _list_perp_symbols():
    fs = sorted(glob.glob(f"{xc.PERP_DIR}/*_1h.parquet"))
    live = {os.path.basename(f).replace("_1h.parquet", ""): f for f in fs}
    # delisted (survivorship): add any not already present
    for f in sorted(glob.glob(f"{xc.DELISTED_DIR}/*_1h.parquet")):
        s = os.path.basename(f).replace("_1h.parquet", "")
        if s not in live:
            live[s] = f
    return live


def _load_listing():
    try:
        ld = pd.read_csv(xc.LISTING_CSV, parse_dates=["listing_date", "first_bar", "last_bar", "delisting_date"])
        return {r.symbol: r for r in ld[ld.category == "binance_um"].itertuples()}
    except Exception:
        return {}


def _read_grid(path, cols):
    d = pd.read_parquet(path, columns=cols)
    tcol = "open_time" if "open_time" in d.columns else d.columns[0]
    d = d.set_index(tcol)
    d.index = pd.to_datetime(d.index, utc=True)
    return d[~d.index.duplicated(keep="last")].sort_index()


def _univ_hash(symbols, t0, t1):
    h = hashlib.md5(("|".join(sorted(symbols)) + str(t0) + str(t1)).encode()).hexdigest()[:12]
    return h


def build_panel(symbols=None, t_start=None, t_end=None, use_cache=True):
    """Return dict of [T,N] arrays + meta. symbols=None → eligible universe."""
    perp_files = _list_perp_symbols()
    listing = _load_listing()

    if xc.SMOKE_PAIRS:
        symbols = [s for s in xc.SMOKE_PAIRS.split(",") if s in perp_files]
    elif symbols is None:
        symbols = sorted(perp_files.keys())

    # --- pass 1: read perp closes, determine grid + eligibility ---
    raw = {}
    for s in symbols:
        try:
            d = _read_grid(perp_files[s], ["open_time", "open", "high", "low", "close", "volume", "quote_volume"])
            if len(d) >= xc.MIN_HISTORY_BARS:
                raw[s] = d
        except Exception:
            continue
    if not raw:
        raise RuntimeError("no symbols loaded")

    t0 = min(d.index[0] for d in raw.values()) if t_start is None else pd.Timestamp(t_start, tz="UTC")
    t1 = max(d.index[-1] for d in raw.values()) if t_end is None else pd.Timestamp(t_end, tz="UTC")
    grid = pd.date_range(t0, t1, freq="1h")
    T = len(grid)

    # --- optional liquidity ranking / limit ---
    syms = sorted(raw.keys())
    if xc.MIN_DOLLAR_VOL > 0 or xc.UNIVERSE_LIMIT > 0:
        dvol = {s: float(raw[s]["quote_volume"].median()) for s in syms}
        syms = [s for s in syms if dvol[s] >= xc.MIN_DOLLAR_VOL]
        syms = sorted(syms, key=lambda s: -dvol[s])
        if xc.UNIVERSE_LIMIT > 0:
            syms = syms[:xc.UNIVERSE_LIMIT]
        syms = sorted(syms)
    N = len(syms)

    cache_key = _univ_hash(syms, grid[0], grid[-1])
    cpath = f"{xc.CACHE_DIR}/panel_{cache_key}_{T}x{N}.npz"
    if use_cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return {k: z[k] for k in z.files} | {"symbols": list(z["symbols"]), "grid": grid}

    def emptyf():
        return np.full((T, N), np.nan, dtype=np.float32)

    close = emptyf(); high = emptyf(); low = emptyf(); dvol = emptyf()
    funding = np.zeros((T, N), np.float32)
    oi = emptyf(); oi_chg = emptyf(); ttls = emptyf(); gls = emptyf(); tls = emptyf()
    of_buy = emptyf(); of_delta = emptyf(); basis = emptyf()
    valid = np.zeros((T, N), dtype=bool)

    gi = pd.Series(np.arange(T), index=grid)

    for j, s in enumerate(syms):
        d = raw[s].reindex(grid)
        c = d["close"].to_numpy(np.float64)
        close[:, j] = c
        high[:, j] = d["high"].to_numpy(np.float64)
        low[:, j] = d["low"].to_numpy(np.float64)
        dvol[:, j] = d["quote_volume"].to_numpy(np.float64)
        vol = d["quote_volume"].to_numpy(np.float64)
        # SURVIVORSHIP FIX: require volume > 0 to exclude phantom stale-price delisted bars.
        # Bars with zero volume after delisting carry a repeated last price and no activity.
        has = np.isfinite(c) & (c > 0) & np.isfinite(vol) & (vol > 0)
        valid[:, j] = has

        # funding 8h → ffill to 1h, shift1 (per-bar scaled later in sim)
        fp = f"{xc.FUND_DIR}/{s}_funding.parquet"
        if os.path.exists(fp):
            fd = _read_grid(fp, ["funding_time", "funding_rate"]).reindex(grid).ffill()
            fr = fd["funding_rate"].to_numpy(np.float64)
            funding[:, j] = np.nan_to_num(np.concatenate([[0.0], fr[:-1]]), nan=0.0)

        # OI 1h + ratios, shift1
        op = f"{xc.OI_DIR}/{s}_oi_1h.parquet"
        if os.path.exists(op):
            od = _read_grid(op, ["open_time", "open_interest", "toptrader_ls_ratio",
                                 "global_ls_ratio", "taker_ls_ratio"]).reindex(grid).ffill()
            o = od["open_interest"].to_numpy(np.float64)
            oi[:, j] = np.concatenate([[np.nan], o[:-1]])
            with np.errstate(divide="ignore", invalid="ignore"):
                ch = np.concatenate([[np.nan], np.diff(np.log(np.where(o > 0, o, np.nan)))])
            oi_chg[:, j] = np.concatenate([[np.nan], ch[:-1]])
            for arr, col in [(ttls, "toptrader_ls_ratio"), (gls, "global_ls_ratio"), (tls, "taker_ls_ratio")]:
                v = od[col].to_numpy(np.float64)
                arr[:, j] = np.concatenate([[np.nan], v[:-1]])

        # orderflow native 4h/8h → ffill 1h, shift1
        ofg = glob.glob(f"{xc.OF_DIR}/{s}_of_*.parquet")
        if ofg:
            ofd = _read_grid(ofg[0], ["open_time", "buy_ratio", "delta_base"]).reindex(grid).ffill()
            br = ofd["buy_ratio"].to_numpy(np.float64); db = ofd["delta_base"].to_numpy(np.float64)
            of_buy[:, j] = np.concatenate([[np.nan], br[:-1]])
            of_delta[:, j] = np.concatenate([[np.nan], db[:-1]])

        # basis = perp/spot - 1, shift1
        sp = f"{xc.SPOT_DIR}/{s}_1h.parquet"
        if os.path.exists(sp):
            sd = _read_grid(sp, ["open_time", "close"]).reindex(grid)
            sc = sd["close"].to_numpy(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                b = c / np.where(sc > 0, sc, np.nan) - 1.0
            basis[:, j] = np.concatenate([[np.nan], b[:-1]])

        # listing-date gate: invalidate bars before listing / after delisting
        r = listing.get(s)
        if r is not None:
            if pd.notna(getattr(r, "listing_date", None)):
                valid[grid < pd.Timestamp(r.listing_date), j] = False
            if pd.notna(getattr(r, "delisting_date", None)):
                valid[grid > pd.Timestamp(r.delisting_date), j] = False

    # 1-bar forward log return of the tradable price (NOT shifted — it's the path)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close > 0, close, np.nan))
    ret = np.full((T, N), np.nan, np.float32)
    ret[1:] = (logc[1:] - logc[:-1]).astype(np.float32)

    out = dict(close=close, high=high, low=low, ret=ret, dvol=dvol, funding=funding,
               oi=oi, oi_chg=oi_chg, toptrader_ls=ttls, global_ls=gls, taker_ls=tls,
               of_buy=of_buy, of_delta=of_delta, basis=basis, valid=valid,
               symbols=np.array(syms), T=T, N=N)
    if use_cache:
        os.makedirs(xc.CACHE_DIR, exist_ok=True)
        np.savez_compressed(cpath, **{k: v for k, v in out.items() if k != "grid"})
    out["grid"] = grid
    return out


if __name__ == "__main__":
    import sys
    os.environ.setdefault("XS_SMOKE_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    p = build_panel(use_cache=False)
    print(f"panel T={p['T']} N={p['N']} symbols={list(p['symbols'])}")
    print(f"valid coverage: {p['valid'].mean()*100:.1f}%  | ret finite: {np.isfinite(p['ret']).mean()*100:.1f}%")
    print(f"funding nonzero: {(p['funding']!=0).mean()*100:.1f}%  basis finite: {np.isfinite(p['basis']).mean()*100:.1f}%")
