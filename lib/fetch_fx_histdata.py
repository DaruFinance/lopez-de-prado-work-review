#!/usr/bin/env python3
"""Fetch HistData.com free FX tick quotes -> per-pair 1-minute base parquet.

Spot FX has no real volume (the tick `vol` field is always 0), so the only
information clock available is TICK COUNT (number of quote updates per minute).
We therefore store a 1-minute base with mid-price OHLC and `count` = ticks,
mirroring the crypto/equity 1m base schema so the same bar machinery applies.
Time bars and TICK bars are then the comparable pair for FX (volume/dollar bars
are not meaningful on spot FX and are reported N/A).

Output: <CACHE>/<PAIR>_fx1m.parquet  cols: open,high,low,close,count  (index=UTC minute)
"""
import os, io, re, zipfile, sys
import urllib.request, urllib.parse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_fx"
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "EURGBP"]
YEARS = [2022, 2023, 2024]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def fetch_month(pair, year, month):
    page = (f"https://www.histdata.com/download-free-forex-historical-data/"
            f"?/ascii/tick-data-quotes/{pair.lower()}/{year}/{month}")
    hdr = {"User-Agent": UA, "Referer": page}
    try:
        html = urllib.request.urlopen(urllib.request.Request(page, headers=hdr),
                                      timeout=60).read().decode("utf-8", "ignore")
    except Exception:
        return None
    m = re.search(r'id="tk"\s+value="([^"]+)"', html) or re.search(r'name="tk"\s+value="([^"]+)"', html)
    if not m:
        return None
    data = urllib.parse.urlencode({"tk": m.group(1), "date": str(year),
                                   "datemonth": f"{year}{month:02d}", "platform": "ASCII",
                                   "timeframe": "T", "fxpair": pair}).encode()
    try:
        blob = urllib.request.urlopen(urllib.request.Request(
            "https://www.histdata.com/get.php", data=data, headers=hdr), timeout=180).read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        csv = next(n for n in zf.namelist() if n.endswith(".csv"))
        df = pd.read_csv(io.BytesIO(zf.read(csv)), header=None,
                         names=["dt", "bid", "ask", "vol"], dtype={"dt": str})
    except Exception as e:
        print(f"  {pair} {year}-{month:02d} fail: {e}"); return None
    mid = (df["bid"].to_numpy() + df["ask"].to_numpy()) / 2.0
    key = df["dt"].str.slice(0, 8) + df["dt"].str.slice(9, 13)     # YYYYMMDDHHMM
    g = pd.DataFrame({"key": key, "mid": mid}).groupby("key")["mid"]
    base = pd.DataFrame({"open": g.first(), "high": g.max(),
                         "low": g.min(), "close": g.last(), "count": g.size()})
    base.index = pd.to_datetime(base.index, format="%Y%m%d%H%M", utc=True)
    return base


def build_pair(pair):
    out = f"{CACHE}/{pair}_fx1m.parquet"
    if os.path.exists(out):
        print(f"{pair}: cached, skip"); return
    tasks = [(pair, y, m) for y in YEARS for m in range(1, 13)]
    parts = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_month, *t): t for t in tasks}
        for fu in as_completed(futs):
            r = fu.result()
            if r is not None and len(r):
                parts.append(r)
    if not parts:
        print(f"{pair}: NO DATA"); return
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(out)
    print(f"{pair}: {len(df):,} 1m bars [{df.index.min().date()}..{df.index.max().date()}] -> {out}")


def main():
    os.makedirs(CACHE, exist_ok=True)
    for p in PAIRS:
        build_pair(p)
    print("FX ALL DONE")


if __name__ == "__main__":
    main()
