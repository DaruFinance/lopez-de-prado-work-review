#!/usr/bin/env python3
"""Fetch Binance USD-M perp 1m klines (full schema) -> one parquet per pair.

Downloads monthly dumps from data.binance.vision, concatenates, dedupes, and
writes <CACHE>/<PAIR>_1m.parquet with columns needed for information bars.
Raw zips/csvs are deleted after parsing. 404 months are skipped silently.
"""
import os, io, zipfile, sys
import urllib.request
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT",
         "AAVEUSDT", "ALGOUSDT", "APEUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT",
         "AVAXUSDT", "BCHUSDT", "DOTUSDT", "ETCUSDT", "HBARUSDT", "ICPUSDT",
         "LINKUSDT", "LTCUSDT", "NEARUSDT", "SUIUSDT", "TRXUSDT", "UNIUSDT",
         "XLMUSDT", "XRPUSDT", "ZECUSDT", "1000SHIBUSDT"]
MONTHS = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13)]
COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume",
        "count", "taker_buy_volume", "taker_buy_quote_volume"]
RAW = ["open_time", "open", "high", "low", "close", "volume", "close_time",
       "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def fetch_month(pair, month):
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
           f"{pair}/1m/{pair}-1m-{month}.zip")
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            data = r.read()
    except Exception:
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        csv = zf.read(zf.namelist()[0])
        # header present in newer dumps; detect by first byte
        first = csv[:10].decode("utf-8", "ignore")
        hdr = 0 if first.startswith("open_time") else None
        df = pd.read_csv(io.BytesIO(csv), header=hdr, names=None if hdr == 0 else RAW)
        df = df[COLS]
        return df
    except Exception as e:
        print(f"  parse fail {pair} {month}: {e}")
        return None


def build_pair(pair):
    out = f"{CACHE}/{pair}_1m.parquet"
    if os.path.exists(out):
        print(f"{pair}: cached, skip"); return
    parts = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_month, pair, m): m for m in MONTHS}
        for f in as_completed(futs):
            d = f.result()
            if d is not None and len(d):
                parts.append(d)
    if not parts:
        print(f"{pair}: NO DATA"); return
    df = pd.concat(parts, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (df.drop_duplicates("open_time").sort_values("open_time")
            .reset_index(drop=True))
    out = f"{CACHE}/{pair}_1m.parquet"
    df.to_parquet(out, index=False)
    print(f"{pair}: {len(df):,} 1m bars "
          f"[{df.open_time.min().date()} .. {df.open_time.max().date()}] -> {out}")


def main():
    os.makedirs(CACHE, exist_ok=True)
    for p in PAIRS:
        build_pair(p)
    print("ALL DONE")


if __name__ == "__main__":
    main()
