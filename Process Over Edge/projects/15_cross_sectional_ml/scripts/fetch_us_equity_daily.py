#!/usr/bin/env python3
"""fetch_us_equity_daily.py : download ADJUSTED daily OHLC for a bounded, liquid
US large-cap universe from a commercial daily-OHLC data API (eq-daily-ohlc).

All API connection details are read from the environment so nothing
provider-specific is committed:

  EQ_API_BASE    base URL of the data API host (e.g. https://api.example.com)
  EQ_API_PATH    request path for the daily-OHLC endpoint
  EQ_API_KEY     API key, sent in the request header
  EQ_API_KEY_HDR header name for the key (default: x-api-key)
  EQ_XS_DIR      output directory, one csv.gz per ticker (config.EQUITY_DAILY_XS)

Respects a conservative request envelope: ONE process, ~4 RPS, csv_gzip,
exponential backoff on 5xx/429, resumable (skip-if-cached). Daily bars for a
few hundred names over a multi-year window are tiny (~1.3k rows/name), so this
stays far under typical monthly data-returned caps.

Universe: a fixed list of large-cap names (broad-index members as of the run
date). This is a CURRENT-CONSTITUENT list, so the panel is
survivorship-aware-but-not-survivorship-free; the writeup states that caveat.
"""
from __future__ import annotations
import gzip, os, sys, time, urllib.request, urllib.error
from datetime import date

# Repo root -> config (data output dir); API connection comes from the env.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import config as cfg  # noqa: E402

BASE = os.environ.get("EQ_API_BASE", "").rstrip("/")
PATH = os.environ.get("EQ_API_PATH", "/api/v1/data/us-equity/eq-daily-ohlc")
KEY_HDR = os.environ.get("EQ_API_KEY_HDR", "x-api-key")
KEY = os.environ.get("EQ_API_KEY", "")
OUT = os.environ.get("EQ_XS_DIR", cfg.EQUITY_DAILY_XS)
PAGE = 10000
RPS = 4.0
MIN_GAP = 1.0 / RPS
START = os.environ.get("EQ_START", "2015-01-01")
# Account's allowed range ends at the current date; do not request future dates
# (the API rejects an end beyond the account cap with HTTP 422).
END = os.environ.get("EQ_END", date.today().isoformat())

# Fixed large-cap universe (well-known S&P 500 members; mega/large-cap, liquid,
# broad sector spread). Current-constituent list -> survivorship caveat noted.
UNIVERSE = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","META","NVDA","TSLA","BRK.B","JPM",
    "JNJ","V","PG","UNH","HD","MA","BAC","XOM","DIS","ADBE",
    "CRM","NFLX","CMCSA","KO","PEP","INTC","CSCO","VZ","WMT","ABT",
    "T","MRK","PFE","TMO","NKE","ORCL","ACN","COST","AVGO","TXN",
    "WFC","MCD","DHR","QCOM","NEE","UPS","PM","LIN","HON","UNP",
    "LOW","MS","IBM","SBUX","RTX","AMD","INTU","CAT","GS","AMGN",
    "BLK","BA","GE","ISRG","NOW","BKNG","DE","SPGI","AXP","GILD",
    "MDT","C","CVX","LMT","ADP","TJX","SYK","MMM","CI","MO",
    "ZTS","CB","DUK","SO","PLD","BDX","CL","ITW","MMC","APD",
    "USB","BSX","NSC","FDX","SCHW","AON","ICE","WM","PNC","CME",
    "EW","SHW","HUM","FIS","ETN","EMR","GD","NOC","COP","SLB",
    "EOG","PSX","VLO","MPC","KMB","GIS","KHC","MDLZ","STZ","HSY",
    "MNST","KDP","ADM","SYY","DG","DLTR","ROST","ORLY","AZO","YUM",
    "MAR","HLT","CMG","F","GM","DOW","DD","ECL","NEM","FCX",
    "PPG","NUE","VMC","MLM","ALB","CTVA","FANG","HES","OXY","WMB",
    "KMI","OKE","D","EXC","AEP","SRE","XEL","PEG","ED","WEC",
    "ES","DTE","AEE","CNP","CMS","PSA","O","SPG","AMT","CCI",
    "EQIX","DLR","WELL","AVB","EQR","VTR","ARE","BXP","ESS","MAA",
    "PRU","MET","AIG","TRV","ALL","AFL","PGR","HIG","WBA","CVS",
    "MCK","ABC","CAH","A","IDXX","IQV","RMD","DXCM","BIIB","REGN",
    "VRTX","MRNA","ILMN","ALGN","WAT","MTD","PKI","BAX","HCA","CNC",
    "ANTM","DVA","UHS","LH","DGX","BMY","LLY","ABBV","AMAT","LRCX",
    "KLAC","MU","ADI","MCHP","NXPI","SNPS","CDNS","ANSS","FTNT","PANW",
    "CRWD","ZS","DDOG","SNOW","NET","TEAM","WDAY","DOCU","OKTA","PYPL",
    "SQ","SHOP","UBER","ABNB","COIN","ROKU","PINS","SNAP","SPOT","ZM",
]


_last = [0.0]
def _pace():
    dt = time.monotonic() - _last[0]
    if dt < MIN_GAP:
        time.sleep(MIN_GAP - dt)
    _last[0] = time.monotonic()


def fetch(params, attempt=1):
    _pace()
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}{PATH}?{qs}"
    req = urllib.request.Request(url, headers={KEY_HDR: KEY, "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            buf = r.read()
        return gzip.decompress(buf).decode()
    except urllib.error.HTTPError as e:
        code = e.code
        body = e.read()[:200].decode(errors="replace")
        if code in (500, 502, 503, 504) and attempt <= 4:
            w = 2 ** attempt
            print(f"    HTTP {code} attempt {attempt}; sleeping {w}s", flush=True)
            time.sleep(w)
            return fetch(params, attempt + 1)
        if code == 429 and attempt <= 4:
            print("    HTTP 429; sleeping 60s", flush=True)
            time.sleep(60)
            return fetch(params, attempt + 1)
        raise RuntimeError(f"HTTP {code} on {url}: {body}")


def pull(ticker):
    safe = ticker.replace("/", "_").replace(".", "_")
    out_path = f"{OUT}/{safe}.csv.gz"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  {ticker:<6} SKIP (cached)", flush=True)
        return "skip"
    all_lines, offset, header = [], 0, None
    t0 = time.monotonic()
    # API ticker form uses '.' for class shares (BRK.B); pass through.
    while True:
        params = {
            "Ticker": ticker, "TradeDate.ge": START, "TradeDate.le": END,
            "adjusted": "true", "limit": PAGE, "offset": offset,
            "response_format": "csv_gzip",
        }
        body = fetch(params)
        lines = body.splitlines()
        if not lines:
            break
        if header is None:
            header = lines[0]; all_lines.append(header)
        data = lines[1:]
        if not data:
            break
        all_lines.extend(data)
        offset += len(data)
        if len(data) < PAGE:
            break
    if header is None or len(all_lines) <= 1:
        print(f"  {ticker:<6} EMPTY", flush=True)
        return "empty"
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("\n".join(all_lines))
    dt = time.monotonic() - t0
    print(f"  {ticker:<6} rows={len(all_lines)-1:>5d}  {os.path.getsize(out_path)/1024:>6.1f}KB  {dt:4.1f}s", flush=True)
    return "ok"


def main():
    if not BASE or not KEY:
        sys.exit("Set EQ_API_BASE and EQ_API_KEY (and optionally EQ_API_PATH / "
                 "EQ_API_KEY_HDR) for your daily-OHLC data API before running.")
    os.makedirs(OUT, exist_ok=True)
    uniq = list(dict.fromkeys(UNIVERSE))
    print(f"Downloading eq-daily-ohlc (adjusted) for {len(uniq)} names {START}..{END}", flush=True)
    print(f"Rate {RPS} RPS, out {OUT}\n", flush=True)
    n_ok = n_skip = n_empty = 0
    for tk in uniq:
        try:
            r = pull(tk)
            n_ok += r == "ok"; n_skip += r == "skip"; n_empty += r == "empty"
        except Exception as e:
            print(f"  {tk:<6} ERROR {e}", flush=True)
    print(f"\nDONE ok={n_ok} cached={n_skip} empty={n_empty}", flush=True)


if __name__ == "__main__":
    main()
