# Project 1 — Information-Driven Bars, across Crypto, US Equities & Forex

**LdP source:** *Advances in Financial Machine Learning* (2018), Ch. 2; "The Volume Clock" (2012).
**Markets:** Crypto (27 Binance USD-M perps), US Equities (9 Algoseek ETFs), Forex (8 HistData majors).
**Verdict:** Reproduced and confirmed in all three markets — *with two practical caveats LdP under-states.*

---

## 1. What LdP proposed

Standard practice samples the market on a **fixed time clock** (1-minute, 1-hour, daily bars). LdP
argues this is the worst choice: markets process information at a time-varying rate, so time bars
over-sample quiet periods and under-sample bursts, producing returns with **excess kurtosis, skew,
and serial correlation** — violating the near-IID-Gaussian assumptions most estimators rely on.

His fix is to sample on an **information clock**: emit a new bar every *X* ticks (tick bars), every
*X* units traded (volume bars), or every *$X* of notional (dollar bars). This "subordinates" price to
an activity clock (Mandelbrot–Clark) and, he claims, recovers returns much closer to IID-Gaussian,
with **dollar bars** the most robust because they are invariant to price level and share-count
changes. The testable predictions: information bars should have **lower excess kurtosis, lower
|skew|, and weaker serial correlation** than time bars at matched frequency.

## 2. Data & method

- **Base data, 1-minute, real:** crypto = clean Binance USD-M perp dumps (2022–2024, 27 pairs);
  equities = Algoseek ETF 1-min (SPY/QQQ/IWM + sector SPDRs + vol ETFs, deep history);
  forex = HistData quote ticks for 8 majors (2022–2024) resampled to 1-min with **tick count**.
- **Bars:** from each 1-min base we build time / tick / volume / dollar bars, all targeting the
  **same ~daily average frequency** (thresholds = total activity / number of days), so the only
  difference is the *clock*, not the bar count — the apples-to-apples setup LdP uses.
- **Metrics:** excess kurtosis (0 = Gaussian), |skew|, first-order autocorrelation, Jarque–Bera.
- **Causal & cost-free by nature:** this is a statistical-property study of the sampling scheme; no
  strategy, no look-ahead. Cost modelling enters later projects.
- Code: `scripts/run_bars_study_clean.py` (crypto), `run_equities_bars.py` (equities, sessions),
  `run_forex_bars.py` (forex), `run_granularity_study.py` (base-resolution), `make_final_figure.py`.

## 3. Headline result (all three markets)

Median excess kurtosis of bar returns, 1-minute base (`tables/FINAL_multimarket_exkurt.csv`,
`figures/FINAL_multimarket_kurtosis.png`):

| bar | crypto (27 perps) | equities (9 ETFs, RTH) | forex (8 majors) |
|---|---|---|---|
| **time** | 5.29 | 9.95 | 2.98 |
| **tick** | 1.61 | 3.97 | 1.06 |
| **volume** | 1.48 | 3.35 | — |
| **dollar** | 1.54 | 5.58 | — |

**Information-driven bars Gaussianize returns in every market** (and cut |skew| similarly:
crypto 0.37→0.10, forex 0.20→0.13). LdP confirmed. But the *how* differs by market, which is where
the value is.

## 4. Crypto — clean confirmation, plus two methodological findings

- **Cross-section (27 perps, clean 1m):** excess kurtosis 5.29 (time) → 1.61 / 1.48 / 1.54
  (tick / volume / dollar); |skew| 0.37 → ~0.10. All information bars roughly **3.4× less
  tail-heavy** than time bars. (`fig1_excess_kurtosis_clean.png`, `fig2_skew_autocorr_clean.png`,
  `fig4_qq_BTCUSDT_clean.png`.)
- **Finding A — the advantage is granularity-dependent** (`fig5`, `fig6`). Building information bars
  from coarser bricks erodes the benefit: dollar-bar excess kurtosis rises from **1.13 (1-min base)
  to ~2.06 (60-min base)**. The directional win over time bars survives at every resolution for
  majors, but the method *wants fine base data*. Practitioners accumulating dollar bars from 30-min
  bricks leave most of the benefit on the table.
- **Finding B — reproduction is fragile to data quality.** Our first run used a legacy third-party
  30-min dataset and produced the *opposite* conclusion (dollar bars *worse* than time bars). The
  same instrument at the same 30-min frequency showed BTC dollar-bar excess kurtosis of **32.7 on
  the legacy data vs 1.05 on clean Binance dumps**. Contaminated early-period prints get
  concentrated by dollar sampling into a few fat-tail bars. We discarded the legacy result. *Lesson:
  the bar comparison is a sensitive instrument; validate the tape first.*

## 5. Equities — the method works, but only with session-aware handling

Naively (all hours, returns taken across the overnight gap) the result looks like a **failure**:
dollar bars (34.4) and tick bars (19.5) come out *worse* than time bars (18.3). This is an
**artifact of session structure**, not of the method. Restricting to regular trading hours
(09:30–16:00 ET) and computing **within-session returns** flips it completely
(`tables/equities_session.md`, `figures/fig8_equities_session_effect.png`):

| bar | naive (all hrs + overnight) | RTH + within-session |
|---|---|---|
| time | 18.3 | 9.95 |
| tick | 19.5 | **3.97** |
| volume | 12.9 | **3.35** |
| dollar | 34.4 | **5.58** |

The overnight gap and open/close-auction volume spikes get bundled into single information bars,
manufacturing fat tails. Handle the session and **all information bars beat time bars**, just as in
crypto. This is a concrete operating instruction LdP's text glosses over: *on session-based markets,
build information bars within-session and never let a bar span the close→open gap.*

## 6. Forex — no volume, so the tick clock is the information bar

Spot FX has no consolidated volume (the HistData `vol` field is literally 0), so volume/dollar bars
are not meaningful. The available information clock is **tick count** (quote-update intensity). Tick
bars (gap-aware, weekend/rollover dropped) Gaussianize FX returns versus time bars on **all 8 majors**
(`tables/forex_bars.md`, `figures/fig_fx_kurtosis.png`): median excess kurtosis **2.98 → 1.06**,
|skew| 0.20 → 0.13. So LdP's principle holds for FX, instantiated as the tick bar.

## 7. Notes, opinions & extensions

- **Volume bars, not dollar bars, are the most robust Gaussianizer** in our data — marginally best in
  crypto (1.48) and clearly best in equities-RTH (3.35 vs dollar 5.58). LdP prefers dollar bars for
  their level/share-count invariance (a robustness argument, not a tail argument); on pure return
  normality, volume bars edge them out. For a *single* default across markets we'd reach for **volume
  bars**, keeping dollar bars where cross-instrument or long-horizon comparability matters.
- **Residual equity tails are fatter than crypto's even after info-bar sampling** (3.4–5.6 vs ~1.5).
  Intraday equity jumps (news, auctions) are not fully tamed by activity-clocking — a hook for a
  jump-robust labeling/feature treatment downstream.
- **Downstream implication:** every later project (labeling, features, models) should be built on
  information bars, per-market: dollar/volume bars for crypto & equities (within-session), tick bars
  for FX. Project 1 is the substrate for the rest of the program.

## 8. Limitations & reproducibility

- Sample is 2022–2024 for crypto/forex (Binance/HistData availability); equities ETFs span much
  longer. Equities are **ETFs, not single stocks** (single-stock Lean data is available for a breadth
  extension). Forex is **8 majors** and uses tick-count, not true volume.
- HistData FX timestamps are US-Eastern; we treat the stream as continuous within the trading week
  and drop >2h gaps. Crypto is 24/7 (no session handling needed).
- Statistical study only — no trading costs (none apply to a sampling-scheme comparison).
- **Rerun:** `python3 scripts/run_bars_study_clean.py` (crypto), `run_equities_bars.py` (equities),
  `run_forex_bars.py` (forex), `run_granularity_study.py` (granularity), then `make_final_figure.py`.
  Data fetchers: `lib/fetch_1m.py` (crypto), `lib/fetch_fx_histdata.py` (forex); equities ETFs from
  the Algoseek local cache.

## 9. Paper-worthiness

A strong **reproduction-plus-extension**, multi-market. The two findings that go beyond LdP — the
**granularity-dependence** of the advantage and the **session-handling requirement for equities**
(with the clean before/after) — are the publishable contributions, especially paired with the
data-quality cautionary result. As a standalone paper it needs one more leg: a **downstream
predictive test** showing models/strategies built on information bars beat time-bar baselines
out-of-sample on deflated metrics (Project 0 harness + a labeling project). On its own it is a
rigorous, honest "how to sample three markets" study and an ideal opening chapter of a broader paper.
