# Project 09, Ensembles (Bagging vs Boosting) & Feature Importance (MDI / MDA / clustered-MDA)

> Article: [daru.finance/research-review/lopez-de-prado/ensembles-and-feature-importance](https://daru.finance/research-review/lopez-de-prado/ensembles-and-feature-importance)

*López de Prado, Advances in Financial Machine Learning (AFML) Ch.6 (ensembles),
Ch.8 (feature importance), Ch.9 (hyper-parameter tuning); Machine Learning for
Asset Managers (ML4AM) Ch.6 (clustered feature importance).*

Headline metric: **Deflated Sharpe Ratio (DSR)**, net of realistic costs, on
purged-CV out-of-fold bets across **40 instruments in 3 markets** (27 crypto
perp dollar-bars, 5 US-equity ETF dollar-bars, 8 FX tick-bars).

---

## 1. Question

López de Prado makes two sharp, testable claims that this project puts to a
multi-market, costed, leakage-controlled test.

1. **Bagging generalizes better than boosting on financial data (AFML 6.3).**
   Financial labels are noisy and serially overlapping; boosting chases that
   noise by re-weighting hard (often mislabeled) cases, so it overfits, whereas
   bagging averages decorrelated trees and is more robust. *Does a RandomForest
   built the AFML way actually show a smaller in-sample-to-out-of-sample
   generalization gap than HistGradientBoosting, and does the better
   generalization translate into a higher Deflated Sharpe Ratio?*

2. **MDI is a biased importance measure; use out-of-sample MDA, and cluster
   correlated features (AFML 8.3/8.5, ML4AM 6).** Mean-Decrease-Impurity (MDI)
   is in-sample and is mechanically inflated for high-cardinality / correlated
   features (substitution effect). Mean-Decrease-Accuracy (MDA, here scored by
   *log-loss*, not accuracy) is out-of-sample and permutation-based; clustering
   correlated features and permuting whole clusters removes the substitution
   artefact. *How large is MDI's substitution bias versus MDA and clustered-MDA,
   and, the part rarely tested, how **stable** is the selected feature set
   across CPCV paths for each method?*

A third, methodological control runs throughout: AFML 9.4 insists you tune a
classifier by a **probabilistic** loss (negative log-loss), not accuracy. We
tune every model both ways and compare the resulting overfit gaps.

---

## 2. Data

| Market | Instruments | Source | Bars | Cost model |
|---|---|---|---|---|
| Crypto (USDT perps) | 27 (BTC, ETH, SOL, … ) | Binance 1-minute | 20k dollar-bars/inst | flat 7 bp/side |
| US equities (ETFs) | 5 of 7 (SPY, QQQ, IWM, XLK, XLE) | AlgoSeek 1-minute RTH | 20k dollar-bars/inst | `lib/realism` time-of-day half-spread + commission (≈2 bp/side) |
| Forex (spot) | 8 majors (EURUSD, USDJPY, …) | HistData 1-minute | tick(count)-bars | `lib/realism` UTC time-of-day half-spread + swap (≈1 bp/side) |

All bars are **information-driven** (dollar bars for crypto/equities, count
bars for FX) per AFML 2.3, they sample on activity, not clock time, so the
return series is closer to IID. Equity/FX frictions are the **causal**,
time-of-day-scheduled half-spreads + commission/swap from `lib/realism.py`
(known ex-ante from the timestamp; no clamping of weekend/overnight gaps).

Two ETFs (XLF, XLV) were dropped by a data-driven gate, not by hand: after the
triple-barrier labeling and 2 bp cost they produced fewer than 50 net-profitable
bets (the minimum-class floor). This is itself a finding, see §6.

---

## 3. Method

**Labeled task (fixed structural shape, this is a model/importance study, not a
label sweep).** A primary EMA(20/60) crossover fixes the side; a triple-barrier
(profit-take 1.5σ, stop 1.0σ, max-hold 50 bars, σ = causal EWMA vol) scanned on
**full intrabar OHLC** (not close-only) gives the realised outcome; the
meta-label is `y = 1[net P&L > 0]` after the realistic round-trip cost. Causal
features only: `side, vol, ma_gap, mom3/6/12, rsi, vol_ratio, ofi, range_atr`
(10 features). Labeling/feature code is reused unmodified from Project 03
(`tbm.py`).

**Two models.**
- *Bagging*, `RandomForestClassifier` built the AFML 4.5/6.2/6.3 way: low
  `max_features ∈ {1,2,3}` (decorrelate trees), `min_weight_fraction_leaf ∈
  {0, 0.05}` (regularize leaves under weighting), **uniqueness sample-weights**
  (down-weight overlapping labels, AFML 4.4), and `max_samples = average label
  uniqueness` (the sequential-bootstrap analogue, AFML 4.5.2, its true
  sequential-bootstrap kernel is implemented and verified, see §7).
- *Boosting*, `HistGradientBoostingClassifier`, grid over `max_iter ∈ {150,300}`,
  `learning_rate ∈ {0.05,0.1}`, `max_leaf_nodes ∈ {15,31}`, `l2 = 1.0`.

**Purging.** All cross-validation is **purged k-fold + embargo**
(`lib/overfit.purged_kfold_splits`, AFML 7) with the label span mapped from bar
space to event space (an event's label lives `hold` bars and overlaps
`ceil(hold/gap)` later events). Train rows whose label window overlaps a test
fold (plus embargo) are dropped. Importance **stability** uses **CPCV**
(`cpcv_splits`, C(6,2)=15 paths, AFML 12).

**Tuning.** Each model is hyper-tuned over its purged grid scored by
**negative log-loss** (headline, AFML 9.4) and, as a control, by **accuracy**.

**Bet & DSR.** The purged out-of-fold `P(profit)` sizes the bet (`size = p` when
`p ≥ 0.5`); costed P&L is spread over the holding period into a per-bar return
series. The selected model's Sharpe is **deflated** (`deflated_sharpe_ratio`,
Bailey-LdP) against the dispersion of the *pooled RF+HGB grid* bet Sharpes,
i.e. the full multiple-testing search the researcher actually ran is the trial
set. DSR ≈ P(true Sharpe > 0 after accounting for selection); the conventional
publishable bar is **DSR > 0.95**.

**Importance.** MDI from a single full-sample RF fit (the biased in-sample
baseline); MDA as purged out-of-fold permutation **scored by log-loss**;
clustered-MDA clusters features by |correlation| (single/average linkage on
`1−|ρ|`) and permutes whole clusters. Substitution bias is measured as the
Spearman ρ between a method's importance and each feature's **mean |correlation|**
to the others, high ρ means importance is leaking onto correlated features.
Selection stability is the mean pairwise **Jaccard** overlap of each method's
top-3 set across CPCV paths, benchmarked against a Monte-Carlo **random-selection
baseline** (the Jaccard you'd get picking 3 of 10 features at random).

---

## 4. Headline results

### 4a. Bagging vs boosting, generalization (paired, n = 40)

| | bagging (RF) | boosting (HGB) | paired test |
|---|---|---|---|
| **IS→OOS log-loss gap (NLL-tuned)** | **0.114** | 0.539 | RF < HGB in **100%** of instruments, Wilcoxon **p = 1.8e-12** |
| IS→OOS log-loss gap (ACC-tuned) | 0.132 | 0.583 | RF < HGB in 100%, p = 1.8e-12 |
| median HGB/RF gap ratio |, |, | **4.8×** (boosting overfits ~5× more) |
| IS log-loss (crypto) | 0.57 | **0.25** | boosting memorizes the train set |
| OOS log-loss (crypto) | **0.68** | 0.81 | … and pays for it out-of-sample |

**LdP's first claim holds unambiguously.** Bagging's generalization gap is ~5×
smaller than boosting's, on every single instrument across all three markets.
The mechanism is exactly the one in AFML 6.3: HistGradientBoosting drives its
*in-sample* log-loss down to ~0.25 (well below the coin-flip ln2 ≈ 0.69) by
fitting the noisy, overlapping labels, but its *out-of-sample* log-loss is the
**worst** of the two models (~0.81 > ln2, literally worse than a coin flip).
The bagged forest keeps IS and OOS log-loss close (0.57 → 0.68) and is the only
one of the two whose OOS log-loss stays near coin-flip rather than blowing past
it. *(See `fig1`, `fig5`.)*

### 4b. … but it does NOT translate into edge (the honest part)

| | bagging (RF) | boosting (HGB) | best-of-two |
|---|---|---|---|
| median OOS DSR | 0.005 | 0.003 | 0.025 |
| median annualized Sharpe | −0.67 | −0.82 |, |
| instruments with DSR > 0.95 | **0 / 40** | **0 / 40** | **0 / 40** |
| instruments with DSR > 0.90 |, |, | 1 / 40 (USDJPY, HGB 0.92) |
| RF vs HGB DSR (paired) |, |, | Wilcoxon **p = 0.74 (n.s.)** |

This is the crucial nuance LdP's claim does **not** cover: *better generalization
is necessary, not sufficient, for edge.* On this EMA-crossover meta-labeling task
**neither** ensemble produces a deflated Sharpe anywhere near the 0.95 bar, the
median bet Sharpe is **negative** for both, and once you deflate against the
14-config search the DSR collapses to ~0. Bagging wins the *generalization*
contest decisively and the *DSR* contest not at all (paired p = 0.74). The single
near-miss is USDJPY (HGB DSR 0.92), and EURUSD (RF DSR 0.56), both forex, where
the costed task is least adversarial.

### 4c. Feature importance, MDI bias vs MDA vs clustered-MDA

| | MDI | MDA | clustered-MDA |
|---|---|---|---|
| substitution bias ρ(importance, mean \|corr\|), median | **0.485** | **0.073** | 0.517 |
| vs MDI bias (paired) |, | lower in 88%, Wilcoxon **p = 1.2e-6** | n.s. (p = 0.22) |
| top-3 selection stability (Jaccard across CPCV paths), median | 0.582 | **0.263** | 0.463 |
| random-selection baseline |, | 0.201 |, |

**LdP's second claim holds for MDA, with an important twist on clustering.**
MDI's importance is strongly rank-correlated with how correlated a feature is to
the rest (ρ ≈ 0.49, highest in forex at 0.67), exactly the substitution artefact
AFML 8.3 warns about, the momentum block (mom3/6/12) and the vol block split the
impurity credit and inflate each other. **MDA, the out-of-fold permutation
measure, shows essentially no such bias (ρ ≈ 0.07; lower than MDI on 88% of
instruments, p = 1.2e-6).** `fig3` shows the qualitative payoff: features MDI
ranks highly (e.g. `mom6`, `mom12`, `vol` in equities/forex) get **negative** MDA
, permuting them *improves* OOS log-loss, i.e. the model was over-relying on
noise. MDI keeps them; MDA flags them as harmful.

The twist (a genuine, honest negative result): **clustered-MDA does *not* remove
the substitution bias on this feature set**, its per-feature-expanded importance
still correlates with |corr| at ρ ≈ 0.52, statistically indistinguishable from
MDI (p = 0.22). With only 10 features collapsing into ~5 clusters, the
correlated features land in the *same* cluster, so the whole-cluster permutation
still attributes large importance to that (correlated) cluster, and expanding it
back to features re-introduces the |corr| association. Clustering's payoff here
is **stability, not de-biasing** (see §5b). The lesson: with a small,
moderately-correlated feature set, *MDA, not clustered-MDA, is the
de-biasing tool*; clustering matters more when there are many tightly-correlated
features to absorb.

### 4d. Tuning objective, log-loss vs accuracy (AFML 9.4 control)

| tuning objective | mean overfit gap | paired test |
|---|---|---|
| negative log-loss (headline) | **0.331** |, |
| accuracy (control) | 0.421 | Wilcoxon **p = 8.3e-6** |

**LdP's tuning rule holds.** Tuning by log-loss yields a significantly *smaller*
overfit gap than tuning by accuracy on the pooled-mean gap (0.331 vs 0.421,
p = 8.3e-6; `fig2`/`fig7`: most instruments sit below the diagonal). The effect
holds for **both** model families when measured per-model: RF gap NLL < ACC
(p = 4.4e-4) and HGB gap NLL < ACC (p = 1.3e-4). The two objectives disagree on
the chosen config often enough to matter, NLL and accuracy pick the **same RF
config in 60%** of instruments and the **same HGB config in 52%**, i.e. roughly
half the time the objective changes the selected model, and when it does, the
log-loss choice generalizes better. (The earlier intuition that RF is wholly
objective-insensitive was a single-instrument artefact; at panel scale both
models are sensitive, and both benefit from log-loss tuning.)

---

## 5. Deepening, what the headline run did not show

*(The deepening run `scripts/deepen.py` recomputes the full multi-market panel
storing the extra fields below; outputs in `tables/deepen_*` and
`figures/fig5-7`.)*

### 5a. Does bagging generalize better *on the noisiest markets*?

Yes, and the IS/OOS *levels* (not just the gap) tell the mechanism precisely:

| market | RF IS ll | RF OOS ll | HGB IS ll | HGB OOS ll |
|---|---|---|---|---|
| crypto | 0.569 | 0.680 | **0.255** | **0.809** |
| equities | 0.134 | 0.557 | 0.153 | 0.659 |
| forex | 0.569 | 0.677 | **0.277** | **0.808** |

On the two genuinely-noisy, high-base-rate markets (crypto, forex) boosting
drives IS log-loss to ~0.25-0.28, far below the coin-flip ceiling ln2 ≈ 0.693,
and its OOS log-loss is ~0.81, *worse than a coin flip*. The bagged forest holds
IS at ~0.57 (it does not memorize) and OOS at ~0.68 (still ≤ coin-flip). This is
the AFML 6.3 story made quantitative: **on noisy financial labels, boosting's
extra capacity is spent fitting noise and is net-negative out-of-sample, exactly
where bagging's variance-reduction is most valuable.** Equities are the exception
*for a different reason*, their IS log-loss is low for *both* models (~0.13-0.15)
because the label is class-imbalanced (base rate ~0.17), so even a bagged forest
trivially predicts the majority class in-sample; the equity gap is partly a
class-imbalance artefact, not pure model complexity (see §6).

### 5b. MDI vs MDA vs clustered-MDA, bias, and the stability question

**Bias (median ρ with feature mean |corr|):** MDI 0.485, MDA 0.073, clustered-MDA
0.517. Only **MDA** is bias-free (lower than MDI on 88% of instruments,
p = 1.2e-6); clustered-MDA is *not* an improvement over MDI on this 10-feature set
(p = 0.22), see the §4c twist. `fig6` (left) shows this directly: the MDA box
straddles zero while MDI and clustered-MDA both center near +0.5.

**Stability (mean top-3 Jaccard across the 15 CPCV paths):** MDI 0.582 >
clustered-MDA 0.463 > MDA 0.263 > random 0.201. Three honest findings:
- **MDA's selection is real but fragile.** It beats random selection on **98% of
  instruments** (paired Wilcoxon p = 5.2e-8, median z ≈ 3.5σ above the random
  baseline), so the OOS-permutation signal is *not* noise, but at Jaccard 0.26
  it is *barely* above the 0.20 you'd get drawing 3 of 10 features at random. The
  out-of-sample feature ranking is genuinely informative yet only weakly stable;
  one should not over-interpret "the top features" on this task.
- **Clustering buys stability, not de-biasing.** Clustered-MDA is *significantly*
  more stable than per-feature MDA (0.46 vs 0.26, p = 1.8e-12, more stable on
  100% of instruments), collapsing correlated features into a cluster removes
  the path-to-path coin-flipping *between* substitutable features. This is the
  ML4AM 6 rationale, confirmed: cluster the substitutes and the *cluster-level*
  selection is far more reproducible, even though the per-feature bias metric does
  not move.
- **MDI looks "most stable" but for the wrong reason.** Its 0.58 stability is
  inflated by the same substitution bias that makes it untrustworthy, it
  *consistently* over-ranks the same correlated block every path, which reads as
  stability but is the artefact, not signal.

### 5c. Log-loss vs accuracy, config sensitivity

At panel scale the two objectives select the **same config 60% of the time for
RF and 52% for HGB**, so the choice of tuning metric flips the selected model on
~40-48% of instruments. When it flips, the log-loss-selected config has the
smaller overfit gap for *both* model families (RF p = 4.4e-4, HGB p = 1.3e-4).
Interestingly the accuracy-tuned RF has a *slightly less negative* median annual
Sharpe (−0.53 vs −0.67, p = 6.4e-4), i.e. accuracy tuning occasionally lucks
into a marginally better *bet* even while generalizing worse on log-loss; but
both are firmly negative and neither is tradeable, so this is a curiosity, not a
counter-argument to AFML 9.4.

---

## 6. Honest caveats & negative findings

- **No edge.** The DSR headline is the honest verdict: **0/40 instruments clear
  DSR > 0.95** on either ensemble. This project is a clean *methodology*
  demonstration (LdP's three claims about bagging, MDI, and tuning all replicate),
  **not** a tradeable strategy. The EMA-crossover meta-labeling task is a vehicle,
  and on costed, purged, deflated evaluation it has no surviving edge.
- **Selection is barely better than random.** MDA's top-3 set is stable across
  CPCV paths at Jaccard ≈ 0.263 vs a random baseline of ≈ 0.201, *statistically*
  above chance (98% of instruments, p = 5.2e-8, §5b) but *practically* close to
  it. Feature selection on this task is fragile; one should not over-interpret
  "the top features." Clustering raises stability to ≈ 0.46 but does not fix the
  per-feature substitution bias on this small feature set (§4c).
- **Equities have a near-degenerate label.** Equity intraday EMA-crossover bets
  clear the cost+barrier profit threshold only ~17-23% of the time (base rate),
  and two ETFs (XLF, XLV) failed the 50-positive-trade floor outright. The equity
  *IS* log-loss is artificially low (~0.13) because the classifier can trivially
  predict the dominant "loss" class, which inflates the IS→OOS gap for reasons
  partly unrelated to model complexity. Equity results are reported but are the
  weakest leg.
- **`max_samples = avg-uniqueness` is an approximation.** sklearn's RF cannot use
  the true sequential bootstrap; we set `max_samples` to the average label
  uniqueness as its first-order analogue (the true sequential-bootstrap kernel is
  implemented and verified but not wired into sklearn's bagging). This is faithful
  to AFML's *intent* but is not the exact sequential bootstrap.
- **DSR trial count is modest (14 configs).** The deflation uses the pooled
  RF+HGB grid as the trial set. A larger grid would deflate harder; our grids are
  deliberately small (the point is the model comparison, not a grid blowout), so
  the DSR is, if anything, *generous*, and it is still ~0.

---

## 7. Reproducibility & engineering

- **Causal, costed, purged.** Every feature/label uses only data up to bar `t`;
  every bet is net of realistic per-side cost; every fold is purged + embargoed.
- **Numba parity (verified bit-identical).** Only the non-sklearn hot loops are
  JIT'd, the sequential-bootstrap draw (AFML 4.5.2) and the co-event count +
  average label uniqueness (AFML 4.4). Both are checked against independent
  pure-Python references in `--verify`: max|Δ| uniqueness = 5.6e-16, count = 0,
  bootstrap index = 0 (bit-identical). sklearn fits are deliberately **not**
  numba'd (they are ~89% of runtime per the profile; the right levers are grid
  size and instrument-level parallelism).
- **Determinism.** `random_state = 0` throughout; BLAS/OMP pinned to 1 thread per
  process so results are independent of `--jobs`. The deepening run reproduced the
  headline run's per-instrument gaps exactly (e.g. APEUSDT RF gap 0.123).
- **Commands.**
  ```bash
  python3 scripts/run_ensembles_importance.py --verify   # numba parity
  python3 scripts/run_ensembles_importance.py --smoke    # 3-inst sanity
  ./run_full.sh                                          # full panel (--jobs 16)
  python3 scripts/deepen.py --jobs 4                      # deepening panel
  ```

---

## 8. Verdict & paper-worthiness

**Scientifically clean, three-for-three on LdP's claims, zero tradeable edge.**

This is a textbook-faithful, multi-market replication that confirms all three of
López de Prado's pedagogical claims on real, costed, leakage-controlled data:

1. **Bagging generalizes far better than boosting** on noisy financial labels,
   a ~5× smaller IS→OOS gap, on 100% of 40 instruments, p = 1.8e-12. Boosting
   literally drives OOS log-loss past coin-flip.
2. **MDI is substitution-biased; out-of-sample MDA is not**, ρ 0.49 vs 0.07,
   p = 1.2e-6, with the striking qualitative result that several MDI-favored
   features have *negative* MDA (they hurt OOS). A refinement of LdP's claim:
   on this 10-feature set, *clustered*-MDA buys selection **stability**
   (0.46 vs 0.26 Jaccard, p = 1.8e-12) but **not** de-biasing, MDA alone is the
   de-biasing tool here.
3. **Tuning by log-loss beats tuning by accuracy**, smaller overfit gap for both
   model families (RF p = 4.4e-4, HGB p = 1.3e-4; pooled p = 8.3e-6).

**Paper-worthiness: a strong methods/replication contribution, not an alpha
paper.** The honest DSR verdict, **0/40 above 0.95**, is the headline integrity
of the work: it is publishable precisely *because* it separates "better
generalization / less bias" (which all replicate cleanly and are useful for any
practitioner) from "edge" (which does not appear). A credible venue would be a
methodological / reproducibility note ("Do López de Prado's ensemble and
feature-importance prescriptions replicate out-of-the-box across crypto, equities
and FX?") with the cross-market generality and the DSR-honesty as its selling
points. As a standalone alpha claim it has nothing; as a disciplined negative
result on edge plus a positive result on methodology, it is sound and citable.
The two genuinely novel angles worth foregrounding are (i) the **cross-market**
test of the bagging-beats-boosting claim and (ii) the **stability of the selected
feature set across CPCV paths**, a question AFML poses but rarely quantifies, and
where the answer here ("barely above random") is a useful cautionary data point.

### 5b-ii. Clustered importance, measured at the cluster level (the actual fix)

The §4c twist (clustered-MDA, *expanded back to per-feature*, still tracks
feature correlation) is real, but it judges clustering on the wrong axis. The
López de Prado prescription (ML4AM 6) is to read importance **at the cluster
level**, not to re-expand it to features. Measured that way, the remedy does
exactly what it is supposed to. On the crypto representative (BTCUSDT, 351 events,
10 features, same triple-barrier task and RF recipe), Ward clustering on the
1 minus absolute-correlation distance yields 6 clusters, one of which is a
4-feature correlated block (side, mom6, mom12, rsi). We then compute
clustered-MDI (sum of within-cluster impurity) and clustered-MDA (permute the
whole block at once under purged CV) and compare to the flat per-feature numbers.

**De-dilution.** This is the substitution effect made concrete. The three
momentum and oscillator members of the correlated block each look only moderately
important on their own under flat MDI (rsi 0.148, mom12 0.119, mom6 0.106), but
they are substitutes that split each other's credit. Scored as one cluster the
block carries 0.385 of total importance, the single largest contributor by a wide
margin, more than 2.5 times its best individual member. A reader scanning the flat
per-feature ranking would conclude no single feature dominates; the cluster view
shows that a correlated momentum block, taken together, is the dominant driver.
The concentration measured by the Gini of the importance vector rises from 0.205
(flat per-feature) to 0.337 (per-cluster) as the diluted credit is reassembled.

**Stability.** Reading importance per cluster is also markedly steadier across the
purged folds (mean pairwise Spearman of the importance ranking across folds):

| method | rank stability (mean pairwise Spearman across purged folds) |
|---|---|
| flat MDI | 0.892 |
| clustered MDI | **0.970** |
| flat MDA | 0.121 |
| clustered MDA | **0.303** |

Clustering raises rank stability for both measures: MDI 0.892 to 0.970, and the
out-of-sample permutation measure MDA from a fragile 0.121 to 0.303 (a 2.5 times
improvement). The mechanism is the same one that drove the cluster-level Jaccard
result in §5b: the per-feature ranking coin-flips between substitutable features
from fold to fold, and collapsing those substitutes into a cluster removes that
instability while preserving the signal. The practical takeaway is the
constructive complement to the §4c caveat: judge clustering by the cluster-level
importance and stability it was designed to deliver, not by re-expanding it to
features, and a correlated block that looks unremarkable feature by feature is
correctly surfaced as the dominant, and most reproducible, driver. Per-cluster
numbers and the figure are in `tables/clustered_importance.csv` /
`clustered_importance.md` and `figures/fig_clustered_importance.png`.

