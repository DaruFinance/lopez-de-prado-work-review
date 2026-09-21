# Meta-Strategy Organization: the research assembly line, tested on its own corpus

> Article: [daru.finance/research-review/lopez-de-prado/meta-strategy-organization](https://daru.finance/research-review/lopez-de-prado/meta-strategy-organization)

This is the capstone study of the methods-review program. The other studies each
reproduce one technique from the López de Prado canon on real, multi-market,
realistically costed data. This study turns the lens on the program itself and
asks the organizational question that frames the whole canon: does running
quantitative research like a disclosed assembly line, rather than as a lone
backtester, actually change the realized out-of-sample outcome?

The thesis under test is López de Prado's "meta-strategy" argument from
*Advances in Financial Machine Learning* (Chapter 1) and the multiple-testing and
deflation discipline that runs through the rest of the book. Research should be a
factory of specialized, separable stations with mandatory disclosure of every
trial, not a single craftsman ("Sisyphus") who discovers strategies by
backtesting, tweaking, and rerunning. The reason the lone path fails is
mathematical, not a matter of effort: the best of many silently searched
backtests looks good even when there is no skill, and only disclosing the trial
count lets you deflate the winner back to its true significance.

We test that empirically with four experiments on the program's own corpus of
real, after-cost strategy results: about 1.20 million strategy configurations
across 42 instruments spanning crypto, equities, and foreign exchange. Every
number below comes from a real experiment on that corpus. Where the result is
negative, it is reported as negative.

## The corpus and the realized-performance convention

The corpus is the program's per-strategy daily profit-and-loss, stored one row
per strategy per trading day. We verified that the daily PnL is the after-cost
(net) figure: summing the daily PnL of one strategy reproduces, to the last
decimal, the sum of that strategy's individual net trade PnL from the trade
ledger (a single equity strategy reconciled at negative 161.2232719 on both
sides, where the gross figure was a positive 537.5). All realized performance in
this study is therefore net of costs. No cell uses information from a day later
than the day it describes, so there is no look-ahead.

For each instrument we build a dense day-by-strategy matrix of net daily PnL on
the instrument's own trading calendar. The build is a scatter-add of sparse
(day, strategy, PnL) triples into the dense matrix; that hot loop is written
twice, as a NumPy reference and as a compiled kernel, and the two agree to the
last bit (maximum absolute difference 0.0 on both random and real data). The
largest instrument matrix is about 50,700 strategies by 2,085 days.

## Experiment 1: Sisyphus versus the disclosed assembly line

**What we ran.** For every instrument we rolled a walk-forward of one-year
in-sample windows followed by one-quarter out-of-sample windows, stepping
quarterly, for 1,208 windows across 39 instruments with enough history. In each
window we ran two competing research processes on the same candidate pool and
recorded the realized out-of-sample result of each.

The Sisyphus process picks, each window, the single strategy with the best
in-sample Sharpe, discloses nothing, deflates nothing, and deploys that one
strategy through the next quarter. The assembly-line process takes the same pool
but gates it: it computes the False Strategy Theorem null for the full disclosed
number of trials searched that window, then keeps only strategies whose Deflated
Sharpe Ratio clears the significance bar, and allocates the survivors by
Hierarchical Risk Parity (reusing the allocator from the portfolio-construction
study); if nothing clears the bar it holds cash. We measured realized
out-of-sample Sharpe, the in-sample to out-of-sample decay (the winner's curse),
the out-of-sample hit rate, the maximum drawdown, and the assembly line's deploy
rate. The per-strategy in-sample Sharpe over a window is the hot loop; it is a
compiled, parallelized kernel verified bit-identical against NumPy (maximum
absolute difference 0.0).

**Headline numbers.** The lone backtester's picks had a median in-sample Sharpe
of 3.09 annualized and a pooled realized out-of-sample Sharpe of negative 0.02
(bootstrap 95 percent band negative 0.21 to positive 0.18). The mean in-sample
to out-of-sample decay was 2.61 annualized Sharpe: the winner's curse erases the
entire apparent edge. The realized out-of-sample Sharpe was negative for 82
percent of instruments, and the median out-of-sample hit rate was 0.40, worse
than a coin flip. On the worst instruments the lone pick had an in-sample Sharpe
above 3 and a realized out-of-sample Sharpe below negative 1.5.

The disclosed assembly line deployed nothing. In every one of the 1,208 windows,
the best single strategy's Deflated Sharpe Ratio against the disclosed-trial null
was effectively zero (the per-window maximum we observed was 0.0001). Relaxing
the gate from 0.95 down to 0.60 changed nothing, because no single strategy ever
came close. The deploy rate was 0.0 at both gates.

**Verdict.** The Sisyphus result is exactly the trap the thesis predicts: a
strong in-sample number that does not survive contact with the next quarter, and
on average no edge at all. The assembly line's refusal to deploy any single
strategy is not a failure of the discipline; it is the discipline correctly
reporting that no single backtest survives honest deflation once you admit how
many were tried. That refusal sets up the real question, which is what López de
Prado actually proposes instead: not one great strategy, but many weak ones
combined.

## Experiment 2: the meta-strategy portfolio

**What we ran.** López de Prado's constructive answer is to combine many
weakly-correlated bets into a diversified sleeve and to deflate the sleeve, not
each part. For every instrument and window we built a diversified shortlist by
taking the strongest in-sample strategies and then greedily dropping
near-duplicates so that the surviving bets were weakly correlated. We allocated
across the shortlist four ways (equal weight, inverse variance, Hierarchical Risk
Parity, and Nested Clustered Optimization on a denoised covariance, all reusing
the portfolio-construction study's machinery) at three shortlist sizes, deployed
each sleeve out of sample, and then deflated the best realized sleeve against a
portfolio-level null whose trial count is the number of construction choices
searched. We also report a fixed, selection-free policy (Hierarchical Risk Parity
at the largest shortlist) so the headline is not itself a cherry-pick.

**Headline numbers.** The diversification was real: the median absolute
correlation inside the shortlists fell to 0.11, and the effective number of
independent bets rose to about 17. But on this after-cost corpus, combining weak
bets mostly combined noise. The pooled fixed-policy sleeve had a realized
out-of-sample Sharpe of negative 0.64, and only 3 of 39 instruments cleared the
0.95 Deflated Sharpe bar at the portfolio level; the median portfolio Deflated
Sharpe was essentially zero.

The three that cleared are the broad equity index funds: the Nasdaq-100 proxy at
a realized out-of-sample Sharpe of 2.50 and Deflated Sharpe of 1.00, the S&P-500
proxy at 1.96, and the small-cap proxy at 0.85. Every crypto sleeve, every
foreign-exchange sleeve, and every sector or volatility equity sleeve had a
negative realized out-of-sample Sharpe and a portfolio Deflated Sharpe at or near
zero.

**Verdict.** Diversification does what it says on the structure: it lowers
correlation and raises the effective bet count. Whether that buys deflated edge
depends entirely on whether the underlying market carries persistent structure
after costs. On broad equity indices it does, and a diversified, deflated sleeve
clears the bar that no single strategy could. On the after-cost crypto and
foreign-exchange corpus it does not, and the honest answer is that a portfolio of
noise is still noise. This is the same message as the rest of the program,
sharpened: the edge that survives is the market, not the search.

## Experiment 3: program-level probability of backtest overfitting

**What we ran.** The Probability of Backtest Overfitting asks how often the
best in-sample strategy lands in the bottom half out of sample under
Combinatorially-Symmetric Cross-Validation. A value near 0.5 means in-sample
ranking carries no out-of-sample information. We ran the cross-validation on the
real net-daily-PnL matrix of each instrument, sampling up to 1,500
activity-filtered strategies per instrument with a fixed seed, and pooled the
out-of-sample logits across the program.

**Headline numbers.** The program-wide probability of backtest overfitting was
0.21 (median per instrument 0.22). By market it was 0.34 for crypto, 0.11 for
foreign exchange, and 0.00 for equities, with a per-instrument range from 0.00 to
0.63. The ordering reproduces the earlier per-corpus study on the smaller
moving-average grid (crypto 0.28, foreign exchange 0.005, equity 0.00); the
larger and more diverse daily corpus here shows somewhat more overfitting room in
crypto and foreign exchange, which is what one expects when the search space
grows.

**Verdict.** Backtest overfitting is real and strongly market-dependent. Equity
index rankings are stable out of sample, crypto rankings are close to a coin
flip, and foreign exchange sits in between. The same diagnostic, applied
consistently, separates the markets cleanly and agrees with the rest of the
program.

## Experiment 4: program-scale expected maximum Sharpe versus the observed best

**What we ran.** The False Strategy Theorem says the expected maximum Sharpe of N
skill-less trials grows with N and with the dispersion of trial Sharpes. We
measured all three ingredients from the real corpus: the trial count, the
empirical dispersion of per-strategy Sharpes, and the effective number of
independent trials from the eigenvalue participation ratio of the real
strategy-correlation structure (computed on a bounded sample per instrument to
keep memory in check). We then compared the expected maximum to the observed best
Sharpe in the corpus.

**Headline numbers.** The program searched 985,570 eligible strategy
configurations (1.20 million in total). The real correlation structure is loose
(median absolute correlation 0.05), so the effective number of independent trials
is 186,139, about 19 percent of the nominal count. The observed best annualized
Sharpe across the entire corpus is 3.21, on the Nasdaq-100 proxy, which
independently reproduces the program's earlier best-of-corpus figure of about
3.21. The expected maximum Sharpe of purely skill-less trials at this scale is
5.61 under the nominal count and 5.22 under the effective count.

**Verdict.** The single most striking number in the study: the observed best
result of the entire program, 3.21, is below what pure selection on noise would
be expected to produce, 5.22 to 5.61, once the roughly one million trials are
disclosed and the correlation structure is accounted for. The curve crosses the
observed best at only about 200 trials. A reader shown only the best backtest in
the program would see an annualized Sharpe of 3.21 and be impressed; the False
Strategy Theorem says that with a search this large, a Sharpe of 3.21 is not even
keeping up with chance. This is the thesis in one figure.

## Experiment 5: the whole assembly line, chained and run on held-out time

The first four experiments each isolate one piece of the argument. Experiment 1
compares the selection rule, Experiment 2 the combination rule, Experiment 3 the
overfitting diagnostic, Experiment 4 the deflation arithmetic. The culminating
experiment puts every piece back together and runs the entire production line as
one disclosed system, end to end, on time it never saw during any tuning, so the
reader can watch the assembled discipline make a deploy decision and then live
with that decision out of sample.

**What we ran.** We chained nine stations into a single pipeline on a handful of
real crypto instruments: load one-minute base data; build dollar bars, which
sample on traded notional rather than the clock; label each event with the
triple-barrier method on full intrabar high and low, with the side set by a
causal moving-average crossover; weight the labels by their sample uniqueness, so
overlapping and therefore non-independent observations are not counted as
independent evidence; train a bagged-tree secondary model to predict the
probability that the primary side's bet is profitable, fitting it with purged and
embargoed cross-validation and the uniqueness weights; gate each bet by a
meta-label threshold, which can veto or size a bet but never flip its side; size
the surviving bets by the model's probability; allocate across the surviving
instrument sleeves by Hierarchical Risk Parity, reusing the portfolio-construction
study's allocator; and finally deflate the assembled track record against the
disclosed number of configurations searched, deploying a sleeve only if its
Deflated Sharpe Ratio clears the bar. Every tuning decision was made on the first
four fifths of each instrument's bars. The last fifth was held out and scored
once. Costs were charged at seven basis points per side on every turnover.

**Headline numbers.** The disclosed pipeline searched twenty-seven configurations
per instrument, one hundred and sixty-two in total, and deployed nothing. No
instrument's tuning Deflated Sharpe Ratio came close to the bar: the highest was
0.57 and most were below 0.05, so the deploy gate held cash on the entire
held-out window and the pipeline's realized out-of-time Sharpe was zero by
construction. The lone backtester, deploying its best in-sample pick on every
instrument with no deflation, earned a pooled realized out-of-time Sharpe of
negative 0.83 with a profit factor of 0.96, losing money after costs. Buy and
hold of the same instruments over the same window returned a Sharpe of 1.81 at a
profit factor of 1.06. Per instrument the lone pick's realized out-of-time Sharpe
ranged from positive 0.58 on one instrument down to negative 2.42 on another, the
familiar pattern of a number that looked tradeable in sample and did not hold.

**Verdict.** Assembled honestly and run forward once, the production line refuses
to deploy, because not one of the candidate configurations survives deflation
against the number of trials that were actually searched. That refusal is the
correct output, not a malfunction: on this after-cost window the only positive
realized Sharpe belonged to simply holding the instruments, the lone backtester's
selected pick lost money, and the disclosed gate is exactly the thing that stops a
desk from shipping that losing pick. The full chain reproduces, as a single
lived-through decision, the same conclusion the four isolated experiments reach
separately. The discipline is not what finds the edge; it is what keeps a search
this large from manufacturing one that is not there.

## Reconciliations

The program scorecard (built from every study's real result tables, and
reproduced here) reconciles the apparent tensions among the headline numbers.

The earlier statement that some per-instrument bests cleared a high Deflated
Sharpe bar and the program-best statement that the best Deflated Sharpe is below
0.03 are different cuts and both true. The per-instrument figure counts
walk-forward survivors deflated against that instrument's own trial count: 19 of
42 instruments in the trend-scanning study and 3 of 42 in the optimal-trading-
rules study cleared their per-instrument bar, and those are real local survivors.
The program-best figure deflates the single best result against the full
program-wide trial count, and against that much larger null it does not survive.
The same result can clear a small local null and fail a large global one; that is
precisely the disclosure effect the thesis is about.

The probability-of-overfitting numbers reconcile across corpus slices: the
smaller moving-average grid and the larger daily corpus give the same market
ordering, with the larger corpus showing slightly higher crypto and foreign-
exchange overfitting, consistent with a wider search.

The observed best of 3.21 in this study matches the program's prior best-of-
corpus aggregation, computed here independently on the full daily corpus with an
activity filter and full history.

Disclosure also has a constructive side, and the program shows it. The null holds
broadly across the naive and single-series regimes, but the gate is not a blanket
no: three disciplined regimes do clear deflation without leaning on a small local
null.

- **Cross-sectional ranking with a gradient-boosted model** clears the Deflated
  Sharpe bar on 10 of 10 holding horizons (probability of backtest overfitting
  0.10, in-sample to out-of-sample rank correlation 0.78, median out-of-sample
  profit factor about 1.12). This is the program's first deflation-surviving
  machine-learning edge, and the cross-sectional survival is not a single-model
  artefact: two further gradient-boosted-tree families clear most horizons on the
  same panel (23 of 30 tree family-by-horizon combinations in total), while
  single-series machine learning clears 0 of about 25,000. Detail in
  `projects/15_cross_sectional_ml`.
- **The Ornstein-Uhlenbeck optimal-trading-rule**, run on its true regime (the
  residual spread of a cointegrated pair rather than a single series), clears the
  Deflated Sharpe bar on 3 of 5 cointegrated pairs at a median profit factor of
  2.11 and beats a Bollinger-band control on 4 of 5. Detail in
  `projects/12_optimal_trading_rules`.
- **Large-universe hierarchical risk parity** shows a real out-of-sample variance
  advantage over equal weight that grows with breadth: the HRP-to-equal-weight
  variance ratio falls from about 0.87 at 25 names to about 0.45 at 200 names
  (smaller is a bigger advantage). Detail in `projects/13_portfolio_construction`.

Each of these *approaches* but does not *beat* the best static structural archetype
near profit factor 1.17, and the fully-disclosed end-to-end pipeline (Experiment 5)
still deploys nothing in a held-out window. So the honest reading is two-sided: the
null is real and broad, naive single-series search clears almost nothing, but
disciplined regimes (cross-sectional ranking; OU on its true mean-reverting regime;
large-universe risk parity) do clear the same deflated bar and match the static edge
without dominating it.

## Engineering and reproducibility

The four experiments run on the full corpus on a desktop. The matrix build, the
per-strategy Sharpe loop, and the expected-maximum-Sharpe loop are the hot paths;
each is implemented as a NumPy reference and a compiled, parallelized kernel and
verified bit-identical (maximum absolute difference 0.0 in every case, with the
analytic expected-maximum-Sharpe formula agreeing with a Monte-Carlo of
skill-less strategies to within 0.0009 per-observation Sharpe). The compute runs
four-way parallel across instruments, with linear-algebra threads pinned to one
each so the worker count is exactly four. Memory is held in check by processing
one instrument at a time, sampling strategies for the correlation and
cross-validation steps, and never materializing a full strategy-by-strategy
correlation matrix; peak resident memory was about 9.7 gigabytes per worker on
the largest instruments. The cross-validation uses Combinatorially-Symmetric
Cross-Validation, the deflation uses the False Strategy Theorem null with the
disclosed trial count, and every realized stream is purged of look-ahead by
construction.

## Honest limitations

The realized out-of-sample streams are pooled in unit-risk terms (each strategy's
PnL scaled by its in-sample standard deviation, with the same divisor applied out
of sample, so there is no look-ahead) because the raw dollar PnL of different
strategies is not on a comparable scale; this makes pooled Sharpe meaningful but
means the pooled equity curves are in risk units, not currency. The assembly-line
deploy rate of zero is a strong result and depends on insisting that a single
strategy clear the disclosed-trial null; a desk that allocated across a
pre-committed basket without re-selecting per window would look more like
Experiment 2 than the all-cash Experiment 1, which is why both are reported. The
effective number of independent trials is measured on a per-instrument sample and
summed across instruments, treating instruments as independent blocks, which is a
conservative-low estimate of the true cross-instrument redundancy. The corpus is
a fixed snapshot of one research program's strategy families and does not claim to
be the universe of all possible strategies.

## What the study shows

Across four independent experiments on a million-trial, multi-market, after-cost
corpus, the organizational thesis holds, and the verdict is two-sided. On the
null side: the lone backtester's best in-sample pick decays by 2.61 Sharpe and
earns nothing out of sample; no single naive strategy survives honest deflation
against the disclosed trial count; diversification buys deflated edge only where
the market carries persistent structure, namely broad equity indices, and not on
after-cost crypto or foreign exchange; backtest overfitting is real and strongest
in crypto; and the program's own best result, impressive in isolation at a 3.21
Sharpe, falls below the skill-less expectation once you admit how much was
searched.

But the null is not absolute. Three disciplined regimes clear the same deflated
bar where the naive search does not: cross-sectional ranking with a
gradient-boosted model (10 of 10 horizons, the program's first deflation-surviving
machine-learning edge), the Ornstein-Uhlenbeck rule on its true mean-reverting
spread (3 of 5 cointegrated pairs, median profit factor 2.11), and large-universe
risk parity (an out-of-sample variance advantage over equal weight that grows with
breadth, about 0.87 at 25 names to about 0.45 at 200). Each approaches but does not
beat the best static structural archetype near profit factor 1.17, and the
fully-disclosed end-to-end pipeline still deploys nothing in a held-out window. So
disciplined work matches the static edge without dominating it, while the naive
Sisyphus search clears nothing at all. The discipline that the assembly line
enforces, disclosing every trial and deflating the winner, is not bureaucratic
caution; it is the only thing standing between a tradeable result and a number that
looks good because it was chosen from a million, and it is also what lets the genuine
disciplined edges be recognized as such instead of buried with the noise.
