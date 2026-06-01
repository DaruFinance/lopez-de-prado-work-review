# Meta-Strategy Organization: the research assembly line and mandatory disclosure

Reproduction and synthesis of Marcos Lopez de Prado's organizational thesis from
*Advances in Financial Machine Learning* (Chapter 1, "the assembly line") and the
multiple-testing and disclosure discipline that runs through the rest of the book.
Unlike the other studies in this program, this is a methodological and
organizational result rather than a single backtest. Its job is to state the
framework and then make it concrete with the program's own honest evidence.

## The claim

Lopez de Prado argues that quantitative research must be run like a factory
assembly line of specialised, separable stations, not like a lone craftsman.
The stations are roughly: data curators turn raw feeds into clean, causal bars;
feature analysts build informative non-leaking features; strategists attach
labels and sides (the bet); backtesters and the validation station check the bet
under purged cross-validation and deflate it for the number of trials; a
deployment station handles sizing and execution; and a portfolio oversight
station allocates across strategies. Two principles tie the line together. First,
**separability**: each station is a distinct skill, and no single person should do
everything, because the person who designs the bet and also judges it will judge
it kindly. Second, **mandatory disclosure**: every trial from every station is
logged, so that when a winner is finally reported, the validation station knows
the true number of trials N that produced it and can deflate the result
accordingly.

He names the opposite pattern as a trap. The lone "Sisyphus" quant who discovers
strategies by repeatedly backtesting one idea, tweaking, and rerunning, is doomed
to overfit. The reason is mathematical, not a matter of discipline: if you try
enough configurations and report only the best, the best looks good even when not
one of the configurations has any real edge. Hiding N is what makes the reported
result meaningless, and only disclosing N lets anyone correct for it.

## The empirical hook: the cost of non-disclosure, quantified

The False Strategy Theorem (Bailey and Lopez de Prado) gives the expected maximum
Sharpe of N skill-less trials, each estimated over a track of T observations:

    E[max SR] = sqrt(V[SR]) * ( (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)) )

where g is the Euler-Mascheroni constant, Z is the inverse standard normal CDF,
and V[SR] is the variance of the per-trial Sharpe estimate (about 1/T for
skill-less trials). The point of the formula is blunt: with zero true edge by
construction, the best of N backtests still climbs steadily with N. That climb is
pure selection.

Plotted on a log-N axis (annualised, T = 1000 observations per trial):

| Number of trials N | Expected max Sharpe (annualised) |
|---|---|
| 10 | 0.79 |
| 100 | 1.27 |
| 1,000 | 1.63 |
| 2,500 | 1.76 |
| 10,000 | 1.94 |
| 100,000 | 2.20 |
| 1,000,000 | 2.44 |

A lone quant who silently searched 2,500 configurations on a single instrument
(the size of this program's per-instrument crypto corpus) and reported only the
best would show an annualised Sharpe near 1.8 from selection alone, with no skill
anywhere in the search. If that quant never tells you N, you have no way to know
the 1.8 is a mirage. Disclosing N is exactly what lets the Deflated Sharpe Ratio
subtract this benchmark and reveal that the winner is luck.

### Validating the formula

The analytic curve was checked against a Monte-Carlo of skill-less strategies, the
only synthetic data used anywhere in this study and clearly labelled as such. Each
trial is an independent stream of zero-mean unit-variance returns (no edge by
construction); for several values of N we draw thousands of such universes, take
the maximum sample Sharpe in each, and average. The Monte-Carlo agrees with the
analytic formula to within Monte-Carlo error at every N tested, with a maximum
absolute difference of **0.0009** in per-observation Sharpe across N from 10 to
5,000. The Sharpe-of-a-matrix inner loop was implemented twice, once in plain
array code and once as a compiled kernel, and the two were verified
**bit-identical** (maximum difference 0.0 over 200 random input matrices). In this
particular workload the random draw dominates the run time rather than the Sharpe
computation, so the compiled kernel is correctness insurance rather than a speed
win here.

## The program as an assembly line: a meta-analysis

The program embodies the assembly line. Across the completed studies, each one
stress-tested one or more stations, every configuration it tried was logged, and
the same validation station (the Deflated Sharpe Ratio, with Probability of
Backtest Overfitting and effective-number-of-trials as supporting diagnostics)
judged all of them on the same bar. The scorecard below reads every number from
the studies' own result tables.

The aggregate result is the thesis in one line: across the strategy studies the
line evaluated roughly **98,000 configurations**, and **essentially none cleared
deflated significance**. Of the handful of per-instrument bests that did clear the
bar, every one ties or loses to its own in-sample-tuned control, so not one is an
edge for the method under test. This is the validation station doing its job: the
factory ran end to end and the quality gate rejected nearly everything, which is
exactly what an honest gate should do on real markets.

Two clarifications keep the count honest. The data-representation studies (bars,
fractional differentiation) and the leakage study are statistical-property work,
cost-free by nature, and carry no strategy-level Deflated Sharpe gate, so they are
excluded from the trial total and marked as such. And the studies that did clear a
few per-instrument gates (a labeller study and an exit-rule study) state in their
own writeups that the surviving effect belongs to the market and the holding
horizon, not to the method being tested, because the method ties or loses to a
plain control.

See `tables/program_scorecard.md` for the full per-study table.

## Separability of stations

The schematic in `figures/assembly_line.png` lays out the six stations and maps
each completed study to the station or stations it stress-tested:

- **Data curators** (raw feeds to clean, causal bars): information-driven bars,
  fractional differentiation.
- **Feature analysts** (informative, non-leaking features): structural breaks and
  entropy, microstructural features, causal factors.
- **Strategists** (labels and sides): triple-barrier and meta-labeling,
  trend-scanning labels.
- **Backtesters and validation** (purged cross-validation, the deflation gate):
  purged cross-validation and combinatorial purged cross-validation, ensembles and
  feature importance, and the overfitting and Deflated Sharpe harness itself.
- **Deployment and sizing** (bet sizing, exit rules): bet sizing, optimal trading
  rules.
- **Portfolio oversight** (allocation across strategies): portfolio construction.

No single study had to do everything, and no study graded its own work, because
the validation station was shared and applied identically.

## Figures and tables

- `figures/expected_max_sharpe_vs_N.png` (and `.svg`): expected maximum Sharpe of
  skill-less trials versus N on a log axis, with the lone-quant marker and the
  Monte-Carlo validation points.
- `figures/assembly_line.png` (and `.svg`): the six-station schematic with each
  study mapped to its station and the disclosure feedback loop into the validation
  gate.
- `figures/program_scorecard.png` (and `.svg`): configurations evaluated versus
  configurations clearing the deflation gate, per study.
- `tables/expected_max_sharpe_vs_N.csv`: the curve.
- `tables/mc_vs_formula.csv`: Monte-Carlo versus analytic agreement.
- `tables/program_scorecard.md` and `.csv`: the cross-study scorecard (study,
  station, trials, trials clearing the gate, methodological claim reproduced).

## Verdict

This is a process and discipline result, not a tradeable edge. The expected-max
formula shows precisely how much free Sharpe selection buys when N is hidden, and
it matches a skill-less Monte-Carlo to within Monte-Carlo error. The program is
itself the worked example of the assembly line: it ran every station, logged every
trial, gated everything on the same deflated bar, and the gate rejected essentially
all of roughly 98,000 configurations. That rejection, reported plainly, is the
thesis. The assembly line plus mandatory disclosure is what separates honest
research from the Sisyphus trap, and the strongest evidence for it is a program
that disclosed its N and let the numbers say no.

## Reproduce

```
python3 scripts/exp_max_sharpe.py        # curve, Monte-Carlo validation, parity check, table
python3 scripts/meta_analysis.py         # reads the other studies' tables, builds the scorecard
python3 scripts/assembly_line_diagram.py # the station schematic
```
