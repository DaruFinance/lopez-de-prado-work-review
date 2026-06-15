# Meta-labeling a primary that already has an edge

Reference: Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Chapter 3; *Machine Learning for Asset Managers* (Cambridge, 2020), Chapter 5.

Meta-labeling presupposes a primary model that already has an edge; the secondary only sizes and filters it. The sibling project `03_meta_labeling` applied the secondary to a plain EMA-crossover primary with no edge net of costs, which is garbage in. This project meets the precondition instead. The primary is a proven structural order-flow / open-interest edge (a closed, already-validated crypto archetype reused read-only through its banked engine), and the same apparatus is applied on top of it: triple-barrier outcomes, purged k-fold cross-validation, realistic per-fill costs, full-OHLC intrabar exits, and the Deflated Sharpe Ratio as the headline with PBO and effective-N. The base run uses a two-pair sample; a scaled run pools the meta-gated sleeves across the full set of structural archetypes and the multi-pair perp universe.

Verdict: a method-scope nuance, not a deployable edge. With an edged primary the secondary amplifies a real edge (profit factor rises, per-trade PnL roughly triples) and the Deflated Sharpe rises to about 0.78. It does not clear the 0.95 bar on this two-pair sample.

## Run

From the repo root. The two-pair study, reading the banked verified ledger:

```
python3 projects/03b_metalabel_real_primary/scripts/run_metalabel_real_primary.py --from-banked
```

A fast sanity pass, and the scaled run over the full archetype set and perp universe:

```
python3 projects/03b_metalabel_real_primary/scripts/run_metalabel_real_primary.py --from-banked --smoke
python3 projects/03b_metalabel_real_primary/scripts/run_metalabel_scaled.py
```

Both scripts also offer a `--live` / engine-driven path that reproduces the primary signals from the structural engine rather than the banked ledger. Data roots and engine paths come from `config.py` and its `LDP_*` environment variables: `LDP_METALABEL_LEDGER` for the banked ledger and verdict, `LDP_STRUCTURAL_ENGINE` (and `LDP_DATA_CACHE`) for the live path. See `../../DATA.md` and `../../config.py`.
