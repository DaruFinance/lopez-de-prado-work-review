# Meta-labeling an EDGED primary: corrected result

_mode: **from_banked**; DSR is the headline metric._


## The correction in one table

| | study 03 (edgeless primary) | this study (edged primary) |
|---|---|---|
| primary | edgeless EMA crossover (no established edge) | a proven structural order-flow / open-interest edge (closed, pre-validated) |
| precondition (LdP) | **violated** | **met** |
| median meta PF | ~1.03 (about break-even) | **1.79** |
| meta per-trade | (near 0) | **148 bp** |
| verdict | precision filter, *not* alpha | meta-labeling **improves a real edge** |

## Primary vs meta (OOS, net of per-fill costs)

| arm                  |   n_trades |    pf |   per_trade_bp |   sr_ann |   dsr |   sr0 |
|:---------------------|-----------:|------:|---------------:|---------:|------:|------:|
| primary (edge alone) |        166 | 1.257 |           47.3 |    10.31 | 0.636 | 0.051 |
| meta (gate + size~p) |        107 | 1.786 |          148.1 |    20.47 | 0.778 | 0.084 |


- **PF lift**: 1.257 to 1.786  (+0.529)

- **Per-trade lift**: 47.3 to 148.1 bp  (+100.7 bp)

- **Precision vs base rate**: profitable-bet rate 53.0% (all edge bets) to 63.6% (meta-kept)

- **Gate keeps** 64% of the edge's bets (it vetoes the low-confidence tail).

- **DSR (headline)**: primary 0.636 to meta 0.778.

- **DSR does NOT clear the program's 0.95 publication bar on this 2-pair sample**: the result inverts study 03's blanket claim (direction + mechanism), it is not a new DSR-passing trophy.