# Positive control: does the acceptance bar accept a true edge?

A negative result is only informative if the test that produced it can recognize a real
edge. This positive control checks that the acceptance gates discriminate. It builds
controlled return panels with a planted edge of known strength, books the strategy PnL,
and runs the exact acceptance battery used everywhere else: net-of-cost pooled Sharpe, a
per-window tail-guard, sign-consistency, persistence, cross-candidate deflation, and a
pollute-and-verify causality test. The edge strength is swept from zero (pure noise)
through marginal, moderate, and strong, and a return-additive leak is planted as a fifth
case.

## Verdict

The gates discriminate. Pure noise is rejected. Moderate and strong planted edges pass.
The marginal case is adjudicated by the gates rather than waved through. The planted leak
clears the economic gates yet is caught by the causality test, so it is correctly failed.
A bar that passes genuine planted edges and rejects noise is calibrated, not reflexive.

## Scope

This control exercises the edge-judgment gates and a return-additive causality check on
controlled panels. It is deliberately not the full corpus stack (clustered standard
errors, stale or forward-filled coarse data, PBO, effective-N). The harder forward-fill
leak, where a coarse-frequency feature survives a naive pollute-and-verify, is demonstrated
and caught separately in [`../apex_audit/`](../apex_audit/) through the same-bar-versus-future
correlation and feature ablation. The two pieces together cover both leak classes.

## How to run

`positive_control_v3.py` is the current version; v1 and v2 are kept for provenance. From
the repository root:

```
python3 positive_control/positive_control_v3.py
```

It takes no arguments, generates its panels internally, and needs no market-data root. It
prints the per-strategy table and the conclusion block, then writes its result JSON next to
the script.
