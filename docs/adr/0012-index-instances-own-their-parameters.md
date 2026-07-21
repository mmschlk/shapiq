# 12. Index instances own their parameters

Date: 2026-07-21

## Status

Accepted

## Context

The representation of interaction indices was probed three times. The shipped design
(ADR 0005/0008/0009) makes indices frozen dataclass instances — `SII(order=2)`,
`WeightedBII(p=0.3, order=2)` — carrying their own parameters, with capability protocols
for dispatch and extensional equality as the equality theory. Two alternatives were
implemented and evaluated on a side branch: plain classes passed uninstantiated with
order owned by the explainer (`ExactExplainer(game, SII, order=2)`), and module-level
singleton values of hidden name-literal-generic classes (`class _SII(Index[Literal["SII"]])`,
`SII: Final = _SII()`) with statically precise closed entry sets. Both survive as commit
`0dfffe84` on `shapiqv2-first-approx`, together with their ADR drafts and probe scripts.

## Decision

Indices are **instantiated value objects that own all of their parameters**. The class-value
and singleton representations are rejected.

The deciding argument is the weighted Banzhaf family (`WeightedBV(p)`,
`WeightedBII(p, order)`, `WeightedFBII(p, order)`): a continuous user parameter like `p`
can only live on an instance. A singleton grammar cannot enumerate over `p`, and moving
index parameters onto explainer signatures scatters one concept across every entry point —
`order` is the same kind of parameter and stays on the index for the same reason.

## Consequences

- User grammar: construct the index, pass the object — `Regression(game, FSII(order=2))`.
  Explainers never own index parameters.
- Dispatch stays on exact index types and capabilities (ADR 0007); instance parameters vary
  freely underneath a registered type without touching dispatch.
- Extensional equality (ADR 0009) remains meaningful and observable — `WeightedBII(p=0.5,
  order=k) == BII(order=k)` is exactly the kind of identity only parameterized instances
  can express.
- The side branch is a record of probed alternatives, not pending work; its ADR numbering
  (0011/0012) collides with this line and is superseded by this document.
