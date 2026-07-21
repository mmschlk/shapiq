# 13. Samplers are vehicles: walk plans, state-owned identity, declared laws

Date: 2026-07-21

## Status

Accepted

## Context

Three ownership questions around sampling had grown answers in the wrong places. The
permutation samplers were per-index subclasses whose only content was estimator-serving
layout — `render_draw`, walk length, and STII's exact lower-order block hidden in a seed
override — coordinated with the estimators through shared pattern functions and docstring
promises ("the pattern order defines the walk layout shared between the sampler and the
approximator"): the layout had two owners, and the family registry's atomicity ceremony
existed to hold them together. Deduplication carried a coalition-identity key map on the
*approximator*, with a branch-detection token and lockstep bookkeeping, because the state
had no notion of identity. And the estimator families on the roadmap (SHAP-IQ, SVARM-IQ,
stratified sampling, unbiased KernelSHAP) are Horvitz-Thompson-shaped — they need
per-coalition multiplicities and sampling probabilities, which v1's `CoalitionSampler`
stored eagerly (`coalitions_counter`, `coalitions_probability`, adjustment weights) and v2
exposed nowhere. The issue-13 investigation established that nothing was missing
informationally — the fold-in determinism makes everything reconstructible from
`(state, sampler)` — the gap was addressability.

## Decision

- **Samplers are vehicles.** One `PermutationSampler` owns the sampling procedure:
  permutation draws via `fold_in`, the emission schedule, pairing as reversal. Which
  coalitions a permutation materializes into is a **walk plan** (`length`, `prelude`,
  `render`) declared by the estimator family that decodes it, so the layout has one owner
  and drift is unrepresentable by construction. The default plan is the prefix chain —
  the bare sampler is the canonical thing its name says. A plan's prelude extends the
  seed block with deterministic evaluations the estimator needs exactly once; STII's
  exact lower-order anchors are a prelude, not sampler behavior.
- **The state owns coalition identity.** `SamplingState.packed_keys()` caches one
  packed-bit identity row per stored coalition; `unique(n_samples)` summarizes the stream
  as distinct coalitions with first positions, multiplicities, and an inverse, in
  first-occurrence order — budget-split invariant like the stream. Deduplication is a
  consumer of this identity, not an owner: the approximator-side key map, its token, and
  its lockstep bookkeeping are gone (and the benchmark got faster).
- **The law is a sampler capability.** `LawfulSampler.log_probability` answers the
  marginal law of one sampled stream position — log-space so many-player binomials stay
  finite, `-inf` outside the support, seeds outside the law. Wrappers transform instead
  of forwarding: `PairedSampler` grafts the complement-symmetrized law at construction
  exactly when the wrapped sampler declares one and pairs by complement (an instance
  attribute, because Python 3.12+ protocol `isinstance` uses `getattr_static` and ignores
  `__getattr__`). Permutation samplers are lawless by design: their stream positions are
  unit-correlated, and walk estimators never need a law.
- **Nothing is stored eagerly.** Identity is a cached derivation, the law is analytic.
  Rejected: v1-style eager counter/probability columns on the state — parallel structures
  with lockstep invariants, the disease the history rework is curing elsewhere.

## Consequences

- A new walk-based estimator is a plan plus an estimator in one module — no sampler
  subclass, no registry-coordination contract, no new export.
- Horvitz-Thompson estimators (the sampled k-SII / SHAP-IQ port) read multiplicities from
  `unique()` and probabilities from the law through public API; the law's first estimator
  consumer arrives with that port (the capability landed ahead of it at the maintainer's
  request, specified by its property tests until then).
- Exactness strata — v1's border trick of enumerating cheap coalition sizes exactly —
  remain the recorded third channel (issue-13): the seed block plus plan preludes are
  already its degenerate form, and the law's contract will condition on the sampled
  stratum when it lands.
