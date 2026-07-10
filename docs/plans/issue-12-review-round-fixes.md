# Issue 12 — Review-round fixes

Status: in progress (2026-07-10). A five-agent API review of the tree story and the API as a
whole produced one convergent bug cluster, a set of contract repairs, and a strategic backlog.
This file tracks the round; crash-grade cext findings were fixed directly in the issue-11 tree.

## Landed

### The explanation representable window (the bug cluster)

Three reviewers converged on the sparse/dense explanation divergence; all three bugs reproduced
on the committed tree/exact pair and are fixed by making explanation arrays consult the index
they already carry:

- **Batched sparse lookup was fake.** `sparse(array_of_interactions)` short-circuited to the
  default attribution — wrong values (stored attributions ignored) and wrong shape. It now
  normalizes each row and answers from storage with the dense output layout; rows that are
  missing on a default-free explanation raise a `KeyError` pointing to `has()` (decided
  2026-07-10).
- **Fabricated order-0.** `sparse(())` returned the zero default for indices that define no
  order-0 attribution, violating ADR 0010's "raises otherwise". Both array types now gate the
  window `index.min_interaction_size .. order` in their normalizers with a teaching error
  ("SII defines no order-0 attribution; the empty-coalition value travels as the explanation's
  baseline"); the sparse zero-default only ever fires inside the window (true tree sparsity).
- **Self-inconsistent iteration.** `iter_interactions()` yielded `()` that the array's own
  accessor then rejected. The method's `min_order` now defaults to the index's smallest
  represented size; an explicit argument still wins.

Prerequisite metadata unification: `min_interaction_size` is now a required member of the base
`InteractionIndex` protocol and exists on all shipped indices (it was missing on FSII, FBII,
WeightedFBII, KSII, KADDSHAP and the generalized values; FBII/WeightedFBII carry 0 for their
fitted intercept). `includes_empty_interaction` is kept but derived (`min_interaction_size == 0`)
on the equality mixin — one stored field, one law, pinned by the index-identity metadata test.

### Tree array story, vectorized evaluation, assert policy

Recorded in [issue-11](issue-11-tree-story.md): array-API/torch inputs at the tree seams with
host-exact float64 routing, ensemble-wide single-pass `_call` (8-58x), default-JAX-precision
outputs (closes float32-under-x64), and the no-asserts/no-bandit-noqa policy.

## Open backlog from the reviews (biggest first)

- numpy on-ramp: core `BaselineMasker` + background-averaging masker, README quickstart, and a
  dense array exporter (`np.asarray(explanation)` is a silent 0-d object array today).
- `min_budget` docstring states a guarantee but is a floor; insufficiency errors should say how
  much more to sample (the machinery knows: "rank 15 of 20").
- Unify `paired` across Permutation (`bool = False`) and Regression (`bool | None = None`).
- Dead contracts: raise `UnsupportedGameError` from TreeExplainer's game axis, find a raiser for
  or delete `SamplingError`; delete the orphan `ShapleyValue`/`BanzhafValue` singletons.
- Reprs: samplers, ExactExplainer, TreeExplainer, InterventionalTreeGame are grade-F defaults;
  SamplingState/TreeModel dump full arrays.
- Exports: promote `ShareSamples`, `AntitheticDraws`, `ExtensionalEquality`, `LeafConstraints`;
  reconcile family-registry docstrings with the registries-are-internal decision.
- Torch call policy: retire `TorchCallableGame` (untested, unused); move no_grad/detach to the
  predictor seam (`ModelMaskedPredictor` builds autograd graphs today).
- Dispatch-duality ADR amendment: "register where the algorithm's variance lives,
  capability-check the other axis"; name the tree-game registry category.
- Docs/glossary: the `baseline` homonym (reference point vs `v(empty)`), TreeExplainer glossary
  entry, ADR 0009/0005 drift, the name-keyed SV/BV check in `validate_interaction_metadata`.
- Strategic v1 regressions to rank: plots, sampled k-SII, xgboost/lightgbm/catboost converters,
  background-dataset baselines, aggregation utilities, PathDependentTreeGame.
