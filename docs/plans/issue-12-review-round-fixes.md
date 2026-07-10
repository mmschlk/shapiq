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

### SII empty-set validation (no code change)

The Grabisch-Roubens formula does define `SII(empty)` — the Shapley-weighted expectation of the
centered game, neither zero nor the baseline — and v1 carried it (uncentered) because
`base_interaction` iterated the powerset from size zero, patching only CHII (undefined at the
empty set). Nothing consumes it: v1's k-SII aggregation projects onto *nonempty* subsets and
set k-SII's `()` to the baseline by fiat; v2 aggregates blocks 1..k. Decision: SII/BII/
WeightedBII keep `min_interaction_size = 1`, preserving the SV <-> SII(order=1) same-shape
coverage that makes extensional equality observable.

### Pairing unified, `min_budget` an honest floor

`paired` is `bool | None = None` on both sampling explainers with one contract — `None`
resolves to the family default (Regression: paired exactly for complement-symmetric kernels;
Permutation: unpaired), explicit booleans force. `min_budget` docstrings now state the floor
semantics (base and Regression override, with the kADD-identifies-later note), and the
insufficiency errors state the shortfall: identification reports "rank r of the C required, so
at least C-r more distinct informative coalitions", SII coverage reports a lower bound on the
completed walks still needed.

### Torch call policy: two owners, one per entry style (decided, not retired)

`TorchCallableGame` stays — as the policy owner for raw coalition callables (bool-tensor
coalitions in via DLPack, no-grad evaluation, detached JAX values out), the peer of
`ChunkedMaskedPredictor`, which owns the masked path (no-grad, devices, chunking). Explicitly
rejected: routing the masked path *through* the game as a middle layer — predictors are not
games, and the extra layer buys nothing. De-zombified with tests (no-grad and detach pins,
boolean-tensor conversion, end-to-end sampling, signature alignment with `CallableGame`, whose
converters are now keyword-only via `KW_ONLY` to close the LSP gap). `ModelMaskedPredictor`
documents that it is backend-generic and applies no call policy; the ceremonial
`link_function=to_jax` example is gone.

### Maskers are math: backend-general masking via the Array API

`BaselineMasker` and `SuperpixelMasker` moved to core `shapiq.games` and compute in whatever
backend their arrays come from (`array_namespace` — NumPy, JAX, torch, anything Array API
compatible), with `grid_labels` returning a host NumPy map and one backend-specific seam
(`coalition_masks_like`: DLPack with a host fallback, device-following). `games.torch` now owns
execution policy only (`ChunkedMaskedPredictor`, `TorchCallableGame`, `to_jax`). This closed
most of the numpy on-ramp: sklearn models explain without ceremony (tested end-to-end), and a
subprocess test pins that numpy masking never imports torch. `to_host_array` moved to
`games/_values.py` as the one host-conversion helper (trees reuse it). Explicitly rejected:
generalizing chunked evaluation the same way — chunking is execution policy, and its
backend-agnostic form belongs to issue 9's evaluation seam (noted there), not to another
wrapper. Backend worlds after this: only torch needs a policy module; a "numpy world" is
deliberately empty, a "jax/flax world" is an example file, and Array API backends like cupy
work through the standard without shapiq code.

## Open backlog from the reviews (biggest first)

- numpy on-ramp remainder: background-averaging masker (marginal/background-dataset baselines),
  README quickstart, and a dense array exporter (`np.asarray(explanation)` is a silent 0-d
  object array today).
- Dead contracts: raise `UnsupportedGameError` from TreeExplainer's game axis, find a raiser for
  or delete `SamplingError`; delete the orphan `ShapleyValue`/`BanzhafValue` singletons.
- Reprs: samplers, ExactExplainer, TreeExplainer, InterventionalTreeGame are grade-F defaults;
  SamplingState/TreeModel dump full arrays.
- Exports: promote `ShareSamples`, `AntitheticDraws`, `ExtensionalEquality`, `LeafConstraints`;
  reconcile family-registry docstrings with the registries-are-internal decision.
- Dispatch-duality ADR amendment: "register where the algorithm's variance lives,
  capability-check the other axis"; name the tree-game registry category.
- Docs/glossary: the `baseline` homonym (reference point vs `v(empty)`), TreeExplainer glossary
  entry, ADR 0009/0005 drift, the name-keyed SV/BV check in `validate_interaction_metadata`.
- Strategic v1 regressions to rank: plots, sampled k-SII, background-dataset baselines,
  aggregation utilities, PathDependentTreeGame (xgboost/lightgbm converters landed with the
  conversion kernel, see issue-11; catboost deferred; TokenMasker landed as the text on-ramp).
