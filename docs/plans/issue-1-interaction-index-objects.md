# Issue 1 — Interaction indices become typed objects

Status: **done** (2026-07-07, ADR 0005) · Order: **first**

## Problem

The library's essential structure is a grid — estimation method × **InteractionIndex** × order —
and the API currently answers "which cell am I in?" in three grammars:

| Family | How the index is picked | Where support is encoded |
| --- | --- | --- |
| Permutation | class name: `PermutationSamplingSII(game, order=2)` | class existence |
| Regression | class name: `RegressionFSII(game, order=2)` | class existence |
| Exact | string: `ExactExplainer(game, "SII", order=2)` | `_SUPPORTED_INDICES` tuple, runtime `ValueError` |

A fourth place, `src/shapiq/interactions/_validation.py:41-58`, hard-codes per-index rules
centrally (`SV`/`BV` force `order == 1`). Three defects follow:

1. **Support is invisible to the type checker.** Which explainer works with which index lives in
   runtime errors and docstrings; `ty` cannot answer it, and neither can autocomplete.
2. **`"k-SII"` is a type-level promise nothing keeps.** It sits in the `InteractionIndexName`
   Literal (`src/shapiq/interactions/_types.py`) but no explainer accepts it — correctly so,
   since k-SII is a *transform* of SII, a different kind of thing.
3. **`order` means two different things and the API cannot express which.** For SII and BII,
   order is *coverage*: a pair attribution is identical whether the explanation is requested at
   order 2 or 3, because base interactions are defined per interaction. For STII, FSII, and
   k-SII, order is *identity*: an STII pair attribution at order 3 is the discrete derivative at
   the empty coalition, while at order 2 it is a kernel-weighted average over coalitions — same
   interaction, same game, different number. Comparing `explanation((0, 1))` across orders has
   silently different semantics per index.

The implementation already reveals the underlying truth: within each method family the index is
*data*. `ExactExplainer.explain()` dispatches on exactly two capabilities — a cardinal weight
function `w(s, t)` (SV, BV, SII, BII, STII) versus a regression specification (FSII). The
permutation family, by contrast, is genuinely index-specific: walk layout lives in per-index
samplers and is not derivable from a weight function.

## Goal

Interaction indices become frozen objects that own their intrinsic parameters and expose
capabilities. Estimators are typed against capability protocols where support is intrinsic, and
stay one-class-per-index where it is not. `ty` answers "which explainer works with which index"
statically; illegal states (SV at order 2) become unrepresentable; a user-defined index that
provides its weights works with the exact explainer without shapiq changes.

Sketch of the target grammar:

```python
SV()                 # no order field: order 1 by definition (SV(order=2) unrepresentable)
SII(order=2)         # order = coverage; default 2
STII(order=3)        # order = identity; default 2
FSII(order=2)

ExactExplainer(game, index)             # index: WeightedDerivativeIndex | RegressionIndex
PermutationSamplingSII(game, order=2)   # stays a class for now: builds SII(order) internally
```

String **names stay** on the objects (for `ExplanationArray` metadata, `repr`, and
serialization); objects replace strings only where behavior and validation are selected. The
coverage-vs-identity distinction is carried as a machine-readable property on each index class
(order semantics: coverage for SII/BII, identity for STII/FSII) so later tooling can warn on
cross-order comparisons — grammar prevents errors, documentation plus the property teach
interpretation.

## Current state

- `src/shapiq/interactions/_types.py` — `InteractionIndexName` string Literal (includes
  `"k-SII"`).
- `src/shapiq/interactions/_validation.py:30-58` — `validate_interaction_metadata` with
  centralized per-index rules.
- `src/shapiq/explainers/_base.py` — `Explainer.__init__(game, interaction_index, order,
  orientation)` calls the central validator; all explainers inherit this signature.
- `src/shapiq/explainers/_exact.py:24` — `_SUPPORTED_INDICES` tuple plus if/elif dispatch in
  `explain()` (the two capability families are already visible there).
- `src/shapiq/explainers/_permutation.py`, `_regression.py` — constructors pass
  `interaction_index="SV"`-style strings up to the base.
- `src/shapiq/explanations/_dense.py:36-47` — explanation metadata validated via the same
  central validator.

## Design direction

- Frozen dataclasses per index in `shapiq.interactions`: `SV()` and `BV()` without an order
  field; `SII(order=2)`, `BII(order=2)`, `STII(order=2)`, `FSII(order=2)` with a uniform,
  defaulted order. Each carries its `name`, its order-semantics property, and validates its own
  intrinsic parameters at construction. `Explainer.order` derives from the index — one source
  of truth.
- Capability protocols (structural typing):
  - `WeightedDerivativeIndex` — provides the cardinal weight profile `w(s, t)` used by the exact
    explainer's weighted-derivative kernel;
  - `RegressionIndex` — provides the regression kernel and constraint structure (FSII today,
    kernel-style SV later).
  - Capabilities are not exclusive (SV has both a weight profile and a regression formulation).
- `ExactExplainer(game, index)` typed against the protocol union; the `_SUPPORTED_INDICES` tuple
  and the central per-index rules disappear. The per-index weight functions in `_exact.py`
  migrate onto the index objects (or the protocols expose them; decide during design).
- The permutation family keeps one class per index; the classes construct their index object
  internally instead of passing strings.
- `"k-SII"` leaves the Literal/registry; it returns later as an explanation-level transform
  (e.g. `KSII.from_sii(explanation)`), out of scope here.
- `ExplanationArray` keeps a string name in its metadata (recommendation); transforms and plots
  check names at runtime.
- **Numerics ride along with the migration** (decision 2026-07-07: pure JAX): `_exact.py` moves
  from host-side numpy to `jnp` end to end — resolving the `TODO` at
  `src/shapiq/explainers/_exact.py:27` — and the FSII least-squares solves (`_exact.py` and
  `_regression.py`) follow. float32 is acceptable; float64 only where it comes cheaply. See the
  open decision on the FSII solve.

## Work breakdown

- [x] Design pass: index objects + protocols, exact field/method names, where weight functions
  live; write ADR 0005 (index representation and the coverage-vs-identity order semantics).
- [x] Implement index objects and protocols in `src/shapiq/interactions/_indices.py`; per-index
  rules moved from `_validation.py` onto the objects.
- [x] Migrate `Explainer.__init__` and `ExactExplainer` (constructor takes an index object;
  dispatch via capability protocols).
- [x] Numerics: numpy replaced with `jnp` throughout `_exact.py` and the FSII solve in
  `_regression.py` (shared machinery in `explainers/_faithful.py`); no test tolerances needed
  adapting — all existing atols held in float32.
- [x] Migrate approximator constructors (`_permutation.py`, `_regression.py`) off strings.
- [x] Update `ExplanationArray` metadata validation; `"k-SII"` removed from the accepted set
  (returns later as an explanation-level transform).
- [x] Update exports and `CONTEXT.md` (**InteractionIndex** rewritten; **Order Semantics** and
  **Index Capability** added). `docs/design/core-interfaces.md` needed no change — it does not
  describe index selection.
- [x] Tests: constructor migrations; index-object semantics test; string and capability-free
  indices rejected with teaching errors; SV-with-order unrepresentable (79 tests green).

## Decisions (settled 2026-07-07, to be recorded in ADR 0005)

- **Order lives uniformly on the index** with default 2 where it exists (`SII(order=2)`,
  `STII(order=3)`); SV/BV have no order field, so `SV(order=2)` is unrepresentable. No
  required-order asymmetry for identity indices — consistency wins; interpretation is taught by
  documentation plus a machine-readable order-semantics property (coverage vs identity) on each
  index class. `order` leaves `Explainer.__init__`; `Explainer.order` derives from the index.
- **FSII solves use exact constraint elimination** in pure `jnp` (default float32): the ∅/N
  equality constraints are substituted out, the big-M rows and `_CONSTRAINT_WEIGHT` disappear
  from both `_exact.py` and `_regression.py`, and the reduced system is well conditioned.
- **Weight functions migrate onto the index objects now** (capability methods), since
  `_exact.py` is being rewritten for `jnp` anyway.
- **`ExplanationArray` stores the name string** (serialization stays trivial); index typing is
  constructor-side only, transforms check names at runtime.
- **Internal dispatch is keyed on index types and capabilities, never name strings**, so issue
  5's unified entry points become a thin step.

## Open decisions

- Exact naming of the order-semantics property and its glossary entry.
- Custom third-party index names vs the closed `InteractionIndexName` Literal on explanation
  metadata (deferred: keep the closed Literal for now; note the limit in ADR 0005).

## Acceptance criteria

- `ty` proves index support statically; unsupported combinations fail at construction with a
  type error, not (only) a runtime `ValueError`.
- `SV`/`BV` with `order != 1` is unrepresentable.
- All existing tests pass (with updated constructor calls); lint and `ty` green.
- ADR 0005 recorded; `CONTEXT.md` and `docs/design/core-interfaces.md` updated.

## Dependencies

None — do this first. Every later issue inherits the grammar this one sets.
