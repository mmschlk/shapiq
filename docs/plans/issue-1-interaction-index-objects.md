# Issue 1 — Interaction indices become typed objects

Status: **in progress** (design pass) · Order: **first** · Expected outcome includes **ADR 0005**

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
SV()                 # order-free: order 1 by definition
SII()                # order-free: order is explanation coverage
STII(order=3)        # order is part of the index identity
FSII(order=2)

ExactExplainer(game, index)             # index: WeightedDerivativeIndex | RegressionIndex
PermutationSamplingSII(game, order=2)   # stays a class: walk layout is bespoke
```

String **names stay** on the objects (for `ExplanationArray` metadata, `repr`, and
serialization); objects replace strings only where behavior and validation are selected.

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

- Frozen dataclasses per index in `shapiq.interactions`: `SV()`, `BV()`, `SII()`, `BII()`
  (no order field) and `STII(order)`, `FSII(order)`. Each carries its `name: str` and validates
  its own intrinsic parameters at construction.
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

- [ ] Design pass: index objects + protocols, exact field/method names, where weight functions
  live; write ADR 0005 (index representation and the coverage-vs-identity order semantics).
- [ ] Implement index objects and protocols in `src/shapiq/interactions/`; per-index rules move
  from `_validation.py` onto the objects.
- [ ] Migrate `Explainer.__init__` and `ExactExplainer` (constructor takes an index object;
  dispatch via protocol).
- [ ] Numerics: replace numpy with `jnp` throughout `_exact.py` and the FSII solve in
  `_regression.py`; adapt test tolerances to float32 where accumulation shows.
- [ ] Migrate approximator constructors (`_permutation.py`, `_regression.py`) off strings.
- [ ] Update `ExplanationArray` metadata validation; remove `"k-SII"` from the accepted set and
  note the future transform.
- [ ] Update exports, `CONTEXT.md` (**InteractionIndex** entry; new entries for the index-object
  and capability language), and `docs/design/core-interfaces.md`.
- [ ] Tests: constructor migrations; a typed-support check (e.g. a `ty`-checked snippet or test
  asserting the protocol bounds); SV-at-order-2 unrepresentable.

## Open decisions

- Does `ExplanationArray` store the index object, the name string, or both? (Recommendation:
  name string; keeps serialization trivial.)
- Does `order` leave `Explainer.__init__` entirely (identity-orders live on the index; coverage
  order becomes an `explain()`/constructor argument for SII/BII)?
- Do weight functions migrate onto the index objects in this issue or in a follow-up (smaller
  first step: objects wrap the existing private functions)?
- How much index typing propagates into `ExplanationArray` (recommendation: constructor-side
  static typing only; runtime name checks for transforms).
- FSII solve numerics under default-float32 JAX: exact elimination of the ∅/N equality
  constraints (drops the big-M rows and their ~1e7 conditioning, making float32 safe, and
  removes `_CONSTRAINT_WEIGHT` from `_regression.py` as well) vs documenting an
  `jax_enable_x64` requirement vs keeping a numpy-float64 island for the solves only.

## Acceptance criteria

- `ty` proves index support statically; unsupported combinations fail at construction with a
  type error, not (only) a runtime `ValueError`.
- `SV`/`BV` with `order != 1` is unrepresentable.
- All existing tests pass (with updated constructor calls); lint and `ty` green.
- ADR 0005 recorded; `CONTEXT.md` and `docs/design/core-interfaces.md` updated.

## Dependencies

None — do this first. Every later issue inherits the grammar this one sets.
