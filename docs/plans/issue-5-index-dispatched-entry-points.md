# Issue 5 — Index-dispatched estimator entry points

Status: **not started** · Order: **after issue 1, timing flexible**

## Goal

One user-facing entry point per estimation-method family, with the **InteractionIndex** object
selecting the concrete implementation by dispatch:

```python
PermutationSampling(game, SII(order=2))   # dispatches to the SII walk layout + estimator
PermutationSampling(game, STII(order=3))  # dispatches to the STII layout
Regression(game, FSII(order=2))           # FSII today; kernel-style SV later
```

Users learn one grammar — *method family × index object* — and the class-per-index surface
(`PermutationSamplingSV`/`SII`/`STII`, `RegressionFSII`) becomes an internal implementation
detail. A later `shapiq.explain(...)` convenience factory rides on the same dispatch.

## Current state

- Issue 1 introduces the index objects and establishes capability-based dispatch inside
  `ExactExplainer` (protocol `isinstance` checks, never name strings) — the pattern this issue
  generalizes.
- The permutation family is class-per-index because walk layout is bespoke per index; the layout
  knowledge lives in per-index samplers (`PermutationWalkSampler`, `PermutationSIISampler`,
  `PermutationSTIISampler`) and per-index `explain()` logic.

## Design direction

- Dispatch keyed on the **index type** (index objects are frozen and hashable): a registry or
  `match`/`isinstance` mapping from index type to the concrete sampler + estimator strategy.
  Support stays statically visible via a closed union per family (e.g.
  `PermutationSampling(game, index: SV | SII | STII)`) or overloads.
- The existing concrete classes become internal strategies. Whether the public class-per-index
  names remain as thin aliases (discoverability) or are removed entirely is an open decision —
  v2 is unreleased, so removal is on the table.
- Extension story: a registry keyed on type gives third-party indices a documented way to plug a
  method family later; keep that door visible but do not build it here.

## Work breakdown

- [ ] `PermutationSampling(game, index)` dispatching to the SV/SII/STII strategies; behavior
  bit-identical to the class-per-index constructors (stream-identity tests must not change).
- [ ] Decide the fate of the class-per-index names (aliases vs removal); migrate tests/examples.
- [ ] Same treatment for the regression family when it has a second index.
- [ ] Update `CONTEXT.md`, `docs/design/core-interfaces.md`, and exports.

## Open decisions

- Aliases vs removal for `PermutationSamplingSV`/`SII`/`STII` and `RegressionFSII`.
- Closed union vs overloads for the static support signature (overloads also let the return
  type vary per index if that ever matters).
- Whether the dispatch registry becomes public extension API now or stays private until a
  third-party index family exists.

## Acceptance criteria

- `PermutationSampling(game, index)` works for SV, SII, and STII with static narrowing; sampled
  streams are bit-identical to the previous per-class constructors.
- The type checker rejects unsupported index types per family.
- Docs and glossary reflect the single-entry-point grammar.

## Dependencies

Issue 1 (index objects). Does not block issues 2–4; natural slots are directly after issue 1
(while the machinery is warm) or after the north star lands.
