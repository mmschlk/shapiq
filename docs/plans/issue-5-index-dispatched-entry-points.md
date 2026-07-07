# Issue 5 — Index-dispatched estimator entry points

Status: **done** (2026-07-07, ADR 0007) · Order: **after issue 1**

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

- [x] `PermutationSampling(game, index)` dispatching to the SV/SII/STII strategies via a
  type-keyed table; sampled streams bit-identical (state-equality tests and byte-identical
  example output).
- [x] Class-per-index names **removed** (v2 unreleased; one grammar beats two documented
  ones); all 85 call sites across tests and examples migrated.
- [x] Regression family unified immediately: `SV` gained the regression capability (the
  KernelSHAP kernel is the order-1 faithful kernel up to constant scaling), so
  `Regression(game, SV())` is KernelSHAP and `Regression(game, FSII(order=k))` covers the
  faithful interactions.
- [x] Rider (2026-07-07 discussion): **orientation moved onto the index objects** — the
  `orientation` parameter left `Explainer.__init__`; `Explainer.orientation` derives from the
  index.
- [x] `CONTEXT.md` (InteractionIndex, Interaction Orientation), exports, ADR 0007.

## Decisions (settled 2026-07-07, recorded in ADR 0007)

- Removal over aliases for the class-per-index names.
- Closed unions over open protocols for family signatures: Regression's unit-weight trick
  requires the sampler distribution to match the index kernel, so an open bound would accept
  silently-wrong custom kernels; permutation layouts are bespoke per index.
- The dispatch tables stay private until a third-party index family exists.
- BV stays derivative-only: its least squares form is unconstrained (Hammer–Holzman) and does
  not match the constrained-regression capability.

## Acceptance criteria

- `PermutationSampling(game, index)` works for SV, SII, and STII with static narrowing; sampled
  streams are bit-identical to the previous per-class constructors.
- The type checker rejects unsupported index types per family.
- Docs and glossary reflect the single-entry-point grammar.

## Dependencies

Issue 1 (index objects). Does not block issues 2–4; natural slots are directly after issue 1
(while the machinery is warm) or after the north star lands.
