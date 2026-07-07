# Issue 6 — Exact index families and declared generalizations

Status: **done** (2026-07-07, ADR 0008) · Origin: prototype worktree `claude/frosty-sinoussi-f97058`

## Goal

Port the prototype's exact computation algorithms for the full index family zoo — CHII, k-SII,
FBII, kADD-SHAP, the generalized values SGV, BGV, CHGV, IGV, EGV, JointSV, and the Moebius and
Co-Moebius transforms — from the string-grammar branch into the index-object API, and bake the
cooperative-game-theory taxonomy into the library: capability protocols named after the
literature concepts, and a declared, numerically tested **Value Generalization** relation
("order-1 SII IS the SV").

## Work breakdown

- [x] Twelve new index objects in `src/shapiq/interactions/_indices.py` with full metadata
  (order semantics, orientation, order-0 conventions, generalizations).
- [x] Capability taxonomy: `WeightedDerivativeIndex` renamed to **`CardinalInteractionIndex`**
  (the Grabisch–Roubens concept it always was) with `min_interaction_size` for the transforms;
  new **`GeneralizedValueIndex`** capability for bloc marginals; SGV/BGV/CHGV literally reuse
  the SII/BII/CHII weight profiles.
- [x] `generalizes` declared on every index (SII, CHII, STII, k-SII, FSII, kADD-SHAP, SGV,
  CHGV, JointSV → SV; BII, FBII, BGV → BV) and verified by a parametrized numerical test —
  a wrong declaration fails CI. Object equality across index types was rejected (ADR 0008).
- [x] `ExactExplainer` dispatch extended: dedicated solvers for k-SII (Bernoulli aggregation),
  FBII (unconstrained uniform fit, intercept order-0), kADD-SHAP (Bernoulli basis, pivot
  elimination of the grand-coalition constraint); capability paths for cardinal and
  generalized-value indices; transforms default `order=None` → all orders.
- [x] Names/validation extended (`InteractionIndexName`, `_INDEX_NAMES`); exports; glossary
  entries (**Cardinal Interaction Index**, **Generalized Value**, **Value Generalization**).
- [x] Tests: `tests/shapiq/test_exact_index_families.py` — brute-force checks per family plus
  the executable generalization taxonomy (125 tests green overall).
- [x] Adversarial verification workflow: nine independent verifiers re-derived each family
  from first principles with their own games and seeds, plus port-fidelity and dispatch/design
  audits.

## Notes

- Sampled estimators are unchanged: `PermutationSampling` and `Regression` keep their closed
  index sets; extending them to new families is per-family future work. kADD-SHAP already
  declares the Shapley regression kernel, so the kernel-matched sampled estimator is the
  natural first extension.
- The prototype worktree can be discarded once this lands; its uncommitted content is fully
  represented here in the object grammar.
