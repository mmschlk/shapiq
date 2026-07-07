# Issue 3 — Vector-valued games (un-bake the scalar-only enforcement)

Status: **done** (2026-07-07, ADR 0006) · Order: **second**

## Problem

`CONTEXT.md` promises vector values — a **Value** "may be scalar-valued, vector-valued, or a
structured array-like element" — but the evidence layer enforces scalars per coalition:

- `src/shapiq/explainers/_evidence.py:176-182` and `:203-212` raise `UnsupportedGameError`
  unless game output has shape `(*target_shape, n_coalitions)`;
- `src/shapiq/explainers/_exact.py` applies the same check in `_game_values`;
- the estimators' `explain()` paths assume a trailing sample axis with nothing after it.

The motivating consumer is issue 2's milestone B: a torch classifier returning all class logits,
explained in one pass with per-class attributions from a single evidence set. (A related pressure
point: 100k samples × 1000-dim float32 logits ≈ 400 MB of evidence — see guidance below.)

## What is already in place

- **The explanation container is vector-ready.** `DenseExplanationArray` threads arbitrary
  trailing value axes untouched (`Ellipsis` indexing at
  `src/shapiq/explanations/_dense.py:97-102` and `:150-151`) and applies no block-shape
  validation at all. The work is the game/evidence contract, not the container.
- **The estimator math is linear in Values.** Weighted-derivative sums generalize by
  broadcasting; the FSII least-squares solve already runs multi-RHS for shared targets.

## Design direction

- **Decide the canonical dense layout** (the ADR-worthy contract): values shaped
  `(*target_shape, n_samples, *value_shape)` — value axes trail the sample axis. Explanation
  blocks then carry `(*target_shape, n_interactions, *value_shape)`.
- **`Game` gains value-shape metadata.** The base class currently declares only `n_players` and
  `target_shape` (`src/shapiq/games/_base.py:12-31`); estimators need the value shape to
  validate and allocate without a probe evaluation.
- **Thread through the evidence layer**: `SamplingState` append/validation, the
  `_stitch_values` scatters and shape checks in `_evidence.py` (replace "scalar values required"
  with "values of the declared value shape required"), and each family's `explain()`.
- **Decide what validation `DenseExplanationArray` gains** — today it checks nothing about block
  shapes; once the layout is contractual, checking blocks against
  `(*shape, n_interactions, *value_shape)` becomes possible.
- **Memory guidance stays in the docs**: evidence scales with value size; a **LinkFunction**
  that reduces predictions (one class, a margin, a scalar loss) remains the recommended path when
  only some outputs matter. Vector values are for when per-component attributions are the point.

## Work breakdown

- [x] ADR 0006: value-space layout contract (declared `value_shape`, boundary validation in
  `Game.__call__`, logical axes → sample axis → value axes, leading-value internal compute
  form).
- [x] `Game` value-shape metadata: base default, `CallableGame` field, `MaskedGame` field at
  the composition site (the game owns the declaration; links stay plain callables),
  `TorchCallableGame` pass-through.
- [x] Evidence layer: `SamplingState` needed **zero changes** — its positional append and
  Ellipsis slicing were already built for the layout; `_evidence.py` scalar checks removed in
  favor of the boundary, dedup stitching made value-aware.
- [x] `ExactExplainer` (leading-form einsums, multi-RHS faithful solve).
- [x] Permutation family and `RegressionFSII` `explain()` paths (`explainers/_valueaxes.py`
  holds the two axis moves).
- [x] Explanation-block validation in `DenseExplanationArray` plus a recorded `value_shape`.
- [x] Tests: `tests/shapiq/test_vector_values.py` — vector runs match per-component scalar
  runs across all three families, per-component efficiency, pending masking, history slicing,
  dedup, misdeclaration errors, block validation; scalar suite untouched (89 tests green).
- [x] `CONTEXT.md`: **ValueArray** entry records the dense layout.

## Decisions (settled 2026-07-07, recorded in ADR 0006)

- Value shape is **declared** on the game (not inferred); the game validates its own output at
  the boundary.
- `DenseExplanationArray` **records `value_shape`** and validates every attribution block.
- Deduplication confirmed unaffected (keys are coalition-side; covered by the vector dedup
  test).
- Still open for issue 4's benchmark: cost of the FSII solve scaling with value size
  (value axes flatten into right-hand-side columns; measure).

## Acceptance criteria

- A vector-valued game runs end to end through the exact explainer, permutation sampling, and
  the FSII regression; per-component efficiency holds where the index guarantees it.
- Scalar behavior is bit-for-bit unchanged (existing suite green without edits beyond
  constructor grammar from issue 1).
- ADR 0006 recorded.

## Dependencies

After issue 1 (write new signatures in the final grammar). Blocks issue 2 milestone B. Touches
`_evidence.py` like issue 4 — run the two in sequence.
