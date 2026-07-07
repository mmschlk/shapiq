# Issue 2 — Torch example with baseline masking

Status: **done** (2026-07-07, milestones A and B) · Order: **third**

## Goal

The north star of this phase: explain a torch model with baseline masking, end to end, through
the composed pipeline —

> **Masker** + model → **ModelMaskedPredictor** → **LinkFunction** → **MaskedGame** →
> **Approximator** → **ExplanationArray**

— exercised by the estimators built so far (permutation sampling, `RegressionFSII`, and an
`ExactExplainer` cross-check). This is both the deliverable users care about and the first stress
test of the Masker/MaskedPredictor/LinkFunction language against a real model.

## Current state

- Pipeline classes are concrete except the masking step: `ModelMaskedPredictor`
  (`src/shapiq/games/_masked_predictor.py:37`) and `MaskedGame` (`src/shapiq/games/_masked.py:15`)
  are frozen dataclasses ready to compose; `Masker` (`src/shapiq/games/_masker.py:11`) is
  abstract and **no concrete Masker exists anywhere in `src/`**.
- `LinkFunction` (`src/shapiq/games/_base.py:34`) is a structural protocol with no
  implementation; `Model` is a plain callable alias.
- Torch boundary helpers already exist: `TorchCallableGame` with `_coalitions_to_torch` (DLPack)
  and `_torch_to_jax` in `src/shapiq/games/torch/_callable.py`.
- torch is available via the `all_ml` dependency group (included by `dev`) — no dependency work
  needed.
- `examples/` holds a single synthetic-game playground (`permutation_sampling.py`); nothing uses
  a model or a masker yet.

## Design direction

- **`BaselineMasker`** (concrete): absent **Players** are replaced by baseline values; for
  tabular inputs, `masked = where(coalition, x, baseline)`. The masker sets `n_players` and
  `target_shape` itself (the abstract base has no `__init__`) and produces model-native torch
  tensors — reuse or promote `_coalitions_to_torch`.
- **LinkFunction implementation**: maps torch predictions to **Values**, including the
  torch → JAX conversion (reuse `_torch_to_jax`). Milestone A uses a scalar link (e.g. the
  predicted probability or log-odds of one class); milestone B maps all logits at once.
- **Example script** `examples/torch_baseline.py`: synthetic tabular data with a known structure,
  a small MLP trained inside the script (no downloads), ≤ 15 Players so the exact explainer can
  cross-check. Demonstrates: lazy start (construction never evaluates the game), budget splits,
  history-based convergence, deduplication, and sampled-vs-exact agreement.
- Expect glossary friction — this is the first real consumer of the Masker/LinkFunction
  language. Record every awkward moment; that feedback is a primary output of the issue.

## Milestones

- [x] **A — scalar link.** Predicted class-1 probability; permutation-sampled SV converges to
  the exact SV of the masked model with exact per-budget efficiency.
- [x] **B — vector link.** Both class log-probabilities in one pass (`value_shape=(2,)`);
  exact and sampled FSII find the planted x0·x1 interaction dominant for class 0.

## Work breakdown

- [x] Concrete `BaselineMasker` in `shapiq.games.torch` (decision 2026-07-07): frozen
  dataclass deriving `n_players`/`target_shape` from the inputs; batched Explanation Targets
  supported and tested.
- [x] `to_jax` link helper in `shapiq.games.torch` (torch → JAX via DLPack); links themselves
  stay plain callables in user code.
- [x] `examples/torch_baseline.py` — trains a small MLP on synthetic tabular data with a
  planted x0·x1 interaction, then runs both milestones with exact cross-checks and shows the
  identification gate honestly.
- [x] Tests: `tests/shapiq/test_torch_baseline.py` (masking correctness, batches, metadata
  validation, closed-form SV of a masked linear model, sampled-vs-exact, scalar link).

## Friction findings (recorded 2026-07-07)

- **DLPack striding.** Link outputs are typically strided views (`predictions[..., class]`),
  and JAX's DLPack import requires compact striding. `to_jax` makes tensors contiguous; the
  same latent bug existed in `TorchCallableGame._torch_to_jax`, which now delegates to
  `to_jax`.
- **Masker authoring pattern.** The abstract `Masker` has no `__init__`; a frozen dataclass
  deriving metadata in `__post_init__` worked cleanly and is the pattern to document for
  masker authors.
- **`LinkFunction` as a plain callable held up.** Lambdas sufficed for both links; the
  game-side `value_shape` declaration (ADR 0006) kept links metadata-free, as intended.
- **`min_budget` is necessary, not sufficient, for identification** under random kernel
  samples (the example shows rank 34 of 35 at `min_budget + 20`). The gate plus its error
  message carry the UX; a coverage-guaranteeing sampling refinement is a possible future
  observation, not a task.
- The pipeline classes (`ModelMaskedPredictor`, `MaskedGame`) composed without any changes;
  no glossary edits were needed.

## Open decisions

- The example shows the single-instance story for readability; batched Explanation Targets
  are exercised in the tests instead. Revisit if the example should also showcase
  `share_samples`.

## Acceptance criteria

- The example runs top to bottom with `uv run --group dev python examples/torch_baseline.py`
  and prints a sampled-vs-exact comparison that agrees within tolerance.
- New library pieces (masker, link) are exported, documented, and tested.
- Friction findings recorded (in this file or as follow-up issues/ADRs).

## Dependencies

Milestone A: none. Milestone B: issue 3 (vector-valued games). Prefer running after issue 1 so
new signatures are written in the final grammar.
