# Issue 2 — Torch example with baseline masking

Status: **not started** · Order: **third** (milestone A may start any time)

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

- [ ] **A — scalar link.** Single scalar Value per Coalition; works on the current scalar-only
  contract. No dependencies.
- [ ] **B — vector link.** All class logits explained at once (per-class attributions from one
  evidence set). Depends on issue 3.

## Work breakdown

- [ ] Concrete `BaselineMasker` with tests (masking correctness, `target_shape` handling,
  batched Explanation Targets).
- [ ] LinkFunction implementation(s) for the torch boundary with tests.
- [ ] `examples/torch_baseline.py` (milestone A), extended for milestone B after issue 3.
- [ ] Record language/API friction; update `CONTEXT.md` and `docs/design/core-interfaces.md`
  where the glossary needs new or sharper entries.

## Open decisions

- Where `BaselineMasker` lives: `shapiq.games` vs `shapiq.games.torch`. The masking math is
  backend-agnostic; producing torch tensors is not. (An option: a backend-agnostic masker core
  plus a thin torch adapter.)
- How the example uses batched **Explanation Targets** (`target_shape` + `share_samples`) —
  explaining several instances at once is the natural showcase.
- Which pieces graduate from the example into the library proper (masker and link likely do;
  the training loop does not).

## Acceptance criteria

- The example runs top to bottom with `uv run --group dev python examples/torch_baseline.py`
  and prints a sampled-vs-exact comparison that agrees within tolerance.
- New library pieces (masker, link) are exported, documented, and tested.
- Friction findings recorded (in this file or as follow-up issues/ADRs).

## Dependencies

Milestone A: none. Milestone B: issue 3 (vector-valued games). Prefer running after issue 1 so
new signatures are written in the final grammar.
