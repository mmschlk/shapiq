# Issue 7 — Faithful Banzhaf in the regression world

Status: **done** (2026-07-07)

## Goal

Bring FBII into the sampled `Regression` entry point, completing the pattern ADR 0007
anticipated: each supported index samples coalitions from its own kernel, so sampled rows enter
the least squares fit with unit weight.

## What landed

- [x] **`BanzhafKernelSampler`** (`src/shapiq/sampling/_kernel.py`): uniform over the powerset
  via independent fair membership coin flips — the Banzhaf kernel — with paired complements
  (uniform by symmetry) and the standard seed block, fold-in unit randomness, and split
  invariance.
- [x] **`Regression(game, FBII(order=k))`**: the entry point dispatches the sampler by index
  type (Shapley kernel for SV/FSII, Banzhaf kernel for FBII) and the solve by kernel family —
  constrained elimination for the Shapley fits, unconstrained least squares with a free
  intercept for FBII, whose order-0 attribution is the fitted intercept (v1 parity: v1 weighted
  rows by `1/2**n` with no big-M for FBII; sampling uniformly with unit weights is the same
  estimator in the v2 architecture).
- [x] `min_budget` accounts for the intercept (`seeds + columns + 1` vs the constrained
  `seeds + columns - 1`); identification rank-gates the intercept-bearing design.
- [x] Tests (`tests/shapiq/test_regression_fbii.py`): exact recovery of 2-additive games once
  identified, order-1 convergence to the Banzhaf value ("kernel Banzhaf"), convergence to exact
  FBII, dedup identity, split invariance, pending masking, identification gate, metadata.

## Notes

- kADD-SHAP remains the natural next `Regression` member: it already declares the Shapley
  regression kernel, so the existing sampler matches; it needs its Bernoulli design and
  grand-coalition pivot elimination on sampled rows plus an identification story.
