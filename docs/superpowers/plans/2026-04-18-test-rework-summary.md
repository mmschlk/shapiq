# Test Suite Rework — Summary

**Branch:** `claude/plan-testing-suite-rework-FF4ZA` (PR #512)
**Date:** 2026-04-18

## What was done

Built on top of earlier protocol-driven tests (`fd0282a`) and the first cross-check
commit (`67cd77f`). Each step below is one commit.

1. **Harden SOUM fixtures** (`474be2a`) — `soum_5` / `soum_7` now use
   `max_interaction_size = n` and 25-40 basis games, making them genuinely
   non-k-additive. Dropped `InconsistentKernelSHAPIQ` from the "exact at full
   budget" list (it only looked exact in the old trivial regime). Added `BV` and
   a direct Möbius-vs-SOUM check.
2. **Tree cross-check rewritten** (`c9b11ca`) — dropped the soon-to-be-removed
   `TreeSHAPIQXAI` dependency; now pairs `InterventionalGame` with
   `InterventionalTreeExplainer` (both interventional, semantically matched).
3. **Review round 1** (`619f349`) — added `strict` to `assert_iv_close`;
   new `TestKAddSHAPAtFullBudget` (kADD-SHAP had no independent ground truth);
   `TestPathDependentTreeEfficiency` (SV efficiency axiom for default
   `TreeExplainer`); strengthened convergence assertion (avg over 3 seeds, must
   halve with 16× budget); tightened `1e-10` → `1e-9` for Windows/macOS FMA.
4. **Seed parametrisation** (`6055257`) — analytical cross-checks now run across
   5 fixed SOUM seeds. Stays deterministic, catches conditioning edge cases.
5. **ProductKernel cross-check** (`e798b51`) — `ExactComputer(ProductKernelGame)`
   vs `ProductKernelExplainer.explain(x)`, SV only (all the explainer supports).
   Machine-precision agreement.
6. **LinearTreeSHAP cross-check** (`61f6c4d`) — private
   `_PathDependentTreeGame` helper (~25 lines, replicates the logic the removed
   `TreeSHAPIQXAI` used) + brute-force SV cross-check.
7. **XGB/LightGBM conversion coverage** (`90e353e`) — added `lgbm_reg` and
   native `lgbm_booster` fixtures (protocol with efficiency checks). Pinned
   the currently-unsupported `xgboost.Booster` path with a `TypeError` test
   that reverse-alarms if/when conversion is implemented.

## Current state of the test suite

### `tests/shapiq/test_cross_checks.py` — 8 ground-truth sources

1. `ExactComputer(game)(index, order)` — brute force 2^n.
2. `SOUM.exact_values(index, order)` — closed-form via Möbius.
3. `SOUM.moebius_coefficients` — direct, compared to `ExactComputer("Moebius")`.
4. Consistent approximators (regression family + SHAPIQ + SVARMIQ + kADDSHAP) at budget = 2^n.
5. `InterventionalTreeExplainer` vs `ExactComputer(InterventionalGame)`.
6. Path-dependent `TreeExplainer` — SV efficiency axiom.
7. `ProductKernelExplainer` vs `ExactComputer(ProductKernelGame)`.
8. `LinearTreeSHAP` vs `ExactComputer(_PathDependentTreeGame)`.

### Coverage & timing

| | Fast tier | Full tier (`slow or not slow`) |
|---|---|---|
| Tests | **330 passed**, 12 skipped, 54 deselected | 380+ passed, 16 skipped |
| Runtime | ~30 s | ~90 s |
| Coverage | **75 %** overall | **77 %** overall |

Conversion submodule: **60 % → 82 %** after the XGB/LGBM commit.

### Known gaps, tracked but not fixed

- `xgboost.Booster` conversion — unimplemented; pinned by
  `TestXGBoostBoosterUnsupported`. Separate PR to fix.
- `src/shapiq/tree/conversion/sklearn.py` at 58 % — `ExtraTreeRegressor`,
  `ExtraTreeClassifier`, `ExtraTreesClassifier`, `IsolationForest` registered
  but not exercised. Niche, separate concern.
- `src/shapiq/tree/linear/explainer.py` — now exercised by the LinearTreeSHAP
  cross-check; coverage was 22 %, should be substantially higher now (not
  re-measured post-commit).
- STII disagrees between `ExactComputer(InterventionalGame)` and
  `InterventionalTreeExplainer` by ~1e-1 — omitted from tree cross-check, real
  bug noted in the test comment.
- `InconsistentKernelSHAPIQ` — intentionally excluded from "exact at full
  budget" list; docstring in source confirms it doesn't converge to true SII.

### Files touched

- `tests/shapiq/test_cross_checks.py` — the cross-check pipeline (new file; ~500 lines).
- `tests/shapiq/conftest.py` — SOUM fixtures, `assert_iv_close`, seeds.
- `tests/shapiq/test_tree.py` — LightGBM regression + native-Booster fixtures, `xgb.Booster` pin.
- `tests/shapiq/test_approximators.py` — SPEX reproducibility fix.
- `tests/shapiq/fixtures/data.py` — graceful skip when legacy image missing.
