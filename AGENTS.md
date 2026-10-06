# AGENTS.md

This role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the Agent.md file to help prevent future agents from having the same issue.

## Commands to interact with the codebase which you should run:

## Conversion package note

Boosting converters live in separate modules such as `xgboost.py`,
`lightgbm.py`, and `catboost.py`, and are hooked up through lazy registrations in
`src/shapiq/tree/conversion/__init__.py`.

## C extension gotchas (learned the hard way)

- Each C extension compiles from ONE listed source (`setup.py`); companion files
  like `linear/cext/linear_tree_shap.cc` or `interventional/cext/utils.cpp` are
  `#include`d and NOT tracked by setuptools. Editing only an included file does
  NOT trigger a rebuild, or you will silently test a stale `.so`.
  **`touch`ing the listed `cext.cc` is not enough either** — observed 2026-08-26:
  after `touch src/shapiq/tree/interventional/cext/cext.cc`,
  `uv run python setup.py build_ext --inplace` finished in ~2s, re-linked and
  copied a `.so`, and recompiled NO object file (`build/temp.*/**/cext.o` kept
  its old mtime). Only `rm -rf build` actually recompiles. Use:

  ```bash
  rm -rf build && uv run python setup.py build_ext --inplace
  ```

  and verify with `stat -f "%Sm %N" build/temp.*/src/shapiq/tree/*/cext/cext.o`
  that the object file is fresh before trusting a benchmark or a test result.
- The kernels are built with `-ffast-math`, which compiles `std::isnan` to
  `false` and silently breaks missing-value (NaN) routing.
  `-fno-finite-math-only` must stay AFTER `-ffast-math` in `setup.py`.
- sklearn `HistGradientBoosting*` with `categorical_features` routes input
  through an internal ColumnTransformer that REORDERS features (categorical
  columns first) and ordinal-encodes the raw category values; the tree
  predictors live in that transformed space. Converters must map feature
  indices and category codes back (see `_convert_hist_tree_predictor` in
  `src/shapiq/tree/conversion/sklearn.py`).
- XGBoost routes in-set categorical values to the RIGHT ("yes") child;
  sklearn/LightGBM route them LEFT. The internal `TreeModel` convention is
  "in set -> left"; the XGBoost parser therefore swaps children at categorical
  nodes.

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

### Run Pre-commit (takes only 3s)

```bash
uv run pre-commit run --all-files
```

## LeverageSHAP stability

- Paired sampling at budget `2 * n` leaves `n - 1` independent interior
  directions for `n - 1` free coefficients. Minimum-norm least squares can solve
  the system accurately while amplifying nonadditive residuals enormously.
- The reference repository added a low-budget `0.001 I` ridge safeguard in
  `f3c0427`, then removed it in August 2026 audit commit `04cc121`. Shapiq PR #583
  aligned sampling without restoring it. The old warning said `1e-6`, but the
  executed Gram penalty was `1e-3`; fixed-count weights use the same scale here.
- The projected Gram matrix always has the efficiency nullspace. Its full
  condition number is therefore not a reliable test for statistical instability
  in the remaining directions. Keep low-budget regularization explicit and
  preserve the unregularized/full-enumeration paths in regression tests.

## KernelSHAP-IQ regression checks

- Consistent KernelSHAP-IQ solves a separate weighted design at each interaction
  order, so rank-deficient higher-order designs need the shared stable solver too.
  InconsistentKernelSHAPIQ's Bernoulli design includes an all-zero empty-term
  column: its previous normal-equation path already fell back to SVD. Do not
  promise a numerical improvement for every regression subclass from the solver
  change, or impose efficiency on the sum of all SII orders; only k-SII aggregates
  those orders into an efficient explanation.
