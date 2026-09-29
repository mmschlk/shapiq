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

## Benchmark integration notes

- The core `src/shapiq/tree/interventional/game.py` and the legacy
  `src/shapiq_games/benchmark/interventionaltreeshapiq_xai/base.py` both define
  `InterventionalGame`, but their classifier output handling differs. Use the
  core game and verify that its output scale matches the exact solver.
- `PathdependentComputer` passes index/order to an explanation call that can
  ignore them. Configure the exact tree solver with the target/order at
  construction and validate its returned metadata.
- Nearest-neighbor explanation games use training rows as players, not features.
  Unweighted KNN utility divides by fixed `k`, even for coalitions smaller than
  `k`; preserve that rule when comparing with `KNNExplainer` ground truth.
- Hopper's login host has a different CPU model from its Slurm compute nodes.
  Record worker affinity and CPU model. `OverSubscribe=NO` alone does not prove
  an exclusive node: also verify that the job owns the node's full CPU count.
  Slurm logs that must be read from the login host need a shared workspace path;
  `/tmp` on a compute node is node-local.
- `RegressionMSR.valid_indices` is inherited from `ProxySHAP` and is broader than
  its constructor's actual SV/BV support. Benchmark capability catalogs must
  follow constructor checks, not inherited registries alone. `kADDSHAP` with
  `max_order=1` returns SV; its default order-two configuration is a different
  target/configuration and must not be silently relabeled as SV.

## Benchmark export gotchas

- Zero-energy truth must be identified from the frozen game, even if every run is
  pending. Both Python summaries and browser filters exclude it for all methods;
  preset identity still includes the originally selected game IDs.
- A public reproduction ZIP is a separate export path from the website. Do not
  retain raw estimator exception messages in either: they can contain local paths.
