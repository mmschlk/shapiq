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

## Games and benchmark gotchas (observed 2026-10-06)

- Every git-tracked file under `src/` ships in the `shapiq` wheel (setuptools_scm
  file finder): the 1.7.0 wheel carried ~89 MB of `shapiq_games` CSVs and JPEGs.
  Never add data files (CSVs, images, weights, precomputed game values) under
  `src/`; datasets are fetched and cached locally. See
  `docs/design/games_and_benchmark.md` for the design.
- `shapiq_games` datasets are downloaded on first use into `~/.cache/shapiq` (override with
  `$SHAPIQ_DATA_DIR`). New games must follow the contract in `shapiq_games/_base.py`, and
  every family is checked by `tests/shapiq_games/test_contract.py`. Tests that download
  pretrained models or OpenML/UCI data only run with `SHAPIQ_RUN_HEAVY_TESTS=1`.
- A pandas CSV round trip is not lossless by default: `DataFrame.to_csv` can drop the last
  significant digit of a float, and `pd.read_csv`'s default fast parser is not correctly
  rounded for 17-digit strings. Data caches that must reproduce values exactly write with
  `float_format="%.17g"` and read with `float_precision="round_trip"` (see
  `shapiq_games/datasets/_tabular.py`).
- `ucimlrepo` does not serve every UCI dataset as the website does (observed 2026-10-07): it
  refuses to export arrhythmia (5) and thyroid (102), has only the small soybean table, and
  some column names differ from the UCI files (`famiily`). Before pointing a loader at it, run
  `SHAPIQ_RUN_HEAVY_TESTS=1 uv run pytest tests/shapiq_games/test_heavy_games.py -k upstream`.
- Hand numpy arrays to torch as a copy (`np.array(x)`), not `np.ascontiguousarray(x)`: torch
  rejects negative strides, and a one-row reversed view counts as contiguous, so
  `ascontiguousarray` returns it unchanged (the ViT game crashed on `coalitions[::-1]`).
- With the full core suite under `pytest -n 8`, the ProxySPEX tests
  (`test_approximator_proxyspex.py`, `test_explainer_proxy_integration.py`) time out or raise
  `LightGBMError: Replace training data failed`; run serially, they pass in seconds. Re-run
  them alone before blaming a change. The SPEX tests also need the optional `sparse` extra
  (`sparse-transform`, `galois`); without it they fail with `ModuleNotFoundError`.

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

### Run Pre-commit (takes only 3s)

```bash
uv run pre-commit run --all-files
```
