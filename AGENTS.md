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
- Zero-cover nodes are NOT always unreachable: the explained point's routing can enter
  one (NaN default routing, out-of-range values; common in CatBoost's oblivious trees),
  often only partway down after cover-weighting an absent feature above it. The
  path-dependent cover ratio below a zero-cover node is defined in ONE place,
  `create_edge_tree_arrays` in `src/shapiq/tree/conversion/cext/cext.cc` (equal split,
  1/2 per child), mirrored by the Python reference
  `tests/.../tests_tree_explainer/conversion_reference.py`; keep the two in sync.
- Binary boosters (sklearn GB/HistGB, XGBoost, LightGBM, CatBoost) have ONE raw output,
  the class-1 log-odds. `class_label=0` must negate it (leaves incl. base score / bias);
  the parsers do this via `binary_class_sign` in `conversion/cext/converter.hpp` (sklearn:
  `_binary_class_sign`) and flag the trees with `TreeModel.negated_class_one`. A model
  counts as binary if it has one output AND is a classifier: the serialized objective/loss
  says so, or Python passes `is_classifier` (custom objectives are not serialized as
  binary). Woodelf ignores `class_index` for single-output models (deliberately, to match
  shap), so `TreeExplainer._run_woodelf` requests class 1 and negates the result when the
  trees carry that flag.
- `class_label` handling must agree across all converters AND Woodelf (which loads the
  original model itself): `None` = class 1, out-of-range / negative = `ValueError`
  (`check_class_label` in `conversion/common.py` and `cext/converter.hpp`). `-1` is the C
  parsers' "unspecified" sentinel, so negative labels are rejected in `convert_tree_model`.
  `test_woodelf_matches_shapiq_for_every_class_index` guards the agreement.
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
