# AGENTS.md

This role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the Agent.md file to help prevent future agents from having the same issue.

## Commands to interact with the codebase which you should run:

## Conversion package note

Boosting converters live in separate modules such as `xgboost.py`,
`lightgbm.py`, and `catboost.py`, and are hooked up through lazy registrations in
`src/shapiq/tree/conversion/__init__.py`.

## C extension gotchas (learned the hard way)

- Each C extension compiles from ONE source listed in `setup.py`; the files it `#include`s
  (`utils.cpp`, `*.hpp`, `linear_tree_shap.cc`, the shared `tree/cext/subset_tables.hpp`) are not
  tracked by setuptools, and `touch`ing the listed source does not recompile either. Always

  ```bash
  rm -rf build && uv run python setup.py build_ext --inplace
  ```

  and check `stat -f "%Sm %N" build/temp.*/src/shapiq/tree/*/cext/cext.o` is fresh before
  trusting a test or a benchmark.
- The kernels are built with `-ffast-math`, which compiles `std::isnan` to
  `false` and silently breaks missing-value (NaN) routing.
  `-fno-finite-math-only` must stay AFTER `-ffast-math` in `setup.py`.
- Interaction layout (`src/shapiq/tree/cext/subset_tables.hpp`, shared by the interventional
  and the quadrature extension; design in `docs/interventional_kernel_plan.md`): per order a
  sorted table of the interactions that can be non-zero (their features co-occur on a path),
  built once per explainer by `preprocess_subset_tables`. An interaction's position in its
  table is its offset in the output array. Positions are found by a hash index of 128-bit keys
  while `n_features ** min(max_order, widest path) < 2**127`, else no index is built and BOTH
  kernels binary-search the table (`TableSearch`); nothing is refused. A near-dense order-2/3
  table (`complete_above`, interventional 0.5, quadrature never) is completed and addressed by
  rank. Every offset is derived from `counts` in three places that must agree: the header
  (`derive_geometry`), `shapiq/tree/subset_layout.py` and the numpy oracle
  `tests/.../subset_index_reference.py`. Tests asserting "exactly the co-occurring
  interactions" use the `no_completion` fixture.
- Interventional kernel (`src/shapiq/tree/interventional/cext/`): two fills produce leaf events
  (`cohort_fill.hpp` over the background data, `single_reference_fill.hpp` for one row, where
  E/R follow from the leaf's position) and one leaf update consumes them (`leaf_update.hpp`,
  templated on order, rank and search mode). Weights are one array per call
  (`build_weight_table`); there is no cache across calls because ProxySHAP builds one explainer
  per proxy tree and calls it once. Test oracle: the cohort fill equals the average of the
  single-reference fills (`_average_single_reference`).
- Per-thread scratch is created per call and sized to the longest path; never allocate per
  work item inside an OpenMP loop (a per-(tree, reference) `reserve` made 14 threads 2-4x slower
  than 4 on macOS) and never size per-leaf buffers by a constant (a `feats[64]` buffer raised on
  paths with more than 64 features; `test_both_fills_handle_paths_with_more_than_64_features`).
  Small thread-private output blocks are merged serially under a lock, large ones in parallel:
  a barrier on 14 threads costs more than summing a few thousand doubles.
- Per-leaf code must be timed on deep models too: a removed weight memo (`unordered_map` keyed
  by an 8-tuple) was 70% of a chain-tree explanation at order 3 and invisible on depth-6
  boosters. Add an unlimited-depth forest and a chain tree to any such benchmark.
- Quadrature kernel: the hot factor `h` and the cover `c` stay separate because the papers'
  `prod(1 + (h/c - 1) t)` overflows on deep hot chains with small covers (issue #545);
  `n_quadrature_points` is a fixed 8 (13-35% faster than `ceil(d/2)` beyond ~16 distinct
  features, deviations within 2e-14 up to depth 100). Its binding validates array dimensions
  before reading `PyArray_DIM(arr, 1)`, which reads past the dims block of a 1-D array.
- sklearn `HistGradientBoosting*` with `categorical_features` routes input
  through an internal ColumnTransformer that REORDERS features (categorical
  columns first) and ordinal-encodes the raw category values; the tree
  predictors live in that transformed space. Converters must map feature
  indices and category codes back (see `_convert_hist_tree_predictor` in
  `src/shapiq/tree/conversion/sklearn.py`).
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

## Test suite gotchas

- Do not run `tests/shapiq` with `pytest -n` (or next to another test run) on a small machine:
  the LightGBM proxies of `ProxySPEX` use all cores with OpenMP spin-waiting, and with the cores
  oversubscribed `test_approximator_proxyspex.py` goes from ~5s to many minutes and looks hung
  inside `Booster.update`. CI runs the suite serially.
- Python 3.15 environments have no torch, tabpfn, catboost, or scikit-image (no wheels yet; see
  the `python_version < '3.15'` markers in `pyproject.toml`). Tests needing them must use
  `pytest.importorskip` instead of importing them (or a module importing them) at module level.
