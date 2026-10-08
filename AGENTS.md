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

## Games and benchmark gotchas (observed 2026-10-06)

- Every git-tracked file under `src/` ships in the `shapiq` wheel (setuptools_scm
  file finder): the 1.7.0 wheel carried ~89 MB of `shapiq_games` CSVs and JPEGs.
  Never add data files (CSVs, images, weights, precomputed game values) under
  `src/`; datasets are fetched and cached locally (`shapiq_benchmark.datasets`).
- `shapiq_games` holds game definitions only, built from objects (a model, data, a point). Datasets,
  the model registry, and building games from names live in `shapiq_benchmark` (`datasets/`,
  `models.py`, `setups/`); do not add dataset loading or a `from_config` to a game. A new game
  follows the contract in `shapiq_games/_base.py` (checked by `tests/shapiq_games/test_contract.py`)
  and, unless it is synthetic, gets a typed setup (`tests/shapiq_benchmark/test_setups.py`
  fails while one is missing).
- `shapiq_benchmark` datasets are downloaded on first use into `~/.cache/shapiq` (override with
  `$SHAPIQ_DATA_DIR`). Tests that download pretrained models or OpenML/UCI data only run with
  `SHAPIQ_RUN_HEAVY_TESTS=1`.
- Run `tests/shapiq_games` and `tests/shapiq_benchmark` serially (observed 2026-10-07): under
  `pytest -n 8` on 4 cores, the multi-threaded model libraries (XGBoost inside the conditional
  imputer, k-means) oversubscribe the cores, and the contract tests took 410 s instead of 4 s.
  Both suites together take about a minute serially.
- A pandas CSV round trip is not lossless by default: `DataFrame.to_csv` can drop the last
  significant digit of a float, and `pd.read_csv`'s default fast parser is not correctly
  rounded for 17-digit strings. Data caches that must reproduce values exactly write with
  `float_format="%.17g"` and read with `float_precision="round_trip"` (see
  `shapiq_benchmark/datasets/_tabular.py`).
- `ucimlrepo` does not serve every UCI dataset as the website does (observed 2026-10-07): it
  refuses to export arrhythmia (5) and thyroid (102), has only the small soybean table, and
  some column names differ from the UCI files (`famiily`). Before pointing a loader at it, run
  `SHAPIQ_RUN_HEAVY_TESTS=1 uv run pytest tests/shapiq_benchmark/test_heavy.py -k upstream`.
- Hand numpy arrays to torch as a copy (`np.array(x)`), not `np.ascontiguousarray(x)`: torch
  rejects negative strides, and a one-row reversed view counts as contiguous, so
  `ascontiguousarray` returns it unchanged (the ViT game crashed on `coalitions[::-1]`).
- With the full core suite under `pytest -n 8`, the ProxySPEX tests
  (`test_approximator_proxyspex.py`, `test_explainer_proxy_integration.py`) time out or raise
  `LightGBMError: Replace training data failed`; run serially, they pass in seconds. Re-run
  them alone before blaming a change. The SPEX tests also need the optional `sparse` extra
  (`sparse-transform`, `galois`); without it they fail with `ModuleNotFoundError`.

## Games and benchmark gotchas (observed 2026-10-07, self-review)

- A torch forward pass on CPU picks kernels by batch size, so ResNet-18 and the ViT gave the
  same coalition values up to 1e-6 apart in a batch of 16 and alone. The image games pad every
  pass to `batch_size` (`vision/_batching.py`) and return the stored v(∅) for empty rows; keep
  both, or the batch-composition contract breaks for the real models (the contract tests only
  use a numpy classifier).
- `DataFrame.to_numpy(dtype=float)` can return a read-only view under pandas 3 (copy-on-write)
  when no conversion is needed; pass `copy=True` before writing into the array.
- Observational value functions are not null-player games: with the conditional imputer, a
  feature the model ignores still gets a nonzero Shapley value (conditioning on it changes the
  sampled background). The null-player contract test covers the interventional games only.
- Setup fields are stored read-only in JSON form (`_FrozenDict`, tuples): derive variants with
  `dataclasses.replace`, and give a setup subclass its own `name=`, or creating it raises.
- LightGBM ignores `subsample` unless `subsample_freq` > 0 (the tuned presets carried a
  `subsample` that did nothing).
- shapiq's Monte Carlo approximators (SHAP-IQ, SVARM-IQ) estimate FSII and FBII of the top order
  only (`estimate.min_order == order`); the runner scores those on that order.

## Games and benchmark gotchas (observed 2026-10-08, DINOv2 / CLIP / TabPFN v3)

- tabpfn 9.1 asks for a Prior Labs license token (`TABPFN_TOKEN`, or a token cached by a
  one-time browser login) before downloading the gated checkpoints v2.5, v2.6, v3 and v3.5 (its
  default); TabPFN v2 is license-free, and a checkpoint already on disk is never checked. Our
  TabPFN default is therefore v2 everywhere (`DEFAULT_TABPFN_VERSION`), users opt into other
  versions with `version=`, and nothing in the repository (tests, docs examples, CI) may need a
  token. Do not fetch the checkpoints directly to get around the license step.
- tabpfn 9.1 refuses more than 1,000 training rows on a CPU unless
  `ignore_pretraining_limits=True` (6.4.1 only warned); the registry builds TabPFN on the CPU.
- TabPFN's predictions depend on the `tabpfn` version, not only the checkpoint: the causal game
  on v2.5 moved by up to 0.47 between tabpfn 6.4.1 and 9.1. Cached ground truth of a TabPFN
  setup is stale after a tabpfn upgrade (the cache does not track package versions).
- In transformers 5, `CLIPModel.get_text_features` returns an output object, not a tensor (the
  paper script divided it by its norm). Use `text_projection(text_model(...).pooler_output)`.
- A float32 matrix product's summation order depends on the number of rows, so
  `embeddings @ text` changed in the 16th digit with the batch; the CLIP game sums row-wise.
- The Hugging Face DINOv2 checkpoint (`facebook/dinov2-base-imagenet1k-1-layer`) has an all-zero,
  untrained mask token (`use_mask_token=True`, but iBOT's token was not converted). Masking is far
  out of distribution for it (one masked player of 20: probability 0.85 -> 0.001, the same through
  HF's own `bool_masked_pos`), so the DINOv2 game only drops tokens, as the paper's script does;
  do not add masking or image-space removal back for DINOv2.
- Feed the Hugging Face models the processor's own `pixel_values` (resized and cropped in float),
  not a PIL resize rounded to `uint8`: that one-pixel-level difference moved DINOv2's values by up
  to 0.2 against the paper's script. The DINOv2 and CLIP games now match their scripts to 3e-6 and
  1e-7 on CPU; their `image` is the processor's pixels shown as `uint8` (`displayed_image`).
- A scikit-learn tree trained without missing values sends NaN to its larger child; a point can
  follow that path at every split, which makes a NaN-baseline game constant. Use a
  model that learns missing-value directions (`HistGradientBoosting*`) in tests.

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

### Run Pre-commit (takes only 3s)

```bash
uv run pre-commit run --all-files
```
