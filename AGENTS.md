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
- Zero-cover nodes are NOT always unreachable: the explained point's routing can enter
  one (NaN default routing, out-of-range values; common in CatBoost's oblivious trees),
  often only partway down after cover-weighting an absent feature above it. The
  path-dependent cover ratio below a zero-cover node is defined in ONE place,
  `create_edge_tree_arrays` in `src/shapiq/tree/conversion/cext/cext.cc` (equal split,
  1/2 per child), mirrored by the Python reference
  `tests/.../tests_tree_explainer/conversion_reference.py`; keep the two in sync.
- Path-dependent SVs of CatBoost models do NOT match CatBoost's native `ShapValues`
  (observed 2026-10-13, gaps up to ~0.1), and that is expected: path-dependent SHAP depends
  on the level order of an oblivious tree. shapiq puts `splits[0]` (the first split chosen in
  training) at the root (`fill_catboost_tree_node` in `conversion/cext/catboost_json.cc`), and
  CatBoost walks the levels in reverse. Both match a brute-force reference for their order to
  1e-15. CatBoost also adds the explained pool's leaf counts to the model's leaf weights when a
  zero-weight leaf has a non-zero value, so its base value can change with the batch.
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
- Core has no dataset loaders anymore (`shapiq.datasets` was removed); examples and the
  README load data with `shapiq_benchmark.datasets.load_dataset` or scikit-learn's fetchers.
  Keep the three CSVs in `data/` at the repository root anyway: released versions (up to 1.7.x)
  download their `shapiq.load_*` files from `main/data/` on first use, so deleting them breaks
  every release's loaders. scikit-learn's California housing differs from that CSV by one unit
  in the last digit of three columns (the CSV lost it); the core test fixture uses scikit-learn.
- scikit-learn's fetchers race when several processes fetch into one data folder (observed
  2026-10-08 under `pytest -n logical` in the coverage job: one worker deletes `cal_housing.tgz`
  while another opens it, 134 errors). The core fixture fetches into one folder per xdist worker,
  and `shapiq_benchmark` into a temporary folder per download (its CSV cache is atomic).
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
- A setup's cache key covers its fields, its class `version` and `RECIPE_VERSION`
  (`setups/_base.py`). Bump `version` when one setup's `build` changes its game, and
  `RECIPE_VERSION` when shared code does (datasets and their preprocessing, `sample_rows`,
  `stratified_rows`, splits, registry defaults); otherwise stale ground truth is served silently.
- The image games derive from `RegionGame` (`vision/_region_game.py`): a subclass sets `image`,
  `regions` and `_fill` (`None` in token space), implements `_evaluate`, and calls
  `super().__init__` last; do not override `value_function` (it answers the empty coalition with
  the stored value). The token backends (ViT, DINOv2) return every class probability; the game
  picks the class.
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
- SigLIP texts must be padded to 64 tokens (`padding="max_length", max_length=64`) and
  lowercased, as the models were trained; the SigLIP 2 tokenizer of transformers 5.3 does not
  lowercase by itself ("A Photo of a Dog" scored 0.046 instead of 0.094). Its tokenizers need
  `sentencepiece` and `protobuf` (both in the games extra). Only the fixed-resolution SigLIP 2
  checkpoints load as `SiglipModel`; the NaFlex ones are `Siglip2Model` with patchified input.
- SigLIP has no class token (it pools all tokens by attention), so token dropping uses
  `n_prefix=0` and the empty coalition is an empty sequence, which pools to the head's output
  bias. SigLIP's processor resizes the whole image to 224 x 224 without cropping.

## Games performance gotchas (observed 2026-10-09)

- The games' values must not depend on the batch, so a speed-up must keep every float's
  arithmetic, not only the result up to rounding. Check a rewrite against the old code value by
  value (`np.array_equal`), one coalition at a time and in a batch.
- Batch model calls only for models whose output for a row does not depend on the other rows of
  the call: tree models (`predicts_row_by_row` in `shapiq_games/_base.py`). A BLAS or torch model
  can round a row differently in a call with more rows (one row of 40 changed in the last digit).
  The core marginal and baseline imputers batch every model and are only exact up to that.
- numpy sums are not always in order: a reduction over a `(k, 1)` or `(1, n)` array collapses to
  one dimension and sums pairwise, while a `(k, n)` axis-0 sum adds row after row. Use
  `np.cumsum(...)[-1]` (or `np.add.accumulate`) where the order must be fixed, e.g. the
  path-dependent tree game summing its leaves in breadth-first order. Python's built-in `sum()`
  of floats is compensated (Neumaier) since 3.12, so it differs from a numpy running sum too.
- Library overhead dominates cheap games: `scipy.stats.entropy` costs about 0.3 ms per call,
  scikit-learn metrics about 0.4 ms, a scikit-learn forest's `predict` about 1 to 5 ms (joblib).
  `_training.py` has numpy versions of the named metrics; `tests/shapiq_games` checks that they
  equal scikit-learn's bit for bit, so an upgrade that changes scikit-learn's arithmetic fails there.
- XGBoost's converted trees hold float32 node weights and leaf values. Computing with them in
  float32 (e.g. `1 - share` of a float32 share) cost the path-dependent game 1e-6 against TreeSHAP;
  cast to float64 first.
- The path-dependent tree game evaluates a tree level by level (one numpy operation per level for
  all coalitions, at least 64 coalitions per pass): on a deep 10-tree forest 100x faster than node
  by node for one coalition, 10x for 64 (the batches of the permutation approximators), about the
  same at 1,000, but 1.45x slower at 4,096 (the level arrays are copied more often than per-node
  vectors). A second, node-by-node path for large batches was left out for readability.

## Citation and docs gotchas (observed 2026-10-12)

- `docs/source/references.bib` is the maintainer's thesis bibliography; keep its entries and keys
  as they are and add new ones in its style (`Surname.Year` keys with an `a`/`b` suffix on
  collisions, e.g. `Jia.2019` is the AISTATS paper and `Jia.2019a` the KNN one; venue names like
  `International Conference on Machine Learning (ICML)`, PMLR `series`/`volume`, `--` page
  ranges). Check a key's title before citing it: several keys mean another work than a guess
  from the name (`Covert.2021` is "Explaining by Removing", `Covert.2021b` "Improving KernelSHAP").
  The file is grouped by entry type, the `@inproceedings` further by venue under `#`/`%` comment
  headers, by year within a venue; copy the venue string of an existing entry (AAAI is
  `Conference on Artificial Intelligence (AAAI)`). pybtex's `alpha` labels and its sorting keep
  LaTeX accents and braced names whole (`[PKHullermeier26]`, `{Banzhaf III}` after "Z", observed
  2026-10-16); `conf.py` builds both from plain names (`_plain_person`), so LaTeX accents and
  Unicode both work. The text cites by author and year; works cited alike get a letter in the
  order of their keys (`Kolpaczki.2024a` to `2024c`, `_year_letters`). To verify bibliographic
  data (observed 2026-10-14): WebFetch cannot reach jmlr.org or link.springer.com, and Springer and dblp answer
  `curl` with a bot check (dblp denies a headless browser too); dblp's SPARQL endpoint
  (`sparql.dblp.org/sparql`), Crossref (`api.crossref.org/works/<doi>`), OpenAlex and the
  publishers' own proceedings pages work. A preprint that gets published becomes the published
  entry, its key following the publication year (`Wever.2025` became `Wever.2026`, AAAI-26), with
  dblp's title. dblp has no pages for NeurIPS, and for NeurIPS 2025 the proceedings site's BibTeX
  numbers the main track alone (observed 2026-10-16); the bib takes the pages of the DOI's
  Crossref record, which continue the numbering of NeurIPS 2024 used by the thesis.
- Its titles are written for LaTeX styles that print them as written, and some DOIs escape `_`
  as `\_`. The docs therefore use the `shapiq_alpha` pybtex style registered in
  `docs/source/conf.py` (titles kept as written, `\_` unescaped in links); plain `alpha`
  lowercases titles ("shapley values") and breaks those DOI links.
- Every citation of the docs (docstrings, gallery examples, pages, the timeline) uses
  `:cite:t:`/`:cite:p:`, so the bibliography page (`references.rst`, `:cited:`) lists every cited
  work with links back to where it is cited. Do not use `:footcite:` or `.. footbibliography::`
  (observed 2026-10-13): footnote citations never reached the bibliography page, which missed 46
  of 98 cited works. `test_the_docs_cite_with_the_cite_role` in
  `tests/shapiq_games/test_docstring_examples.py` fails on a `:footcite:`.
- `uv run pre-commit` in a git worktree: the `ty` hook runs `uv run ty check` without
  `--no-sync`, so with `UV_PROJECT_ENVIRONMENT` pointing at the main checkout's `.venv` it
  re-syncs that shared environment to the worktree (editable `shapiq` re-pointed, locked packages
  changed), and ty's `python = "./.venv"` fails where the worktree has no `.venv`. In a worktree,
  run `SKIP=ty uv run --no-sync pre-commit run --all-files` and
  `uv run --no-sync ty check --python <venv>` separately.

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

CI does not build the docs by default (observed 2026-10-12): the `Documentation` workflow runs
only on pull requests with the label `build docs` or from the Actions tab, with warnings as errors
(`-W`). Before pushing a docs change, build locally and check the output for `WARNING`. A new
public name of `shapiq_games` or `shapiq_benchmark` (anything in an `__all__`) also needs a line in
`docs/source/api_reference_games.rst` or `api_reference_benchmark.rst`; the docstring-example
tests fail until it has one.

Docs layout (observed 2026-10-12):

- Every folder `examples/<group>/` is its own sphinx-gallery gallery (`EXAMPLE_GROUPS` in
  `docs/source/conf.py`), linked from the hand-written landing page `docs/source/examples.rst`.
  A new example goes into its group's entry of `EXAMPLE_ORDER` in `conf.py` (unlisted files come
  last), and a new group needs `EXAMPLE_GROUPS`, a `GALLERY_HEADER.rst` and a card in
  `examples.rst`.
- The first paragraph of an example's docstring is the text of its gallery card, and sphinx-gallery
  strips roles badly (a citation role became ":footciteMusco.2025", `:math:` raw LaTeX). Keep that
  paragraph plain; put citations and math into the next one.
- `nitpicky` is off, so an unresolvable cross-reference renders as plain text without a warning.
  The core API reference documents objects under their subpackage (`shapiq.approximator.KernelSHAP`);
  a `missing-reference` hook in `conf.py` links the public aliases (`shapiq.KernelSHAP`) and bare
  names that match one documented object. Check new references in the built HTML.
- The sidebar and the previous/next links name the generated API pages by the last part of their
  dotted name (`_short_navigation_titles` in `conf.py`); the page heading keeps the full path. On
  `api_reference_games.rst` the game sections list games only; helpers, constants and types go under
  Utilities.
- The timeline page reads `docs/source/_data/timeline.toml` (directive in
  `docs/source/_ext/shapiq_timeline.py`); its API, example and citation references are checked by
  Sphinx and warn at the entry's line. A new method in shapiq deserves an entry there.
- The logo is inlined as SVG (`_inline_logo` in `conf.py`, `_templates/sidebar/brand.html`) so that
  `_static/js/logo.js` can swap the landing page's `<img>` for a copy and move its players (only
  there: the sidebar's logo is a link home on every page and stays still).
  Edit `_static/logo/logo_shapiq_{light,dark}.svg` only: the script finds the players as circles
  with `r > 15`, the caps as smaller circles and the lines as two-point paths with `fill="none"`.
- Moved pages keep their old addresses: `_REDIRECTS` in `conf.py` writes a redirect page for each
  after the build. Add an entry when a page (e.g. a gallery example) moves.
- In a git worktree, `src/` has no compiled tree extensions (the `*.so` files are git-ignored), but
  `conf.py` puts the worktree's `src` first on `sys.path`: copy the `*.so` files from the main
  checkout before building the docs there.

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
