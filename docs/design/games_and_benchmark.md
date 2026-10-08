# Design: harmonizing `shapiq_games` and `shapiq_benchmark`

Status: implemented on `feature/games-benchmark-harmonization` · 2026-10-06

## Goal

`shapiq_games` becomes a curated, tested catalog of cooperative games. `shapiq_benchmark`
becomes a small harness whose central idea stays what it is today: **a benchmark is a game
plus a computer** that produces the exact (ground-truth) values of that game. Together they
are the platform that checks whether an implementation of shapiq (today's or a future
rewrite) computes the right numbers.

## Decisions already taken

| # | Decision |
|---|----------|
| 1 | Both packages stay where they are and keep shipping inside the `shapiq` distribution (code only). |
| 2 | Every game family is kept, including the model-specific ones (trees, KNN, product kernel, …). |
| 3 | All datasets stay usable (32 classic, of which 10 synthetic, + 51 TabArena = 83). |
| 4 | No hosting and no website: everything runs and caches locally, rudimentary on purpose. |
| 5 | No data files ship in any wheel: no CSVs, images, weights or precomputed game values. This includes the three CSVs in core `shapiq/datasets/data`. |
| 6 | Breaking changes in both packages are fine. PR #602 (Teal) integrates on top of this afterwards. |
| 7 | Every game that nothing in core uses moves into `shapiq_games` (see "Games moving out of core"). |
| 8 | Important core bugs found during the audit (below) are fixed in separate PRs after this work. |
| 9 | **Core stays (almost) untouched.** The only core changes are the ones listed in "Core changes"; everything else is built in `shapiq_games` and `shapiq_benchmark`. |

## Dependency direction

```
shapiq_benchmark  ──►  shapiq_games  ──►  shapiq (core)
  setups, datasets,      game definitions      algorithms: ExactComputer, MoebiusConverter,
  model registry,        (the game zoo)        TreeSHAPIQ / TreeExplainer, InterventionalTreeSHAPIQ,
  computers, Benchmark,                        KNNExplainer, ProductKernelComputer, imputers
  metrics, runner, cache
```

Nothing ever points the other way. Games know nothing about datasets, model names, computers or
caches, and core knows nothing about either package (core *tests* may import games, as they
already do for `DummyGame`).

## `shapiq_games`

### Layout

```
shapiq_games/
  _base.py              the game contract (bool coalitions, explicit x, class index)
  _training.py          clone-per-coalition training, metrics, constant predictors
  typing.py             the games' Literal choices (Task, ImputerName, ...), next to shapiq.typing
  synthetic/            DummyGame, UnanimityGame, SOUM, RandomTableGame (replaces RandomGame)
  tree/                 PathDependentTreeGame (was TreeSHAPIQXAI), InterventionalTreeGame (moved from core)
  nn/                   KNNGame, WeightedKNNGame, ThresholdNNGame (moved from core)
  kernel/               ProductKernelGame (moved from core)
  local_xai.py          TabularLocalExplanation (marginal / conditional / baseline imputers, TabPFN)
  global_xai.py         TabularGlobalExplanation (SAGE-like)
  feature_selection.py  FeatureSelection
  valuation.py          DataValuation, DatasetValuation (one shared implementation)
  ensemble_selection.py EnsembleSelection, RandomForestEnsembleSelection
  uncertainty.py        UncertaintyExplanation
  clustering.py         ClusterExplanation
  unsupervised.py       UnsupervisedData
  causal.py             GlobalConfoundingXAI, LocalConfoundingXAI
  vision/               ImageClassifier (ViT, DINOv2, ResNet, custom), ImageTextSimilarity (CLIP)
                                                                           [torch, transformers]
  language.py           SentimentAnalysis                                  [transformers]
```

There is **one class per game family**, configured by arguments. The 153 classes today
(82 near-identical local-XAI dataset classes, 13 interventional ones, 18 valuation/selection
subclasses, …) collapse to roughly 25. The `benchmark` sub-package name goes away: these are
games, and benchmarking is `shapiq_benchmark`'s job.

### Construction

`shapiq_games` is a collection of game definitions: it shows how a problem becomes a cooperative
game. The constructor of every game takes plain objects (a model, data, a point, an image, a
text) and nothing benchmark-specific; the task of a model-based game is read off the model, and
the causal games fit their own reference effect. Every class docstring shows this construction,
and `tests/shapiq_games/test_docstring_examples.py` runs all docstring examples.

```python
game = FeatureSelection(DecisionTreeRegressor(), x_train, y_train, x_test, y_test)
```

Loading datasets, fitting models by name, and identifying games for a cache are benchmark
concerns. They live in the setups of `shapiq_benchmark` (below), not in the games.

### Game contract (enforced by `tests/shapiq_games`)

1. Subclasses `shapiq.Game` and implements `value_function`.
2. **Deterministic.** v(S) is a pure function of the constructor arguments, including `random_state` (the name core shapiq uses). It must not depend on call order, batch composition, repetition or process. Randomness is drawn once at construction from `np.random.default_rng(random_state)`. No global or stateful RNG is used during evaluation (the conditional imputer of core, which has one, is reseeded before every evaluation).
3. **Explicit explained point.** Local games take `x` as an index into the explanation data or as an array. The default is index `0`, never a random point. One name, `x`, everywhere (today it is `x`, `x_explain`, `target_instance`, `explain_point` and `x_explain_path`). The exception is the causal games, which follow the causal-inference convention: `x` is the covariate matrix and the explained unit is `unit`. The image setup picks its image by `index` in the dataset.
4. **Explicit class.** Classification games resolve `class_index` at construction and store it as an int. Following the shapiq explainers, `None` means class `1` (previously class 1, class 0 or argmax depending on the game). Image games default to the class predicted on the image.
5. **Meaningful empty coalition.** v(∅) comes from the game's semantics, for example a model with no features predicting the training mean or majority class, and is computed through the value function, so a normalized game is exactly 0 on ∅. Where nothing has a natural score (a model trained on no data, an empty ensemble, no features to cluster), the game takes an explicit `empty_value` (default 0): the valuation, ensemble selection and cluster games. `normalize=True` then centers by that value.
6. **Declared output space.** Each game documents what its values are (probability, margin or log-odds, loss, score) and in which direction a higher value is better.
7. **Typed public attributes** that a computer needs, for example:
   - tree games: `model`, `x`, `class_index`
   - interventional trees: `reference_data`
   - SOUM: its Möbius coefficients
   - KNN games: `model`, `x`, `class_index`
   - product kernel: `model`, `x`
8. **Import hygiene.** No import-time warnings. Optional dependencies (torch, transformers, tabpfn, openml, xgboost, lightgbm, catboost) are imported lazily, with an error that names the missing package.
9. **Named external models.** A pretrained model is identified by its name (a Hugging Face model id, optionally with a `revision=`; a torchvision weights enum). A new model version is a new name; package versions are not tracked.

Each family gets contract tests on small in-memory objects (no downloads, no setups):
- the same coalition repeated gives the same value
- a permuted batch and one-by-one evaluation give the same values (on fresh instances, so a game's cache cannot hide a difference)
- two instances with the same arguments are equal
- integer and boolean coalitions give the same values
- centered games are 0 on ∅
- a feature the model ignores is a null player (Shapley value 0) for the interventional value functions (marginal and baseline imputation, global explanation, both tree games); observational ones (the conditional imputer) need not satisfy it

### Games moving out of core

No method in `src/shapiq` uses them. Only `__init__` re-exports and tests reference them.

| Today | Moves to |
|-------|----------|
| `shapiq.tree.InterventionalGame` | `shapiq_games.tree.InterventionalTreeGame` |
| `shapiq.explainer.nn.games.{KNN,WeightedKNN,TNN}ExplainerGame` | `shapiq_games.nn` |
| `shapiq.explainer.product_kernel.game.ProductKernelGame` | `shapiq_games.kernel` |

- `WeightedKNNExplainerGame` instantiated a `WeightedKNNExplainer` to call its private weight-discretization helpers. Their round trip is just rounding to multiples of `2**-n_bits`, so the moved game implements that rounding itself: no core change, and the ground-truth game no longer depends on the explainer it checks.
- Imputers (`MarginalImputer`, `TabPFNImputer`, …) are also `Game`s but are used by core explainers, so they stay in core. `TabularLocalExplanation` wraps them.

### Deleted

- `benchmark/product_kernel/` (three one-line placeholder files), replaced by the moved `ProductKernelGame`
- `tabular/` (an unused second copy of `LocalExplanation`)
- the games-package fork of `InterventionalGame` (it diverges from core on multiclass LightGBM)
- `RandomGame`: it is not a set function, because its values depend on batch position. It is replaced by `RandomTableGame`, a seeded table of values drawn once at construction.
- the 1,536 hardcoded LayerNorm constants in `_vit_setup.py` (use the model's own `layernorm`)
- the California torch model and its weights file
- the deprecation and import warnings in both `__init__`s
- `GameBenchmarkSetup` and `get_x_explain`, replaced by the registries and the game contract

## `shapiq_benchmark`

### Setups

The games are built from objects. A **setup** builds one from names instead, for benchmarks: a
frozen, keyword-only dataclass with exactly the fields its kind of game needs, and a `build()`
that returns the plain game.

```python
setup = TabularLocalExplanationSetup(dataset="adult_census", model="xgboost", x=3)
game = setup.build()                        # a plain shapiq_games.TabularLocalExplanation
benchmark = Benchmark.from_setup(setup)     # exact values cached under setup.key
setup_from_dict(setup.to_dict()) == setup   # the stored form, e.g. for run specifications
```

- **One setup per game**, except the synthetic games, which take only plain values anyway
  (21 setups). Their fields differ with the kind of game: `TabularSetup` holds `dataset`,
  `random_state`, `test_size` and `dataset_params`; `ModelSetup` adds `model`, `preset` and
  `model_params`; the image, text and causal setups have their own fields.
- **Typed.** Fields have defaults and `Literal` choices, and a setup is validated when it is
  created: field types and `Literal` choices, dataset and model names, tuned presets, the task
  of the dataset (the nearest-neighbor and uncertainty setups need classification), and
  cross-field rules (`imputer="tabpfn"` needs `model="tabpfn"`) raise immediately.
- **Frozen JSON form.** The fields are stored as they read back from JSON: read-only dicts
  with string keys, tuples for lists, and ints in float fields as floats. A setup thus cannot
  change under its cache key, is hashable, and `setup_from_dict(setup.to_dict()) == setup`.
  A subclass needs a registered name of its own; otherwise it would share its parent's cache.
- **Key.** `setup.key` hashes the setup's name, its `version` and its fields (SHA-256), stable
  across processes and machines. Runtime fields (device, batch size) are excluded. A setup bumps
  `version` when `build()` changes the game it builds for the same fields. Package and model
  versions are not tracked: a genuinely new model gets a new name.
- **Registry.** Every setup registers under its name (`SETUPS`); `setup_from_dict` rebuilds a
  setup from its dictionary form.

### Datasets

- **One registry**, name → `DatasetSpec` with these fields:
  - source
  - `task` ("classification" or "regression"), declared explicitly rather than guessed from the labels
  - preprocessing
  - feature names
- **Sources.** Every real-world dataset comes from its original source; nothing is served from
  this repository:
  - OpenML by dataset id (ids are immutable): adult (1590), amazon (1457), arrhythmia (5),
    bike sharing (42713), bioresponse (4134), leukemia (45090), micro-mass (1515), and the 51
    TabArena datasets;
  - the UCI repository via `ucimlrepo`: annealing, hepatitis, ionosphere, mushroom, nursery, zoo;
  - the UCI repository's raw files: soybean, thyroid, wine quality, real estate, forest fires
    (`ucimlrepo` does not serve arrhythmia and thyroid, and has only the small soybean table);
  - scikit-learn: breast cancer (bundled) and California housing (`fetch_california_housing`);
  - shap's data folder: NHANES I and communities and crime;
  - fast.ai: Imagenette (below);
  - seeded generators for synthetic data.

  Single files are checked against a pinned SHA-256 hash. Upstream tables (OpenML, UCI,
  scikit-learn) are cached as a lossless CSV and checked against the shape the loaders were
  written for, so a changed upstream fails loudly. `tests/shapiq_benchmark/test_heavy.py` compares every live
  upstream table with the previously bundled file; all 16 match (checked on 2026-10-07). The
  loaders then give bit-identical datasets for 16 of the 18 bundled tables; bike sharing and
  California housing differ by at most 5e-15 (relative), because the old CSVs had dropped the
  last digit of some floats and the new values are the exact upstream ones.
- **Local cache.** Files go to `$SHAPIQ_DATA_DIR`, defaulting to `$XDG_CACHE_HOME/shapiq` (or `~/.cache/shapiq`), and are written atomically (temp file + rename) so parallel test workers are safe. A complete extraction or a verified file is never deleted while another process may read it, a cached table of the wrong shape is downloaded again, and the TabArena cache name changes with the OpenML id, the target and the preprocessing version. Nothing is ever written into the installed package.
- **Clean features.** `load_dataset` returns features without missing values, so every registry model can be fitted: empty columns are dropped, a missing category is a category of its own, and the remaining gaps (NHANES I) get the column median. Like the TabArena imputation, this uses all rows before the split. Labels are encoded in their natural order, and `class_names` holds the original labels.
- **Deterministic splits.** Seeded train/test splits, stratified for classification.
- **Data removed from the tree** (the files remain in git history, but no loader reads them from there):
  - `shapiq_games/datasets/data` (81 MB)
  - `shapiq/datasets/data` (8.5 MB)
  - the 31 ImageNet example JPEGs, replaced by Imagenette (below)
- **Core loaders keep their source.** Core's three public loaders (`load_california_housing` & co.) already fall back to downloading from `main/data/` on GitHub when their CSV is missing. Without the bundled CSVs that fallback always runs, so it now caches the downloaded file verbatim in `~/.cache/shapiq/core_datasets` (or `$SHAPIQ_DATA_DIR`) with an atomic write, instead of in the installed package. The repo-root `data/` folder stays; it is not part of any wheel.
- **Removal in token space or image space.** The transformer image games remove players in
  token space. DINOv2 drops their patch tokens from the sequence (the present tokens keep their
  position embeddings), as in the paper, and only so: its checkpoint has no trained mask token
  (it is zeros), and its head averages all patch tokens, so masking would be far out of
  distribution (one masked player of 20 dropped the explained probability from 0.85 to 0.001).
  The vision transformers and CLIP mask them (`mask_strategy="mask"`: a zero mask token, the
  position embedding kept; the ViT default) or drop them (`"remove"`; the CLIP default, as in the
  paper), or with a `fill` remove them in image space, like ResNet-18 and custom classifiers. The
  players are a grid of near-equal rectangular blocks of the token grid (`np.array_split`),
  DINOv2 and CLIP see a `224 x 224` center crop (the game's `image`), and every forward pass is
  padded to the batch size, so values do not depend on the batch. With a full coalition every
  strategy is the plain model. DINOv2 and CLIP get exactly the processor's `pixel_values`, as in
  the paper's scripts, and match them to 3e-6 and 1e-7 on CPU (a PIL resize rounded to `uint8`
  had moved DINOv2's values by up to 0.2).
- **Missing values.** `TabularLocalExplanation(imputer="baseline", baseline=np.nan)` passes absent
  features as missing values to a model that reads them (`np.inf` for TabPFN with
  `PASSTHROUGH_INF`). `baseline` sets the values of core's `BaselineImputer` (one value or one
  per feature) instead of the background mean, so no new imputer was needed. The setups say
  `baseline="missing"` and pick NaN or `+inf` for the model, so their fields stay free of NaN.
- **TabPFN by version.** Users choose the TabPFN version (`version="v3"` in
  `tabpfn_regressor`, in the registry's `model_params`, or in the causal setups'
  `regressor_params`); `shapiq_games._tabpfn.build_tabpfn` builds it with tabpfn's
  `create_default_for_version`, so a new TabPFN needs no code here. The default everywhere is
  TabPFN v2, whose checkpoints download without a license: nothing in the repository (tests,
  docs, CI) needs a Prior Labs license token, and the paper's versions (v2.5 for the causal
  games, v3 for the `+inf` game) are an explicit opt-in. The `games` extra needs `tabpfn>=6.0`
  (version selection); a version the installed tabpfn does not know, and `+inf` before
  tabpfn 8.1, raise. The predictions also depend on the tabpfn version.
- **Images.** The image games use [Imagenette](https://github.com/fastai/imagenette) (fast.ai, Apache-2.0), a ten-class subset of ImageNet with full-size photos: `load_imagenette(split, size)` downloads the official archive (160 or 320 px) from fast.ai, verifies its SHA-256, extracts the JPEGs once into the cache (path-checked), and returns them with their ImageNet class indices, so pretrained ImageNet classifiers explain them directly. The 31 example JPEGs that were served from a pinned commit of this repository are gone.
- **Declared dependencies.** `openml`, `ucimlrepo` and `openpyxl` are part of the `benchmark` extra; before, they were not declared at all.

### Models

- `build_model(name, task, random_state=..., preset=..., **params)` covers decision trees, random forests, XGBoost, LightGBM, CatBoost, MLPs, linear models, SVMs, Gaussian processes, nearest neighbors, and TabPFN (v2 unless `version=` chooses another). `random_state` is always passed through, and models run single-threaded where the library allows it.
- Hyperparameter presets (previously the Optuna JSONs in `shapiq_benchmark`) are Python dicts next to the registry (`preset="tuned"`). The Optuna script stays a benchmark tool and now works for any registered dataset.
- The California torch network (it currently loads weights from a `tests/` path and silently falls back to random weights) is dropped in favor of the seeded generic `mlp` model.

### Computers

```python
class Computer[G: Game](ABC):            # G: the game family it understands
    name: ClassVar[str]                      # part of the cache file names
    game: G                                  # bound at construction
    @classmethod
    def supports_game(cls, game) -> bool: ...
    @classmethod
    def supported_indices(cls) -> tuple[str, ...]: ...   # read from core's declarations
    def supports(self, index: str, order: int) -> bool: ...
    def exact_values(self, index: str, order: int) -> InteractionValues: ...
    def _compute(self, index: str, order: int) -> InteractionValues: ...   # the core call
```

- **Location.** Computers live here as thin adapters around core algorithms. They never re-implement the algorithms, and each one binds to the game types it understands.
- **Index support comes from core, not a parallel list.** `supports(index, order)` reads the declarations core already has for the wrapped algorithm:
  - `ExactComputer.valid_indices`
  - `ValidMoebiusConverterIndices`
  - `TreeSHAPIQIndices`, `InterventionalTreeSHAPIQIndices`, `QuadratureTreeSHAPIndices`
  - `ValidNNExplainerIndices` and `ValidProductKernelExplainerIndices` (with their order-1 checks)

  Constraints core does not declare (for example a maximum order) are added inside the computer, not in core. A drift test calls each core algorithm for every index and order: everything `supports` accepts must run, and everything it rejects must be rejected by core too.
- **Never a silent fallback.** If `supports(index, order)` is false, `exact_values` raises an error instead of computing something else. Today `PathdependentComputer` returns order-1 SV when asked for order-2 k-SII.
- **One output convention.** A computer returns the values of the game *as `game(...)` evaluates it* (same output space, same class): the interactions of order 1 to `order`, with the game's v(∅) as `baseline_value` (0 when normalized). The order-0 term is excluded: core algorithms disagree on it, and no metric uses it. The result is as sparse as the core algorithm's (an interaction that is not stored is 0): nothing in the benchmark lists all `C(n, k)` interactions, so the structured computers serve games with hundreds of players (a 1,000-feature tree at order 4 stores under 1,000 interactions).

| Computer | Games | Core algorithm |
|----------|-------|----------------|
| `BruteForceComputer` | any game, with a player cap (default 20, overridable) | `ExactComputer` |
| `MoebiusComputer` | Dummy, Unanimity, SOUM (also their `Moebius` values themselves) | `MoebiusConverter` |
| `PathDependentTreeComputer` | `PathDependentTreeGame` | `TreeSHAPIQ` / `TreeExplainer` |
| `InterventionalTreeComputer` | `InterventionalTreeGame` | `InterventionalTreeSHAPIQ` |
| `KNNComputer` | KNN, weighted KNN, threshold NN games | `KNNExplainer` & co. |
| `ProductKernelComputer` | `ProductKernelGame` | `ProductKernelComputer` |

`default_computer(game)` picks from a small registry: the structured computer for the game's
type if there is one, otherwise `BruteForceComputer` (and raises an error above the player cap).
A `Benchmark` with a defaulted computer falls back to brute force (within the player cap) for
an index or order its computer does not support; `benchmark.computer_for(index, order)` names
the one used. An explicitly passed computer never falls back.

### Benchmark

```python
benchmark = Benchmark(game)                       # any game; computer = default_computer(game)
benchmark = Benchmark(game, computer=BruteForceComputer(game))
benchmark = Benchmark.from_setup(setup)           # the game of a setup, with a local cache
gt = benchmark.exact_values(index="k-SII", order=2)   # cached under the setup's key
results = run(benchmark, approximators, budgets, index="k-SII", order=2, seeds=[0, 1])
```

- `run` evaluates approximators × budgets × seeds, scores them with the metrics and returns a tidy table, which it can also write to a local CSV/JSON file. Which approximator runs for which index comes from the approximator's existing `valid_indices`; unsupported combinations are skipped and recorded as such, not dropped silently. Approximators that estimate the top order only (SHAP-IQ and SVARM-IQ for FSII and FBII) are scored on that order, marked in the `scored_orders` column, and get no faithfulness.
- Exact values of `Benchmark.from_setup` are cached under `$SHAPIQ_DATA_DIR/ground_truth/<setup name>/<setup key>/<computer>_<index>_<order>.json` using `InteractionValues.to_json_file`, next to a `setup.json` that records what the key stands for. `Benchmark(game)` is not cached. The cache does not track package or model versions: the setup identifies the game, so a genuinely new model (say, a new TabPFN) gets a new model name, and the cache is deleted (or `cache=False` passed) to recompute. TabPFN's predictions change with the `tabpfn` version even for the same checkpoint, so delete the cached values of TabPFN setups after a tabpfn upgrade.
- `LocalXAIBench`, `PathdependentBench`, `InterventionalBench`, `TabPFNBench`, `ImageBench`, `bench_types.py` and `setup.py` go away. Building games from names lives in the setups.

### Chain of trust (enforced by `tests/shapiq_benchmark`)

1. Closed-form values (unanimity games: SV = 1/|T| on T and 0 elsewhere; DummyGame; SOUM through its Möbius representation) validate `BruteForceComputer`.
2. `BruteForceComputer` validates every structured computer on small instances of its game family, for every `(index, order)` that computer claims to support.
3. Only then are structured computers trusted as ground truth at large player counts.

This matters because the structured computers are shapiq's own explainers, which is the code
under test. Brute force on small games is the root of trust.

### Metrics

The metrics are rewritten and property-tested. An identical estimate must score perfectly, and
a uniform shift or rescaling must behave as documented.

- **Errors:** MSE, MAE, SSE and SAE over all interactions of order 1..k. The order-0 entry is excluded explicitly.
- **Sparse.** The metrics visit only the interactions where the ground truth or the estimate is nonzero; the means still divide by the number of all interactions of the compared orders. The ranking metrics leave out pairs that are 0 in both, which carry no ranking information (otherwise τ of a sparse game drifts with the number of players).
- **Ranking:** Kendall τ and Spearman ρ are computed on the values themselves. Today Kendall τ correlates `argsort` positions: a true τ of 0.97 scores 0.55.
- **Top-k:** Precision@k uses top-k by |ground truth|, counting every value tied with the k-th largest. Today `KendallTau@k` uses the k *smallest* values, and `Spearman@k` ignores k.
- **Ties:** values within 1e-6 × max |ground truth| rank as equal, so float noise among the many zeros of a sparse game does not decide a ranking.
- **Faithfulness:** R² of the reconstructed game on seeded coalition samples, for every index rather than FBII only.

## Core changes

Core stays as it is except for exactly these changes:

| Change | Kind | PR |
|--------|------|----|
| Delete the three CSVs in `shapiq/datasets/data` | files (the existing GitHub fallback takes over) | project PR |
| Cache that fallback's downloads in the user's cache, not the installed package | bug fix (one function) | project PR |
| Move the model-specific games out (delete the modules, drop them from `__init__` exports, point core tests at `shapiq_games`) | moves | project PR |
| Fix `MarginalImputer`'s null-player violation | bug fix | merged (#615) |
| Cast integer 0/1 coalitions to bool in `Game` | bug fix | merged (#614) |

Not core changes, handled elsewhere:
- **Order-0 convention.** `MoebiusConverter` puts 0 at `()` while `ExactComputer` puts the baseline value. The computers translate every result to the game's own convention, so core is left alone.
- **Unstable `game_id`.** It is based on Python's `hash()`. Setups get a stable `key` instead.
- **Inconsistent index declarations** (a `valid_indices` attribute here, a `Literal` alias there). Computers read whatever exists; harmonizing them in core is out of scope.

No test in this PR depends on either fix, so none is marked `xfail`.

## PR plan

Everything except the two core bug fixes lands as **one complete PR**, so the whole design can
be reviewed and ironed out in one sweep. It contains, in build order:

1. **Data layer.** The fetch-and-cache helper and dataset registry in `shapiq_benchmark` (with explicit task types), and removal of the bundled data files (games CSVs and JPEGs, core CSVs).
2. **Games.** Family classes, the move of the core games, deletions, `tests/shapiq_games` with contract tests.
3. **Benchmark.** Setups and the model registry, computers, `Benchmark`, metrics, runner, local cache, chain-of-trust and drift tests.

The two core bug fixes are separate PRs, done independently.

## Resolved questions

1. **Data vs. dataset valuation.** Both stay public under their literature names, backed by one
   private base class: players are disjoint groups of training rows, and v(S) is the test score
   of a model trained on the union of the groups in S. `DataValuation` uses one point per group
   (Data-Shapley-style defaults). `DatasetValuation` uses groups from a split strategy
   (uniform, increasing, random) or user-given groups such as data sources or owners.
   Mathematically, data valuation is the singleton-group special case of dataset valuation.
   The empty-coalition value stays an explicit, documented parameter, since a model trained on
   no data has no natural score.
2. **California torch network.** Dropped; the seeded generic `mlp` replaces it.
3. **Brute-force player cap.** Default 20 (needed for the TabPFN use cases), overridable per
   call. Tests stay at about 10 players or fewer so they remain fast.
4. **Games versus benchmark configuration.** A first version gave every game a `from_config`
   classmethod with a `config` and a `fingerprint`. That mixed benchmark plumbing (datasets,
   model names, cache keys) into the game zoo. The typed setups of `shapiq_benchmark` replace
   it: the games keep only their natural constructors, and every game except the synthetic
   ones stays benchmarkable through its setup. The datasets and the model registry moved to
   `shapiq_benchmark` with them.

## Verified with network access

The opt-in tests in `tests/shapiq_benchmark/test_heavy.py` (`SHAPIQ_RUN_HEAVY_TESTS=1`) download
the real data and models. All 25 passed on 2026-10-07:

- the 16 upstream tables against the previously bundled files (see Sources),
- a TabArena download, the UCI raw files (all pinned by SHA-256), and Imagenette,
- the vision transformer, ResNet-18, and DistilBERT sentiment games with their pretrained weights,
  and the TabPFN recontextualization and confounding games. The Hugging Face models load their
  default branch unless a `revision` is given.

Beyond the tests, all 51 TabArena datasets were downloaded and loaded: every one has finite
features, the declared task, and (for the 38 classification datasets) the number of classes
OpenML lists. `real_estate` (which needs `openpyxl`) loads its 414 rows, and the Sphinx docs
build without a warning, running all 40 gallery examples with the real models.

The vision transformer test caught a crash on reversed coalition arrays (torch rejects negative
strides); `tests/shapiq_games/test_image_games.py` now covers it without downloads.

## Core issues found while building (fixed separately)

The chain-of-trust tests and the self-review surfaced these core bugs. Each is left to a
separate core PR; the games guard against them in the meantime.

- **`class_index=0` on binary gradient boosting classifiers** (fixed on main, #618).
  `TreeExplainer` (path-dependent and interventional) explained the positive-class margin
  whatever class was requested, so class 0 silently returned the class-1 values, for
  scikit-learn GradientBoosting and HistGradientBoosting, XGBoost, LightGBM and CatBoost. The
  tree games rejected class 0 for these models until the fix; they now explain it as the
  negated class-1 margin, and `test_chain_of_trust.py` checks both classes of every binary
  booster against brute force.
- **Zero-cover nodes on the explained point's path.** The path-dependent quadrature TreeSHAP
  treats every zero-cover subtree as unreachable and drops it. The explained point's own path
  can still enter one (common in CatBoost's oblivious trees, through NaN routing or a value
  outside the training data). When a feature is absent below that node, its mass is lost, so
  a tree with a constant output gets nonzero Shapley values. In the reproducing CatBoost model,
  `PathDependentTreeComputer` differs from brute force on `PathDependentTreeGame` by 6.5e-4.
- **Gaussian processes with `normalize_y=True`.** The product-kernel conversion
  (`convert_gp_reg`) ignores the target scaling (`_y_train_mean`, `_y_train_std`), so
  `ProductKernelExplainer` explains a rescaled model: for a test GP the grand coalition is 0.141
  where `predict` gives 8.990. `ProductKernelGame` rejects such models until this is fixed. An
  RBF kernel with one length scale per feature also fails, with a numpy broadcasting error.
