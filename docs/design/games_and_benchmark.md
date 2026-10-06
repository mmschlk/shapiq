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
| 3 | All datasets stay usable (31 classic + 51 TabArena). |
| 4 | No hosting and no website: everything runs and caches locally, rudimentary on purpose. |
| 5 | No data files ship in any wheel: no CSVs, images, weights or precomputed game values. This includes the three CSVs in core `shapiq/datasets/data`. |
| 6 | Breaking changes in both packages are fine. PR #602 (Teal) integrates on top of this afterwards. |
| 7 | Every game that nothing in core uses moves into `shapiq_games` (see "Games moving out of core"). |
| 8 | Important core bugs found during the audit (below) are fixed in separate PRs after this work. |
| 9 | **Core stays (almost) untouched.** The only core changes are the ones listed in "Core changes"; everything else is built in `shapiq_games` and `shapiq_benchmark`. |

## Dependency direction

```
shapiq_benchmark  ──►  shapiq_games  ──►  shapiq (core)
  computers,             games, datasets,      algorithms: ExactComputer, MoebiusConverter,
  Benchmark, metrics,    model zoo             TreeSHAPIQ / TreeExplainer, InterventionalTreeSHAPIQ,
  runner, local cache                          KNNExplainer, ProductKernelComputer, imputers
```

Nothing ever points the other way. Games know nothing about computers, and core knows nothing
about either package (core *tests* may import games, as they already do for `DummyGame`).

## `shapiq_games`

### Layout

```
shapiq_games/
  _base.py              the game contract (bool coalitions, explicit x, class index, fingerprint)
  _setup.py             from_config plumbing (load, split, fit)
  _training.py          clone-per-coalition training, metrics, constant predictors
  synthetic/            DummyGame, UnanimityGame, SOUM, RandomTableGame (replaces RandomGame)
  tree/                 PathDependentTreeGame (was TreeSHAPIQXAI), InterventionalTreeGame (moved from core)
  nn/                   KNNGame, WeightedKNNGame, ThresholdNNGame (moved from core)
  kernel/               ProductKernelGame (moved from core)
  local_xai.py          LocalExplanation (marginal / conditional / baseline imputers, TabPFN)
  global_xai.py         GlobalExplanation (SAGE-like)
  feature_selection.py  FeatureSelection
  valuation.py          DataValuation, DatasetValuation (one shared implementation)
  ensemble_selection.py EnsembleSelection, RandomForestEnsembleSelection
  uncertainty.py        UncertaintyExplanation
  clustering.py         ClusterExplanation
  unsupervised.py       UnsupervisedData
  causal.py             GlobalConfoundingXAI, LocalConfoundingXAI
  vision/               ImageClassifier (ViT, ResNet, custom classifiers)   [torch, transformers]
  language.py           SentimentAnalysis                                  [transformers]
  datasets/             dataset registry, loaders, local cache, example images
  models.py             model registry + tuned presets
```

There is **one class per game family**, configured by arguments. The 153 classes today
(82 near-identical local-XAI dataset classes, 13 interventional ones, 18 valuation/selection
subclasses, …) collapse to roughly 25. The `benchmark` sub-package name goes away: these are
games, and benchmarking is `shapiq_benchmark`'s job.

### Construction

Each family has two entry points:

```python
# 1. explicit objects: the primary, fully general constructor
game = FeatureSelection(estimator, x_train, y_train, x_test, y_test, task="regression")

# 2. string configuration: resolves dataset and model through the registries
game = FeatureSelection.from_config(dataset="california_housing", model="random_forest", random_state=0)
```

Games built with `from_config` carry a JSON-serializable `config` and a stable `fingerprint`
(a SHA-256 hash of class name + config). Today's `game_id` relies on Python's `hash()`, which
changes between processes, so it cannot be used as a cache key.

### Game contract (enforced by `tests/shapiq_games`)

1. Subclasses `shapiq.Game` and implements `value_function`.
2. **Deterministic.** v(S) is a pure function of the constructor arguments, including `random_state` (the name core shapiq uses). It must not depend on call order, batch composition, repetition or process. Randomness is drawn once at construction from `np.random.default_rng(random_state)`. No global or stateful RNG is used during evaluation (the conditional imputer of core, which has one, is reseeded before every evaluation).
3. **Explicit explained point.** Local games take `x` as an index into the explanation data or as an array. The default is index `0`, never a random point. One name, `x`, everywhere (today it is `x`, `x_explain`, `target_instance`, `explain_point` and `x_explain_path`).
4. **Explicit class.** Classification games resolve `class_index` at construction and store it as an int. Following the shapiq explainers, `None` means class `1` (previously class 1, class 0 or argmax depending on the game). Image games default to the class predicted on the image.
5. **Meaningful empty coalition.** v(∅) comes from the game's semantics, for example a model with no features predicting the training mean or majority class. It is never a hard-coded 0, so `normalize=True` always does what it says.
6. **Declared output space.** Each game documents what its values are (probability, margin or log-odds, loss, score) and in which direction a higher value is better.
7. **Typed public attributes** that a computer needs, for example:
   - tree games: `model`, `x`, `class_index`
   - interventional trees: `reference_data`
   - SOUM: its Möbius coefficients
   - KNN games: `model`, `x`, `class_index`
   - product kernel: `model`, `x`
8. **Import hygiene.** No import-time warnings. Optional dependencies (torch, transformers, tabpfn, openml, xgboost, lightgbm, catboost) are imported lazily, with an error that names the missing package.
9. **Pinned external models.** Pretrained models are pinned to an exact revision (Hugging Face `revision=`, a torchvision weights enum).

Each family gets contract tests on a small offline configuration:
- the same coalition repeated gives the same value
- a permuted batch gives a permuted output
- two instances with the same arguments are equal
- `n_players` and the normalization are right
- axioms where they hold (efficiency of the computed values; null and dummy players for games that have them)

### Datasets

- **One registry**, name → `DatasetSpec` with these fields:
  - source
  - `task` ("classification" or "regression"), declared explicitly rather than guessed from the labels
  - preprocessing
  - feature names
- **Sources, in order of preference:**
  1. The original upstream: OpenML ID and version, or an sklearn fetcher.
  2. A raw GitHub URL pinned to a commit of this repo where the file already sits in history. This needs no new hosting.
  3. Seeded generators for synthetic data.

  Every downloaded file is checked against a SHA-256 hash.
- **Local cache.** Files go to `$SHAPIQ_DATA_DIR`, defaulting to `$XDG_CACHE_HOME/shapiq` (or `~/.cache/shapiq`), and are written atomically (temp file + rename) so parallel test workers are safe. Nothing is ever written into the installed package.
- **Deterministic splits.** Seeded train/test splits, stratified for classification.
- **Data removed from the tree** (the files remain reachable through git history and the pinned URLs):
  - `shapiq_games/datasets/data` (81 MB)
  - `shapiq/datasets/data` (8.5 MB)
  - the ImageNet example JPEGs
- **Core loaders stay unchanged.** Core's three public loaders (`load_california_housing` & co.) already fall back to downloading from `main/data/` on GitHub when their CSV is missing, so deleting the CSVs needs no core code change. The repo-root `data/` folder therefore stays; it is not part of any wheel. The fetch-and-cache helper lives in `shapiq_games` only.
- **Undeclared dependencies.** `openml`, `ucimlrepo` and `openpyxl` get declared as optional dependencies; today they aren't declared at all.

### Models

- `build_model(name, task, random_state=..., preset=..., **params)` covers decision trees, random forests, XGBoost, LightGBM, CatBoost, MLPs, linear models, SVMs, Gaussian processes, nearest neighbors, and TabPFN. `random_state` is always passed through, and models run single-threaded where the library allows it.
- Hyperparameter presets (previously the Optuna JSONs in `shapiq_benchmark`) are Python dicts next to the registry (`preset="tuned"`). The Optuna script stays a benchmark tool and now works for any registered dataset.
- The California torch network (it currently loads weights from a `tests/` path and silently falls back to random weights) is dropped in favor of the seeded generic `mlp` model.

### Games moving out of core

No method in `src/shapiq` uses them. Only `__init__` re-exports and tests reference them.

| Today | Moves to |
|-------|----------|
| `shapiq.tree.InterventionalGame` | `shapiq_games.tree.InterventionalTreeGame` |
| `shapiq.explainer.nn.games.{KNN,WeightedKNN,TNN}ExplainerGame` | `shapiq_games.nn` |
| `shapiq.explainer.product_kernel.game.ProductKernelGame` | `shapiq_games.kernel` |

- `WeightedKNNExplainerGame` instantiated a `WeightedKNNExplainer` to call its private weight-discretization helpers. Their round trip is just rounding to multiples of `2**-n_bits`, so the moved game implements that rounding itself: no core change, and the ground-truth game no longer depends on the explainer it checks.
- Imputers (`MarginalImputer`, `TabPFNImputer`, …) are also `Game`s but are used by core explainers, so they stay in core. `LocalExplanation` wraps them.

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

### Computers

```python
class Computer(Protocol):
    game: Game
    def supports(self, index: str, order: int) -> bool: ...
    def exact_values(self, index: str, order: int) -> InteractionValues: ...
```

- **Location.** Computers live here as thin adapters around core algorithms. They never re-implement the algorithms, and each one binds to the game types it understands.
- **Index support comes from core, not a parallel list.** `supports(index, order)` reads the declarations core already has for the wrapped algorithm:
  - `ExactComputer.valid_indices`
  - `ValidMoebiusConverterIndices`
  - `TreeSHAPIQIndices`, `InterventionalTreeSHAPIQIndices`, `QuadratureTreeSHAPIndices`
  - `ValidNNExplainerIndices` and `ValidProductKernelExplainerIndices` (with their order-1 checks)

  Constraints core does not declare (for example a maximum order) are added inside the computer, not in core. A drift test calls each core algorithm for every index and order: everything `supports` accepts must run, and everything it rejects must be rejected by core too.
- **Never a silent fallback.** If `supports(index, order)` is false, `exact_values` raises an error instead of computing something else. Today `PathdependentComputer` returns order-1 SV when asked for order-2 k-SII.
- **One output convention.** A computer returns the values of the game *as `game(...)` evaluates it* (same output space, same class): the interactions of order 1 to `order`, with the game's v(∅) as `baseline_value` (0 when normalized). The order-0 term is excluded: core algorithms disagree on it, and no metric uses it.

| Computer | Games | Core algorithm |
|----------|-------|----------------|
| `BruteForceComputer` | any game, with a player cap (default 20, overridable) | `ExactComputer` |
| `MoebiusComputer` | Dummy, Unanimity, SOUM | `MoebiusConverter` |
| `PathDependentTreeComputer` | `PathDependentTreeGame` | `TreeSHAPIQ` / `TreeExplainer` |
| `InterventionalTreeComputer` | `InterventionalTreeGame` | `InterventionalTreeSHAPIQ` |
| `KNNComputer` | KNN, weighted KNN, threshold NN games | `KNNExplainer` & co. |
| `ProductKernelComputer` | `ProductKernelGame` | `ProductKernelComputer` |

`default_computer(game)` picks from a small registry: the structured computer for the game's
type if there is one, otherwise `BruteForceComputer` (and raises an error above the player cap).

### Benchmark

```python
benchmark = Benchmark(game)                       # computer = default_computer(game)
benchmark = Benchmark(game, computer=BruteForceComputer(game))
gt = benchmark.exact_values(index="k-SII", order=2)   # cached locally by game fingerprint
results = run(benchmark, approximators, budgets, index="k-SII", order=2, seeds=[0, 1])
```

- `run` evaluates approximators × budgets × seeds, scores them with the metrics and returns a tidy table, which it can also write to a local CSV/JSON file. Which approximator runs for which index comes from the approximator's existing `valid_indices`; unsupported combinations are skipped and recorded as such, not dropped silently.
- Exact values are cached under `$SHAPIQ_DATA_DIR/ground_truth/<fingerprint>/<environment>/<computer>_<index>_<order>.json` using `InteractionValues.to_json_file`, where `<environment>` hashes the installed versions of shapiq and the model libraries, so an upgrade never reuses ground truth computed with other versions. Games built from objects (no fingerprint) are not cached.
- `LocalXAIBench`, `PathdependentBench`, `InterventionalBench`, `TabPFNBench`, `ImageBench`, `bench_types.py` and `setup.py` go away. String configuration lives in the games' `from_config`.

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
- **Ranking:** Kendall τ and Spearman ρ are computed on the values themselves. Today Kendall τ correlates `argsort` positions: a true τ of 0.97 scores 0.55.
- **Top-k:** Precision@k uses top-k by |ground truth|. Today `KendallTau@k` uses the k *smallest* values, and `Spearman@k` ignores k.
- **Faithfulness:** R² of the reconstructed game on seeded coalition samples, for every index rather than FBII only.

## Core changes

Core stays as it is except for exactly these changes:

| Change | Kind | PR |
|--------|------|----|
| Delete the three CSVs in `shapiq/datasets/data` | files only, no code (the existing GitHub fallback takes over) | project PR |
| Move the model-specific games out (delete the modules, drop them from `__init__` exports, point core tests at `shapiq_games`) | moves | project PR |
| Fix `MarginalImputer`'s null-player violation | bug fix | separate |
| Cast integer 0/1 coalitions to bool in `Game` | bug fix | separate |

Not core changes, handled elsewhere:
- **Order-0 convention.** `MoebiusConverter` puts 0 at `()` while `ExactComputer` puts the baseline value. The computers translate every result to the game's own convention, so core is left alone.
- **Unstable `game_id`.** It is based on Python's `hash()`. Games get their own `fingerprint` instead.
- **Inconsistent index declarations** (a `valid_indices` attribute here, a `Literal` alias there). Computers read whatever exists; harmonizing them in core is out of scope.

Contract tests that depend on the two bug fixes are marked `xfail` with a reference to the PR that will fix them.

## PR plan

Everything except the two core bug fixes lands as **one complete PR**, so the whole design can
be reviewed and ironed out in one sweep. It contains, in build order:

1. **Data layer.** The fetch-and-cache helper and dataset registry in `shapiq_games` (with explicit task types), and removal of the bundled data files (games CSVs and JPEGs, core CSVs).
2. **Games.** Family classes, model registry, the move of the core games, deletions, `tests/shapiq_games` with contract tests.
3. **Benchmark.** Computers, `Benchmark`, metrics, runner, local cache, chain-of-trust and drift tests.

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

## Not verified in the build environment

The build environment could not reach OpenML, the UCI repository, Hugging Face, or
download.pytorch.org. The following were therefore implemented but not run end to end; the
opt-in tests in `tests/shapiq_games/test_heavy_games.py` (`SHAPIQ_RUN_HEAVY_TESTS=1`) cover them:

- downloading the 51 TabArena datasets (their OpenML ids were checked against TabArena's metadata),
- downloading `wine_quality`, `real_estate`, and `forest_fires` from UCI (not yet pinned by
  checksum),
- the vision transformer, ResNet-18, and DistilBERT sentiment games, and the TabPFN games. The
  Hugging Face models are not yet pinned to a revision.

All other datasets were checked to reproduce the previous loaders exactly.
