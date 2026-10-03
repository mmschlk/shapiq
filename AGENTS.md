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

- Digits includes constant border pixels. Clustering scores are undefined for a
  singleton coalition whose column is constant on the actual clustering rows.
  Select nonconstant columns using those training rows and record their original
  IDs; random multi-feature probes alone do not expose this failure.
- TabPFN also rejects nonempty coalitions containing only constant columns.
  Its 64 training rows can have additional constant Digits pixels beyond those
  constant in the full dataset. Configurable TabPFN recipes select nonconstant
  columns on those exact rows and record the rule and original column IDs.
  Changing that selection defines a new game: do not reuse its old payoff chunks
  or replace undefined coalitions with the empty-coalition prediction.
- `ExactComputer.compute_fii` allocates a dense diagonal matrix with `2**d`
  rows and columns (8 TiB at 20 players). Benchmark tables above twelve players
  use qualified direct first/second discrete-derivative formulas for the six
  supported targets. A Möbius-transform shortcut amplified roundoff on parity
  games; retain tests for exactly zero SV and tiny genuine signals.
- Larger benchmark payoff tables use independently cached 4096-coalition chunks,
  each reconstructing the same seeded recipe. This deliberately freezes a
  batch-dependent realization; it is not independent Monte Carlo noise between
  chunks. Recorded per-coalition oracle costs are batch-amortized wall-time
  estimates, not individual timings or measured uncached estimator runtime.

- Gaussian and Gaussian-copula imputers reject categorical columns, including
  Bike Sharing's binary calendar features. Higher-dimensional recipes for these
  games use continuous Wine features and a classifier's class-one probability;
  do not bypass the categorical guard or regress on arbitrary class labels.

- Increasing neighbor-game input dimensionality while retaining a fixed radius can
  produce constant TNN games (observed with Wine's 13 features and radius 2).
  New configurable TNN recipes set the radius from the median nonzero pairwise
  distance of standardized training rows, with the rule recorded in metadata;
  legacy recipes retain radius 2. Never tune the radius against benchmark errors
  or choose held-out examples to force nonzero truth.

- Paired LeverageSHAP at a budget of `2 * n` has only `n - 1` independent
  interior directions for `n - 1` free coefficients. Large errors near this
  interpolation threshold can be statistical instability despite an accurate
  SVD solve; regularization changes the estimator and must be explicit.
  The reference repository added a low-budget `0.001 I` ridge safeguard in
  `f3c0427`, then removed it in August 2026 audit commit `04cc121`. Shapiq PR #583
  aligned sampling without restoring it. Check reference history before assuming
  minimum-norm least squares preserves the author's intended low-budget behavior.

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
- A small first-n training prefix can omit the held-out true class (Digits with
  16 players, seed zero, omitted class seven). New structured KNN recipes request
  stratified row selection explicitly. Keep legacy first-n recipes unchanged;
  never change the held-out label to make a constructor succeed.
- Hopper's login host has a different CPU model from its Slurm compute nodes.
  Record worker affinity and CPU model. `OverSubscribe=NO` alone does not prove
  an exclusive node: also verify that the job owns the node's full CPU count.
  Slurm logs that must be read from the login host need a shared workspace path;
  `/tmp` on a compute node is node-local.
- Hardware standardization does not require every campaign to use one host.
  `himem01`, `himem02` and `gpu15` were verified as 128-core AMD EPYC 9754 nodes
  with SMT disabled. Verify actual allocations and worker records when selecting
  another node; pinning every independent job to `himem02` needlessly serialized
  preparation, timeout retries and evaluation. Preserve each campaign's selected
  host and CPU affinities on resume.
- `RegressionMSR.valid_indices` is inherited from `ProxySHAP` and is broader than
  its constructor's actual SV/BV support. Benchmark capability catalogs must
  follow constructor checks, not inherited registries alone. `kADDSHAP` with
  `max_order=1` returns SV; its default order-two configuration is a different
  target/configuration and must not be silently relabeled as SV.

## Shared environment checks

- On this checkout, plain `uv run pre-commit run --all-files` can rebuild and
  reinstall the editable shapiq package when Git-derived version metadata changes.
  This changes package provenance for running jobs even if their source checkout
  is frozen. Use `UV_NO_SYNC=1 uv run pre-commit run --all-files` while any campaign
  uses the environment, and run checks sequentially with source-sensitive tests.

## Benchmark export gotchas

- Downloaded snapshots can retain an absolute production `duplicate_registry`
  path and baseline constructor options absent from the local checkout. Private
  candidates authenticate the historical snapshot but execute only their own
  factory and must not access that campaign registry. Built-in reruns still require
  compatible constructors and the configured registry; never rewrite an archived
  snapshot merely to bypass these checks.

- Zero-energy truth must be identified from the frozen game, even if every run is
  pending. Both Python summaries and browser filters exclude it for all methods;
  preset identity still includes the originally selected game IDs.
- A public reproduction ZIP is a separate export path from the website. Do not
  retain raw estimator exception messages in either: they can contain local paths.

- Sparse estimator interaction keys can contain `numpy.int64`; convert them to
  Python integers before serializing benchmark results. Set success only after
  serialization succeeds, and clear score fields on any output failure.
- The legacy ResNet image game uses fixed gray 127 masking (the callable-model
  path uses mean color). Its SLIC clipping reports one unused final player;
  benchmark preparation can remove that verified null player without changing
  the library game. Set inference batch size one to avoid float32 batch-dependent
  rounding when qualifying a frozen table.
- Full estimator extras increase worker import time. Timeout regression fixtures
  need enough startup allowance for a valid second worker (ten seconds here).

- Reading actively replaced benchmark checkpoints over Hopper's shared filesystem
  can transiently raise `ESTALE` (stale file handle). Retry monitoring reads;
  export the final report after workers have stopped writing.
- Fully ordered five-method Elo panels can reach numerical line-search roundoff
  before an overly tight gradient tolerance. Keep the convergence guard and the
  strict-order regression test when changing Bradley–Terry optimizer settings.
- SPEX's sparse-transform dependency samples through global NumPy/Python RNGs;
  estimator `random_state` alone does not seed those draws. Isolated benchmark
  cells must seed both globals before estimator construction. Older SPEX records
  without this protocol are not reproducible from their recorded seed alone.

- Browser weighted medians must use the same relative `1e-14` half-mass tolerance
  as Python. With 198 equal-weight cells split between zero and one, ordinary
  cumulative rounding otherwise misses the required midpoint of 0.5.
  Partial summaries renormalize successful weight; no successes means no score.

- The frozen `local_baseline_forest` explanation has eight declared players but
  only player zero affects the payoff; near-zero error at 2d is legitimate.
  Declared dimensionality alone does not establish game difficulty. Budget
  charts show the requested cap; actual query usage can be smaller.
- `InterventionalTreeSHAPIQ` and `ExactComputer` agree on nonempty SII/FBII
  coefficients but use different empty-coefficient conventions. For benchmark
  qualification, compare nonempty coefficients and check the actual oracle
  baseline separately; do not impose efficiency on SII/FBII coefficient sums.
- `ProductKernelExplainer` currently rejects a `ProductKernelModel` through its
  base explainer despite accepting that type in its annotation. Pass the fitted
  sklearn SVC/SVR, or use `ProductKernelComputer` with serialized kernel arrays.
- sklearn forests cast prediction inputs to float32, whereas exact tree routing
  can use their original float64 values. A breast-cancer seed exposed a 0.027
  Shapley discrepancy despite matching endpoint predictions. Round benchmark
  tree inputs through float32 before passing the same values to both paths;
  retain exhaustive small-game qualification across construction seeds.

- Specialized select padding (for example `.compactSelect select`) can override
  generic chevron clearance through CSS specificity. Verify computed right
  padding on every styled select; otherwise the arrow can overlap its text.

- Long-lived preparation workers can hit their address-space limit during exact
  qualification after successfully saving all payoffs. The min11 causal-local
  12-player failures were 128 MiB allocations for a 4096-by-4096 FII matrix,
  not failed oracle evaluations. Inspect saved NPZ files before recomputing a
  costly game; qualify them in a fresh process, authenticate the saved bytes,
  and preserve the original snapshots for the recovery audit.

- A shared editable `.venv` can import the main checkout even when a Slurm job
  starts in a frozen worktree. Min11 sweep 361009 therefore used the wrong
  LeverageSHAP/OddSHAP implementations. Set `PYTHONPATH="$PWD/src"` after changing
  directory, verify imported package paths, and compare actual run provenance
  with the prepared snapshot before starting. Per-cell consistency checks alone
  accept a consistently wrong source. Preserve and quarantine such results;
  never relabel them as measurements of the intended implementation.

- OddSHAP's default rejects budgets below `min(10, 2**n)` before any oracle
  calls. PR #560 deliberately screens singleton terms below `10*n`; this is
  different from the paper's low-budget tree-surrogate fallback. Do not diagnose
  these `ValueError`s as numerical failures or remove the guard without choosing
  and documenting the intended low-budget estimator behavior.

- OddSHAP can have a pre-existing efficiency residual with a nondefault support
  factor: eight players, `interaction_factor=3`, seed zero, budgets 8/9, and
  `7 + sum((i+1)*z_i) + 2*z_0*z_1*z_2`. Independent guard-removal auditing
  reproduced identical old/new outputs; investigate that regression numerical
  issue separately rather than attributing it to newly accepted tiny budgets.

- Expanded benchmark datasets must use their shipped loader outputs, not docstring
  dimensions: Ionosphere drops a constant column and returns 33 features, despite
  its 34-feature example. Mushroom retains missing cells and a constant column;
  NHANES retains missing features and signed, censored survival labels. Benchmark
  median repair uses training rows only; NHANES regression is explicitly a surrogate.
  Shapiq Wine Quality is regression with 12 columns including binary `type_white`,
  unlike sklearn Wine classification. Never silently rebind historical `wine` IDs.
- TabArena task types come from OpenML task metadata, not numeric target dtype.
  Shipped loaders may encode/impute on the full dataset and cache CSVs beside their
  module. First uncached loads need optional `openml` and `pyarrow`; do not install
  into a shared environment while frozen jobs are running. Categorical singleton
  KMeans games can have nearly zero within-cluster variance and huge Calinski–Harabasz
  scores; new recipes select noncategorical columns with more than three values.
  TabArena cache writes are nonatomic and initial OpenML arrays can round differently
  from CSV reloads. Warm caches before parallel workers and freeze the reloaded CSV
  representation, not the first-download return value.

- Model-cache identities need ordered feature names, array shapes/dtypes and loader
  source as well as numeric bytes. Renaming a feature without changing values
  must not reuse stale reproduction metadata.
- New website scripts must be added to both report.py's export assets and the
  explicit GitHub Pages workflow copy list; local previews alone do not catch a
  missing deployed script.
- CUDA preparation must not inherit the CPU worker's 12-GiB address-space cap:
  CUDA reserves much larger virtual ranges. Use Slurm memory limits, one worker
  per GPU, synchronized timing and explicit backend/precision in cache identities.

- Profiled XGBoost predictors honor early stopping, but tree converters can retain
  every stored boosting round. Trim a copied booster to the selected rounds before
  constructing a tree game; boosted classification tree games use explicit margins.
- ThresholdNNExplainer can put the correct empty coefficient in its result while
  leaving baseline_value at zero. Structured benchmark TNN copies that unchanged
  empty coefficient into baseline metadata and qualifies it against enumeration.

- Fixed TabPFN float32 predictions vary slightly with batch shape (measured
  probability differences about 5e-7 on L40S). Benchmark preparation records an
  explicit float32 qualification tolerance and batch/repeat/reverse/singleton
  discrepancies; exact scores refer to the saved canonical table. This tolerance
  is a qualification policy, not a certified global floating-point error bound.
- The path-dependent Python tree game can retain float32 XGBoost leaf values
  and sample weights while its exact solver computes in float64. Freeze both as
  float64 for the same oracle/reference; do not hide mismatches by relaxing exact
  qualification, especially for coefficients that should be exactly zero.

- Slurm `sacct --array` can still omit individual pending tasks before their
  accounting records exist. Completion watchers combine exact task IDs from
  `squeue --array` with terminal accounting; absence from either is not success.
- A healthy watcher process does not prove wake-up delivery. Exercise the actual
  message route: a full-rollout test found corrupt local Codex state/queue
  databases despite healthy Slurm jobs. Retain delivery errors and retry without
  a success receipt; do not delete session databases or restart other sessions
  as part of routine benchmark monitoring.

- Free space from `df` does not establish remaining home-directory quota. Full
  rollout writes failed with `EDQUOT` despite filesystem free space. Put large
  campaign caches on lab storage; stop writers and verify every copied file hash
  before relocating. An open NFS lock can leave a temporary `.nfs` file until its
  holder closes it, even after all other campaign files have moved.
- Inspect exact Slurm task reasons after holding arrays or changing dependencies.
  Four phase-three tasks became `JobHeldAdmin` during quota recovery; owner-level
  release was denied. The reason alone does not identify who caused the hold.
  An administrator must release it; do not bypass it by resubmitting those tasks.
- `scontrol show job $SLURM_JOB_ID -o` can return multiple array-task records when
  the running task owns the array's numeric root ID. Flattening their fields into
  one dictionary can substitute a pending sibling's allocation (observed for
  `361642_6`). Select `${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}` for arrays and
  require exactly one returned record before checking exclusive-node hardware.
- Raw benchmark shards repeat the full snapshot metadata: the first expanded
  batch had 128 files of roughly 48 MB each. Read shards one at a time during
  audits and exports; eagerly parsing the whole batch multiplies memory use
  without adding information. Keep full duplicate/provenance validation.
- `src/shapiq_benchmark/metrics.py` is an existing legacy metrics API. New
  per-order benchmark scoring lives in `order_metrics.py`; do not replace the
  legacy module when adding scoring helpers. Check that a proposed new path is
  unused before applying an Add File patch, which can overwrite an existing file.

- Compact benchmark checkpoints store a small `results.json` manifest plus a
  `records.jsonl` journal and shared snapshot. Use
  `shapiq_benchmark.results_io.read_results`, not direct JSON loading, when
  consuming either legacy or compact results. Strict exports authenticate all
  companions; interrupted-tail recovery is reserved for the locked runner.
- Qualifying an eight-player structured oracle proves small-game consistency,
  not native-dimensional solver feasibility. Time and memory gate the actual
  requested player count, target, truth serialization and every construction seed
  before launching larger structured games.
- Moving the benchmark environment to lab BeeGFS can make 128 simultaneous
  Python imports stall in filesystem metadata calls before any result checkpoint
  exists (observed on himem02, job 362344_0). Empty logs alone do not prove a
  runner deadlock: inspect process state and blocked paths. Any node-local import
  staging must preserve package bytes and full provenance; never install or
  replace dependencies underneath running workers.
- The shipped Wine Quality loader reads its two remote CSV URLs on every call;
  it has no persistent local dataset cache. HTTP gateway failures are operational
  preparation failures, not evidence that a recipe failed scientific qualification.
  Retry only affected recipes in a separately authenticated supplement; preserve
  the original preparation decision and snapshots.
- The benchmark branch tracks `fork/benchmark` (rtealwitter/shapiq); `origin`
  points to mmschlk/shapiq. Check the tracking remote before pushing benchmark
  work so updates reach the existing PR rather than creating an upstream branch.
- Shipped `ExactComputer` FSII uses finite endpoint weight `big_M=1e8`.
  Exhaustive payoffs therefore still yield a small numerical reference floor
  (observed about `2e-16` normalized squared coefficient error) and efficiency
  residual. Independent derivative formulas and endpoint-weight diagnostics can
  distinguish that floor from estimator error; never rewrite frozen snapshots.
- Hopper lab BeeGFS advisory locks are host-local: an isolated probe confirmed
  that an exclusive `flock` on himem02 blocks another himem02 process, but the
  login host can simultaneously acquire the same lock/inode. Do not infer that
  remote writers stopped from a successful login-host lock. Publication requires
  terminal scheduler state, complete manifests, and stable authenticated hashes.
  Shared duplicate registries likewise need same-host writers or a separately
  qualified distributed locking mechanism before multi-host evaluation.
- Historical exports can contain constructor options absent from the exporting
  checkout (for example OddSHAP `ridge` from a frozen experimental branch).
  Authenticate the recorded source and parameter schema without requiring the
  current constructor to accept them. Execution must retain strict local
  constructor validation; never strip recorded options to make an export pass.

- CPU preparation launches 16 workers; reserving all 128 node cores needlessly
  blocks it. Estimator sweeps currently use diagnostic timing, so a full-node
  launch guard does not make their timings isolated. Use explicit shared CPU
  worker counts and authenticate operational overrides separately from frozen
  scientific inputs. A GPU-node hostname does not imply GPU allocation: inspect
  Slurm GPU GRES. Preserve completed shards when changing hardware; their resume
  identities include hostname and affinity and must not be rewritten.

- Slurm CUDA-visible numeric ordinals can differ from host `nvidia-smi` indices.
  Resolve visible devices through the CUDA driver to UUIDs before monitoring.
  Utilization is whole-device occupancy, not proof of speedup. Place the GPU
  guard inside `srun`: qualification can launch new process sessions, so Slurm
  job cleanup must also stop descendants outside the guard's process group.
