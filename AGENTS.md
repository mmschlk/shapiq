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

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

### Run Pre-commit (takes only 3s)

During active benchmark campaigns, set `UV_NO_SYNC=1` before this command; see
"Shared environment checks" below.

```bash
uv run pre-commit run --all-files
```

## Benchmark integration notes

- Table preparation pilots do not serialize `n_players`; audit their actual
  coalition counts and cost-projection fields. Native pilots do record it.
  Use real serialized pilot fixtures when testing campaign audits.
- Proportional stratification can omit rare classes from tiny training-player
  samples. Binary weighted KNN then lacks an opposing class, while data valuation
  can reject incomplete class coverage. Authenticate and replay the original
  selection before classifying these as structural exclusions; do not silently
  choose different rows or accept arbitrary constructor exceptions.

- Digits includes constant border pixels. Clustering scores are undefined for a
  singleton coalition whose column is constant on the actual clustering rows.
  Select nonconstant columns using those training rows and record their original
  IDs; random multi-feature probes alone do not expose this failure.
- Legacy clustering filters only applied to Digits or datasets declaring
  `categorical_features`; Bike Sharing lacked that catalog field. Its binary
  singleton coalitions produced Calinski-Harabasz scores near 1e35 and misleading
  high-order Fourier mass. Use the explicit `cluster_continuous_v1` recipe for
  universal training-row cardinality filtering. Publication checks label CH
  values above `(N - 2) / float64_eps` as numerical controls, preserving frozen
  payoffs and recording the additional quality policy. Requested cluster count
  is not actual cluster count for binary coalitions; do not assume they match.
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

- Gaussian model profiles can fail with "Too few continuous fitting-row features"
  even when the dataset's total column count is sufficient. Their rule excludes
  catalog categorical names and requires more than TWO distinct finite values
  on the exact seeded, bounded fitting rows (different from clustering's >3 rule).
  Audit these exclusions by replaying the split and feature counts from pinned
  cached data; never accept arbitrary constructor ValueErrors as known limits.

- The continuous-feature rule above belongs to Gaussian/copula imputation games,
  not automatically to a Gaussian-process predictor. `local_baseline` with the
  `gaussian_process` model profile uses the ordinary feature rule. Check the
  construction's `feature_rule` before excluding binary or categorical inputs.

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
- Matrix preparation uses batch-local model caches and does not write the shared
  duplicate registry. Requiring the previous wave's recovery or audit to finish
  before preparing the next wave needlessly serializes independent work. Keep
  evaluation dependent on its own qualified inputs and preserve evaluation order
  for duplicate ownership. Budget concurrent preparation, evaluation, recovery
  and audits together; lowering an array throttle does not stop existing tasks.
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

- `ExactComputer` k-SII aggregation omits exact-zero coordinates, while the
  direct table formula serializes them explicitly. Compare their sparse vectors
  with missing coordinates interpreted as zero; do not require identical key
  sets or rewrite frozen references. Nonzero missing coefficients still need to
  pass the numerical comparison.

- Snapshot IDs use `runner.identity` with default JSON spacing, while recovery
  attempt software hashes use compact `recovery.row_hash` JSON. These hashes
  differ for the same object; reuse the matching encoding when auditing each
  field rather than treating all canonical JSON hashes as interchangeable.

- Splitting report files does not avoid GitHub Pages' 1-GB published-site limit.
  Release assets work as direct downloads but are not automatically browser-fetch
  storage: public HEAD and ranged GET checks on 2026-10-05 returned no CORS
  permission for this site's origin. Verify payload size and delivery before
  moving lazy-loaded data from same-origin Pages URLs to release URLs.

- Reproduction ZIPs sanitize and reconstruct result manifests. Hashing those
  manifests with `merge_results` does not recover the original published run IDs.
  Historical matrix reuse must preserve authenticated public row/run mappings;
  keep the ZIPs as reproduction evidence rather than silently replacing run IDs.

- SQLite cannot drop a temporary selection table while another result iterator
  is active. The disposable record store uses one indexed selection table with
  per-query IDs and savepoints; retain the interleaved-iterator regression test.

- `metadata.cluster_id` is a weighting/bootstrap group, not a unique game
  instance: phase six has 188 instances but 178 such groups. Count authenticated
  construction/artifact identities across all targets; an SV-only count misses
  phase seven's sixteen interaction-only instances.

- Historical release audits can reference mutable checkout UI files as well as
  immutable exported copies. Later UI edits require a separate revalidation
  receipt mapping only those references to archived bytes that match both the
  original audit and its Git revision. Preserve the original audit and verify
  every other input unchanged; never skip hash checks or rewrite old evidence.

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
  Sum successful weight with compensation: 1,377 cells accumulated enough error
  in a naive JavaScript sum to move an exact midpoint outside that tolerance.

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

- Legacy `TreeSHAPIQXAI` traverses `goes_left` without `TreeModel.cast_input`,
  while the quadrature exact solver honors XGBoost's float32 input precision.
  A float64 point just below a float32 split can therefore take opposite branches.
  Freeze XGBoost benchmark points through float32 for both paths, like sklearn
  forests; preserve strict endpoint and exhaustive coefficient checks. Core384
  XGBoost pilots exposed this with efficiency gaps up to 30.6, not roundoff.

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
  representation, not the first-download return value. A clean Git worktree omits
  ignored loader CSV caches; seed authenticated cache bytes into a new frozen
  worktree before launching when its fixed environment lacks download extras.

- Model-cache identities need ordered feature names, array shapes/dtypes and loader
  source as well as numeric bytes. Renaming a feature without changing values
  must not reuse stale reproduction metadata.
- New website assets belong in `benchmark/site/assets.json`, the shared explicit
  list used by local report exports and GitHub Pages. Local previews alone do
  not catch a missing deployed script; keep the manifest complete.
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
- NumPy 2 scalar promotion keeps `64 * np.finfo(np.float32).eps * scale`
  as `numpy.float32`, which standard JSON cannot encode. Cast the already
  computed `oracle_validation.absolute_tolerance` to Python `float` when recording
  metadata; preserve the original tolerance computation and qualification probes.
  Exercise snapshot serialization, not only payoff preparation, in regression tests.
- The path-dependent Python tree game can retain float32 XGBoost leaf values
  and sample weights while its exact solver computes in float64. Freeze both as
  float64 for the same oracle/reference; do not hide mismatches by relaxing exact
  qualification, especially for coefficients that should be exactly zero.

- Slurm `sacct --array` can still omit individual pending tasks before their
  accounting records exist. Completion watchers combine exact task IDs from
  `squeue --array` with terminal accounting; absence from either is not success.
  A finished root can also disappear from Slurm's live job cache, making
  `squeue -j ROOT` fail with `Invalid job id specified`. Query the user's live
  queue and filter exact IDs when checking for remaining allocations; establish
  terminal completion separately through full task accounting.
- Live accounting counts can transiently disagree with the planned task count
  during transitions (343 reported rows for a 342-task preparation wave).
  Verify exact task IDs and uniqueness, and retry inconsistent reads before
  updating progress or claiming completion.
- Register array tasks, not redundant array-root IDs, in the watcher's `jobs`
  list. Expanded accounting may never emit a separate root record. Registering
  both blocked all checks for retry arrays 365191/365192 despite complete task
  coverage. Keep root IDs and membership in campaign metadata; verify that
  `job_states` resolves every registered ID after changing the tracking list.
- Benchmark record sequence IDs can retain SQLite IDs from an earlier import
  (the Phase7 profile starts at 235225). Compare their exact values and uniqueness
  with the frozen input; do not assume they start at zero or renumber them.
- Updating an active array's throttle can return a nonzero status for an already
  finished sibling while successfully updating pending and running tasks. Read
  back each active entry's `ArrayTaskThrottle` before retrying or assuming that
  the mutation failed (observed for array 364505).
- SQLite can favor the record-order index for game-filtered benchmark exports,
  scanning every stored row for each family/method. `RecordStore.select_indexed`
  selects through the existing unique cell index for these filters and sorts
  only matching rows. Retain its query-plan regression when changing selectors.
- Browser verification pages must declare UTF-8 in HTML or HTTP headers. Classic
  scripts inherit the document encoding while worker scripts use UTF-8; a test
  page without a charset decoded the target separator `·` as `Â·`, selecting no
  games in the main thread while the worker selected the correct panel. The
  production page already declares UTF-8; fix the fixture, not the scores.
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

- `ExactComputer` uses a finite `1e8` endpoint weight for FSII. Independent
  analytic references can therefore differ slightly below thirteen players,
  where the benchmark still uses this solver. Phase-four auditing found a
  maximum whole-vector discrepancy of 4.81e-17 nMSE. Record this numerical floor
  and check per-order signal eligibility; do not relabel it as estimator error
  or silently rewrite frozen truth while evaluations are running.

- Offline special-model jobs require warming the exact shipped checkpoint IDs.
  ViT uses `google/vit-base-patch32-384`, not another ViT resolution. Prewarm
  its processor, config and weights in the inherited lab `HF_HOME`, record the
  revision and hashes, and verify both loaders offline before releasing jobs.
  A missing download is an operational issue, not an unsupported game.

- Regression normal equations can return enormous finite coefficients without
  raising on a rank-deficient design. The exception-only fallback in
  `solve_regression` does not detect this: a 16-player game at budget 16 returned
  KernelSHAP nMSE 6.63e21 with design rank 14; the existing direct weighted SVD
  returned 224.93 on identical samples. Audit coefficient magnitudes/rank as well
  as score arithmetic. Do not clip scores or silently rewrite frozen results.

- Slurm can reject `afterok` for a completed job that has expired from controller
  memory while `sacct` still reports completion. Remove only the expired dependency
  after verifying successful accounting and its final artifact audit; retain the
  audit gate in the dependent job.

- Bradley–Terry L-BFGS can stop on objective tolerance before Elo ratings are
  accurate, especially with weak overlap. A small maximum gradient alone is not
  a rating-error guarantee. Use the L2 strong-convexity bound on the gradient norm
  and refine as needed; the phase-four audit found errors of about 0.031 Elo points.

- A CUDA-backed model does not imply sustained GPU utilization during preparation.
  Phase-five TabPFN pilots spent most time constructing models and processing small
  coalition batches; the unchanged utilization guard stopped them at 2.44% average.
  CPU recovery needs new backend/cache identities and an authenticated replacement
  for the failed batch. Do not reuse CUDA payoffs or invent a scientific exclusion
  merely to make the original campaign export complete.

- The website's real/synthetic panel and its control toggle are independent filters.
  Enabling controls in the real panel does not include synthetic diagnostics.
  Phase five has 90 real core, 34 real control and 24 synthetic control instances;
  browser audits must check both panels instead of expecting all 148 in one view.

- A stratified training pool does not guarantee that its first 64 rows contain
  both classes. The frozen TabPFN APS Failure seed-one recipe has only class zero
  in that prefix, so selecting probability column one raises an IndexError.
  Audit this exclusion using the exact dataset, split, fitting rows and traceback;
  do not accept unrelated IndexErrors or change the rows under the same game ID.

- Even a stratified small sample can omit a rare class: all four frozen Taiwanese
  Bankruptcy Weighted KNN instances select only class zero in their 11 training
  rows. The constructor requires at least two classes. Authenticate the exact
  selected rows and constructor error before accepting this exclusion; a larger
  or differently sampled training set defines a new game.

- Mixing node requirements within one Slurm array can leave suitable CPUs idle.
  Slurm skips remaining array elements after an unrunnable element; a task waiting
  for memory on one host can therefore delay tasks targeting another host. Inspect
  per-task node requirements and scheduler state before assuming a CPU shortage.
  Prefer separate arrays for distinct host requirements in new campaign plans.

- On Hopper, `srun` can remove an explicitly empty `CUDA_VISIBLE_DEVICES` from
  a CPU-only job's child environment. Probe365477 verified this while CPU affinity,
  hardware, thread limits and source provenance all passed. Launch strict CPU
  recovery/audit workers with `srun ... /usr/bin/env CUDA_VISIBLE_DEVICES= python ...`
  so the child retains the required setting; keep the no-GPU runtime checks.

- Public game metadata has two whitelists: `report.METADATA_FIELDS` and
  `partitioned.FILTER_METADATA`. Scoring identity such as `focused_design` must
  survive both. Test actual exported filter partitions against browser scoring;
  full-detail rows or direct helper tests alone do not cover live rankings.

- Slurm's GPU-node step setup can remove an exported empty `CUDA_VISIBLE_DEVICES`
  even from a CPU-only job (observed on gpu15). Set it inside `srun` with
  `env CUDA_VISIBLE_DEVICES=`; verify zero allocated GPU TRES separately. The
  focused preflight correctly rejects an unset variable before constructing games.

- Preserve the invocation path of a virtualenv interpreter when serializing job
  commands. Resolving `venv/bin/python` symlinks can select the bare interpreter
  and lose its site-packages. Use `absolute()`, and authenticate its environment
  separately; test delegation as well as the parent interpreter.

- Batching tree predictions can change NumPy's ensemble reduction order even
  with the same `np.sum(..., axis=0)`: a singleton background makes that axis
  contiguous, while flattened coalition batches make it strided. A cancellation
  fixture exposed different payoff bits. Preserve singleton-background calls
  and test batch/reverse/singleton equality before adopting an oracle speedup.

- `focused.sbatch` deliberately sets `TMPDIR` to lab storage. A plain
  `tempfile.TemporaryDirectory()` therefore is not node-local; diagnostic-only
  traces need an explicit local directory. Preserve production request/response
  placement when diagnosing I/O, and record any changed placement separately.
  `focused_campaign.slurm_allocation()` returns selected stable fields, excluding
  `TimeLimit`; query the exact task separately when verifying its wall-time limit.

- Focused model-quality preparation exclusions omit `maximum_seconds_per_instance`;
  they were not rejected by a cost threshold. Report sanitization must retain those
  exclusions without requiring or inventing a time limit. Qualify exports against
  real prepared snapshots as well as cost-based fixtures; never rewrite frozen
  snapshots or their run identities to fit an older report schema.

- The superseded `benchmark/ROADMAP.md` and multi-phase launchers were removed
  from the active checkout. Their frozen copies and Git history retain historical
  reproduction. Use `benchmark/FOCUSED.md` and its campaign journal for job state;
  historical descriptions do not authorize reviving cancelled jobs.

- Node-local dependency staging must set `PYTHONPATH` before starting the Python
  interpreter. Updating it or `sys.path` inside the worker leaves `.pth` bootstrap
  modules such as `_virtualenv` loaded from the shared environment. Qualify the
  actual allocation entrypoint with a fresh interpreter before broad admission;
  successful estimator subprocess pilots alone do not test parent startup.

- Recovery `started.json` hardware describes the host and affinity; it has no
  `slurm_job_id`. Each result's `worker.slurm_job_id` matches the allocation's
  numeric `JobId`, which can differ from the canonical `ArrayJobId_ArrayTaskId`
  stored in the attempt header. Validate both identities against the allocation,
  using real serialized fixtures rather than adding nonexistent header fields.

- Campaign watcher messages must follow the active plan named in `WAKEUP.md`.
  A hard-coded instruction to continue `ROADMAP` phases survived the focused
  campaign replacement; successful delivery alone does not validate message scope.

- A held Slurm job requested with `--nodes=1` can report `NumNodes=1-1`
  before allocation, then `NumNodes=1` while running. Accept either exact
  one-node spelling when validating held jobs; retain all CPU, memory, time,
  host and no-GPU checks. If submission already succeeded, reconcile the same
  held job and preserve its intent, reservation and reviewed input hashes;
  do not resubmit or edit an admitted candidate to fix the validator.

- The focused CPU ledger retains each original reservation after settlement.
  `settled` overrides its reserved cost with actual CPUTimeRAW; overlap between
  these collections is required, not duplicate accounting. Do not remove
  historical reservations or reject that overlap. Only an active worker must
  have a reservation without a settlement. Reuse `budget_status` and test
  admission helpers against the actual ledger schema.

- Legacy focused collectors authenticate each snapshot's `cell_timeout_policy`
  but omit that field from the public suite projection. A strict cross-component
  composer therefore cannot infer a missing policy from public metadata alone.
  Carry it through an explicit adapter authenticated against the original suite,
  preparation plan and snapshots; preserve original rows and run IDs. Interrupted
  journal audits also need authenticated absent-file facts, not only hashes of
  files that exist, and must recheck those absences before final composition.

- Native neighbor preparation in `games.py` historically discarded the bounded
  training indices returned by `load_dataset` and split the full returned array
  again. Its player limit therefore cannot be inferred from the loader's default
  512-row cap. New explicit `training_rows` recipes use the loader's fitting
  pool directly, including training-only imputation; retain that path and its
  recorded nested row/feature selections when reproducing those games.

- `UV_NO_SYNC=1` prevents shared-environment installation but can still create new
  `.pyc` files when development commands import modules. An old complete dependency
  inventory may then reject that shared directory. Build later byte-identical
  node-local archives from the already authenticated node-local copy; verify every
  copied byte against its original receipt. Never silently relax the inventory.

- Focused/comprehensive summary weights currently follow application, subtype,
  then recipe (`summary.weight_groups`); they do not contain a dataset-level
  balancing step. More recipes can give a dataset greater weight within an
  application. Describe the implemented hierarchy accurately in public docs;
  do not claim equal dataset weights or change frozen weighting through a copy edit.

- At 960 pooled workers, node-local `PYTHONPATH` dependencies alone did not avoid
  startup timeouts: `/proc` syscall evidence showed shared CPython standard-library,
  `lib-dynload`, and automatic virtualenv-site lookups blocked in BeeGFS. Stage the
  exact standard-library and frozen-source bytes too, and verify the resulting
  runtime provenance; preserve the pinned interpreter spelling and shared environment.
- Sending STOP to a Slurm pooled job can stop its batch parent while descendants
  launched in fresh sessions keep running. Verify every owned PID in the exact
  allocation cgroup before declaring containment; parent state is insufficient.
- Standard-library path filters must account for both the `cpython-3.13` alias
  and `cpython-3.13.13` installation spelling. Filtering only the latter left
  shared `lib-dynload` searches active under full concurrency. Verify the actual
  remaining `sys.path` entries or blocked syscall paths, not just a serial probe.
- A node-local import hook runs after executable loading and CPython startup.
  At 960 workers, a trivial pinned-Python launch spent 11.7 seconds in `execve`
  on shared storage before the hook could run. Successful local imports alone
  do not establish that a 30-second cell limit leaves adequate estimator time;
  distinguish startup overhead from recorded estimator CPU time when explaining
  operational timeouts, and preserve the original attempt outcomes.
- Hopper's `/usr/bin/python3` is older than the benchmark runtime. It can run
  small standalone monitoring scripts, but importing repository helpers can fail
  on newer Python syntax or runtime types (for example `isinstance(x, int | float)`).
  Use the pinned benchmark interpreter for repository code and ledger helpers.

- A bounded campaign can finish its admitted allocations with unattempted cells.
  Keep execution completeness separate from explicit final coverage accounting;
  terminal Slurm state does not prove that the global CPU budget was exhausted.
  Partitioned reports retain coverage in lazy detail blocks, but that alone does
  not make unprepared games visible in the dashboard. Preserve missing results
  as missing, expose the intended design, and require an authenticated closeout
  before publishing an incomplete cohort.

- The About page serves committed `benchmark/site/about.html`, generated from
  `about.md`; deploying changed Markdown alone leaves the visible article stale.
  After editing the Markdown, run `benchmark/render_about.py` and its `--check`
  mode, and commit both files.

- `RecordStore` deletes its disposable SQLite file on exit, including exceptions.
  Long exports must save authenticated collection checkpoints before reference,
  summary and packaging work; otherwise a late export error repeats hours of
  collection. Operational pool checkpoints preserve rows and original run IDs.
- Equal payoff tables can occur across deliberately different comprehensive
  recipes. Deduplicate within application/subtype/recipe/qualification role;
  preserve declared weights across branches and record their equal-payoff groups.

- GitHub releases allow at most 1,000 assets per release; the comprehensive
  report has 1,154 partitions. Bundle the exact manifest and partitions in one
  checksum-pinned stored ZIP for Pages deployment, retaining each member's hash.
  The optional `data-source.json.archive` transport avoids changing report bytes.
  See https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases.

- The comprehensive node-local runtime archive contains redundant stdlib files
  under `src` as well as `stdlib`, plus an inactive copied `stdlib/site-packages`.
  gpu01 lacked three zero-byte duplicate package markers while their canonical
  stdlib copies and every scientific source file were unchanged. Validate the
  actual active import trees and explicit equivalences; preserve unexpected
  differences as errors. A bootstrap failure on one host does not imply that
  sibling allocations failed: inspect each job's receipts and claims before
  stopping any running sibling.

- Even a fully node-local `PYTHONPATH` and `sitecustomize` hook leave normal
  Python `.pth` scanning before that hook. October 8 cgroup samples found about
  40% CPU utilization with estimator children blocked reading the shared editable
  package `.pth`. A `-S` operational adapter must explicitly run the authenticated
  hook and propagate to preparation/evaluation/isolated-estimator children; verify
  unchanged source provenance and retain the pinned `sys.executable`. Do not edit
  active worker scripts or assume this removes executable/shared-artifact I/O.

- Combined campaign pools can preserve original task directories as authenticated
  symlinks. The reproduction packager's blanket nonsymlink input guard rejects
  those otherwise valid snapshots. Resolve only links pinned in the collection's
  authenticated selection map, verify each target and its ancestry, and retain
  every original artifact hash check. Do not broadly enable arbitrary symlinks
  or rewrite original snapshots merely to package the combined cohort.
  Snapshot pins can use the case-link path while artifact pins use its canonical
  target. Map a missing alias pin only through that authenticated link to an
  existing canonical pin; retain the artifact's byte-hash check.
- Browser worker queries now prefer precomputed summaries, whose deferred query
  counts are `null`. For exact main/worker parity, request `load_details: true`
  on the worker; separately compare the fast path against Python scores with
  the established numerical tolerance and exact coverage/selection checks.
- Select reproduction report files from the authenticated manifest's asset
  descriptors. A `partition-` filename prefix also matches UI scripts such as
  `partition-about.js`, which are not result partitions.
- Hopper's Slurm node names can be short names (`gpu01`) while `hostname` and
  Python report fully qualified names (`gpu01.cluster`). Runtime receipts retain
  the actual hostname; map its short name explicitly when matching a Slurm node
  or node-keyed admission, and verify the full receipt hostname separately.
- Hopper's normal QOS also limits a user to 128 running jobs. Hundreds of
  single-CPU array tasks can exhaust those slots while using few CPUs and block
  Hopper Monitor. Check both CPU and job limits; group independent workers into
  shared allocations and reserve job slots for monitoring. Preserve active claims
  and estimator intents when changing allocation grouping.
