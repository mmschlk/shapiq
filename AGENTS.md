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
