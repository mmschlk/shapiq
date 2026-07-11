# Issue 11 — The tree story: model-specific games and closed-form explainers

Status: slice 1 (pure-Python interventional path) landed as `8ad90070`; slice 2 (the C
kernel) landed as `7bd6ea3a` after the five-agent review round.

## The design

The v1 tree stack (branch `main`: `shapiq/tree/` with conversion cexts, interventional and
linear TreeSHAP cexts, TreeSHAP-IQ) re-enters v2 with the game carrying the semantics:

- **`TreeModel`** (`shapiq/trees/_model.py`): the unified node-array layout; leaf values may
  carry trailing value axes (class probabilities become the game's `value_shape`).
- **`to_tree_model`** (`_conversion.py`): flextype-dispatched conversion. `TreeModel`s and
  sequences pass through; scikit-learn trees and forests register lazily
  (`delayed_register` on `sklearn.tree._classes.BaseDecisionTree` /
  `sklearn.ensemble._forest.BaseForest`) — shapiq never imports scikit-learn on its own.
  Forest leaf values are scaled by `1/n_trees` because tree games sum trees.
- **`InterventionalTreeGame(trees, inputs=x, baseline=r)`** (`_interventional.py`): the
  interventional semantics of baseline masking computed exactly on the tree structure —
  the vocabulary deliberately mirrors `BaselineMasker(inputs, baseline)`. The game
  precomputes per-leaf present/absent constraints (`LeafConstraints`) by routing both
  points; features both points route the same way constrain nothing. It is a plain `Game`,
  so `ExactExplainer`, `Regression`, and the samplers consume it unchanged (tested), and
  the constraint tables are what slice 2's C kernel will consume.
- **`TreeExplainer(game, index)`** (`explainers/_tree.py`): closed form in one pass over
  the leaves instead of `2**n_players` evaluations. The algorithm dispatches on the exact
  game type (`tree_explanation`, one atomic registry per ADR 0011); the planned
  path-dependent game registers as a sibling. Any index with the cardinal capability works:
  a leaf with sets `(E, R)` carries Moebius masses `m(E|W) = (-1)^|W| v` for `W` inside
  `R`, and `I(T) = sum_{Q >= T} m(Q) * omega_t(|Q\\T|)` with `omega` the superset-summed
  `derivative_weights` — per-leaf coefficients depend only on `(|E|, |R|, |T&E|, |T&R|)`,
  the same counting trick as v1's C weight tables. Output is a `SparseExplanationArray`
  (only path-co-occurring interactions carry mass; zero default), with the centered-game
  correction at the empty interaction for min-size-0 indices.

Parity is pinned against `ExactExplainer` for SV, BV, SII, BII, WeightedBII(p), CHII,
STII, Moebius, and Co-Moebius, on hand-built ensembles, vector-valued leaves, and
converted scikit-learn models. The value-shape story holds end to end: scikit-learn
classifier leaves store class fractions, so the converted game IS `predict_proba` and one
explanation covers every class (pinned), exactly as booster multiclass margins become
vector games — and the interventional C kernel serves vector values too (leaf values are
rows of the flattened value shape accumulated into an arena of equally wide sums; measured
9.5x on a 3-class 200-tree depth-14 forest at SII order 2, 128 -> 13 ms, with the scalar
path unregressed at 12.5x).

## Slice 2 — the C extension

v1's kernel was adapted rather than ported wholesale. What v1 computed C-side in
`weights.cpp` (`get_weight(n, e, r, a, b, t, index_enum, ...)`) is exactly the
`(|E|, |R|, |T&E|, |T&R|)` coefficient the Python path derives from
`index.derivative_weights` — so the v2 kernel takes that table as INPUT, computed in
Python by the same `_leaf_coefficient` as the pure path (one source of truth for the
math, any cardinal index served, no index enum in C). The kernel
(`src/shapiq/trees/cext/interventional.cc`, ~200 lines vs v1's ~3500 across four files)
consumes the game's flattened `LeafConstraints` (no tree walk in C) and runs only the hot
loop: per-leaf combination enumeration accumulating into an
`unordered_map<uint64, double>` with interactions packed as four 16-bit player fields.

- Serviced: scalar leaf values, `order <= 4`, `n_players <= 65534`; everything else falls
  back to the pure path silently (`_use_cext`), pinned by tests.
- Binding: raw CPython API + buffer protocol — no numpy headers, no pybind11, no OpenMP
  (v1's macOS static-libomp saga deliberately deferred; the kernel is single-threaded).
  `Extension(optional=True)` keeps pure-python installs working without a compiler.
- Build: minimal `setup.py` (C++17, -O3, no -ffast-math — parity over speed); pyproject's
  stale v1 cibuildwheel test-command replaced, numpy dropped from build requires, macOS
  libomp before-build removed.
- Parity: kernel vs Python paths agree to float64 on random forests across SV, SII,
  BII(3), WeightedBII(p), Moebius(4), CoMoebius(3) including the empty interaction;
  kernel path also pinned against ExactExplainer.
- Measured (200 trees, depth 14, 40 features, ~35k constrained leaves, baseline far from
  the explicand): SII order 2: 607 ms -> 42 ms (14.4x); BII order 3: 1758 ms -> 208 ms
  (8.4x). With a mean baseline the E/R pruning shrinks the workload so far (~300 leaves)
  that Python is already at ~3 ms — the kernel matters for deep ensembles, high orders,
  and distant baselines.
- Future knobs recorded: OpenMP over leaves (release the GIL around the leaf loop first —
  it is held for the whole hot loop today), order > 4 via wider keys (a seam rewrite, not
  a knob — the packed-uint64 return ABI hard-wires 4x16 bits), and moving the E/R
  extraction itself to C if game construction ever dominates. Vector leaf values landed
  (value_width rows into an arena of sums), so `_use_cext` gates only on order and player
  count.
- Review-fleet fixes applied (2026-07-10, five-agent round): the bridge's dense
  coefficient table skips infeasible (|E|, |R|) cross pairs whose maxima came from
  different leaves (was: omega IndexError on ordinary forests); the kernel validates all
  cheap cross-buffer invariants instead of trusting five of eight (was: SIGBUS or silent
  garbage through the private seam); zero weights are skipped before binomial factors so
  the Moebius family stays serviceable at any player count (dense-weight indices still
  overflow near 1023 players — log-space omega accumulation is the honest fix, still
  open); `[tool.uv] cache-keys` covers the .cc so editable rebuilds pick up kernel edits
  (was: silently testing stale binaries); direct kernel-call error tests, the
  different-leaves regression, single-node trees, identical points, and large-n Moebius
  are pinned. Known-open from review: tree paths hardcode float32 output under JAX x64,
  and CHANGELOG.md is referenced by pyproject's readme but does not exist.
- Array story + vectorized evaluation (2026-07-10, review-fix round 1): tree seams accept
  any array backend — `TreeModel` fields and the game's `inputs`/`baseline` normalize via
  `as_host_array` (direct NumPy view, `to_values` fallback for device tensors), because
  split routing against float64 thresholds is an exact host-side computation that JAX
  without x64 cannot represent. Game evaluation is the opposite direction: the whole
  ensemble concatenates into one pre-transposed constraint set at construction (int32
  membership counting, exact comparisons instead of float32 equality), so `_call` is a
  single two-matmul pass — measured 8-58x over the per-tree loop with its per-call
  host-to-device conversions (100 trees depth 12: 16 ms -> 0.3-1.3 ms per batch).
  Explainer output follows the default JAX precision (float64 under x64 — closes the
  float32-under-x64 finding; pinned with `jax.enable_x64()` tests). Library-wide policy
  applied here first: no `assert` in src and no bandit-rule noqa — the game's
  ensemble-total assert dissolved with the loop, the cext-guard assert became a
  RuntimeError.

- Host-vs-jax evaluation revisited (2026-07-10, decided: keep jax): the question was whether
  `_call` should compute in NumPy float64 all along instead of jax copies of host data.
  Measured (200-tree/4459-leaf forest, exact int32 counting vs host float64 BLAS): jax is
  2.5-3x faster at realistic batches (batch 1024: 1.4 ms vs 4.8 ms; batch 16384: 29 ms vs
  66 ms; only micro-batches of ~64 favor NumPy by dispatch overhead), values agree. The Array
  API has no referent here — both game boundaries are contractually jax (coalitions in from
  samplers, values out to evidence), so a namespace would always resolve to jax; the masker
  rule ("follow the user's arrays") does not transfer because the game has no user arrays,
  only host tree structure. The refined tree-world law: **host float64 where exactness is
  semantic** (split routing and closed forms — a rounding flip changes *which* leaf), **stack
  precision where it is ordinary numerics** (value accumulation, like every other game;
  follows the x64 knob). The float64-to-float32 step happens once at construction, the same
  downcast every game's values undergo — per-call bouncing died with the vectorization.

- Booster converters landed (2026-07-10): XGBoost and LightGBM join `to_tree_model` with the
  same lazy registration as scikit-learn. Conversion is hot-path infrastructure (AutoML-scale
  estimators depend on these libraries), so a second ~500-line kernel
  (`src/shapiq/trees/cext/conversion.cc`, module `_conversion_cext`) parses the FAST dumps —
  XGBoost's `save_raw()` UBJSON (all key lengths are 8-byte `L` int64s; `tree_info` is a
  counted-but-untyped array) and LightGBM's `model_to_string()` text (leaf children encoded as
  `~id`, `decision_type` bit 0 guards categorical splits). The same seam split as the
  interventional kernel: C owns the byte loops and returns flat arrays; Python owns every
  policy — the one-ulp `nextafter` threshold shift (XGBoost routes on `x < t`, the layout on
  `x <= t`; pinned by boundary tests placing the explicand exactly on a split), leaf values
  from `split_conditions`, `base_score` as a lone-leaf constant tree (logit-transformed for
  logistic objectives, read from `save_config()`), and multiclass rounds as vector-valued
  leaves (no class_label knob; explanations come out per class). Both converters keep a pure
  Python fallback over the slow dumps (JSON / dict walk) as the correctness oracle and for
  installs without a compiler; kernels fall back silently on unparseable streams. Measured
  (30 features, depth 8): LightGBM 500 trees 371 -> 22 ms and 2000 trees 1596 -> 70 ms
  (17-23x, the dict dump was the bottleneck); XGBoost 500 trees 170 -> 34 ms (5x) and 2000
  trees 226 -> 18 ms after two follow-up fixes: the kernel's per-tree exact `reserve()`
  defeated geometric vector growth (quadratic in tree count — an earlier note here blamed
  TreeModel construction, which measurement refuted: validation was 13 ms of 183), and the
  converters now build trees through `trusted_tree_model` (module-internal, unvalidated:
  producers guarantee the layout by construction, their parity suites pin it, and a
  revalidation test rebuilds converted trees through the validating constructor). The
  validating `TreeModel` constructor remains the only public path — no `validate=False`
  knob, since the teaching checks exist exactly for hand-built trees. Full-sweep parity vs native margins on every coalition, kernel-vs-python
  parses bit-compatible, laziness pinned by subprocess. CatBoost landed right after (added
  to the all_ml group): oblivious trees unroll into the binary layout — leaf-index bit ``j``
  is ``splits[j]``'s ``x > border`` (verified against native predictions), which is the exact
  complement of the layout's ``x <= threshold``, so borders carry over UNSHIFTED (no ulp
  trick); multiclass leaf values arrive grouped per leaf and become vector leaves;
  ``scale_and_bias`` becomes a leaf-value scale plus a constant tree. No C parser needed —
  measured 171 ms for 1000 trees depth 8 (511k nodes) in pure Python, because symmetric trees
  share their splits and the JSON stays small relative to the node count (v1's
  catboost_json.cc remains adaptable if this ever changes). Same full-sweep, border-boundary,
  multiclass, and closed-form-parity tests as the other boosters. Environment quirk, reproduced
  outside any sandbox: on this macOS wheel set the booster natives segfault in their
  data-ingestion paths once torch's machinery has run in the same process
  (OMP_NUM_THREADS=1 saves xgboost but not lightgbm), so the substantive conversion tests
  run in a fresh interpreter via one wrapper test — not a shapiq bug, and Linux CI may not
  need the isolation.

## Later slices

- `PathDependentTreeGame` as the sibling game (TreeSHAP path-dependent semantics; weights
  from node sample counts — `TreeModel` will need `node_sample_weight` back).
- NaN/missing-value routing (`default_left`) if explained points with missing features
  become a story; non-symmetric CatBoost grow policies (Depthwise/Lossguide) if requested.
- Background-dataset baselines (v1 supports reference data, not just a point): a batch of
  baselines maps naturally onto explanation targets.
- Linear TreeSHAP as another registered family if wanted.
