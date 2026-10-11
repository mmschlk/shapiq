# Interventional tree kernel: redesign plan

Status: implemented in the `perf/interventional-sparse-flat-map` PR (stages 1-7 below; the
section "What was built" records the result). Every stage was gated by the main-versus-branch
comparison harness (identical values on the model grid) and by the deep-tree and sweep
benchmarks.

## Goal

One output layout, two fill algorithms, one leaf update. No caches that assume a second
call, no feature-count limit, no fallback kernels behind thresholds.

```text
preprocess (shared)      fill (one of two)              readout (shared)
──────────────────       ─────────────────────────      ────────────────
trees + order            cohort fill: background rows   layout_to_dict
  -> subset layout       single-reference fill: 1 row     -> {tuple: value}
  (tables, index,            both emit leaf events
   per-order mode)           into the shared leaf update
```

The interventional explanation is a sum over leaf events `(E, R, cohort size, leaf value)`:
for every subset `S` of `E ∪ R` up to the order, add `weight(|E|, |R|, |S∩E|, |S∩R|) ·
cohort size · leaf value` at the output position of `S`. The two fills differ only in how
they produce the events.

## What is removed

- `compute_interactions_batched_sparse`, the per-reference DFS in `interventional.cpp`, the
  `FlatSubsetMap` and the `BitSet`-keyed `SparseInteractionMap`, and the two `*_to_pydict`
  readouts. Their job is done by the single-reference fill (one row) and the cohort fill (many).
- `WeightCache` (memoising hash map keyed by an 8-tuple) and the per-call
  `structural_weight_table` built through it. Weights become a plain array computed at the
  top of each call for the `(e, r)` pairs the ensemble can produce.
- The route selection in Python: `_STRUCTURAL_MAX_ROWS`, `_use_sparse_path`, the sparse branch
  of `explain_function`, and the completion guard that only existed because of the budget.
- The base-n 64-bit key and its overflow mode (`keys_fit`, `has_index = False`, the row-set
  fallback in the collector).
- `shapiq/tree/subset_layout.py` as a layout mirror: the collector returns its offsets, Python
  stops deriving layout arithmetic. The small table helpers used by tests may stay.
- Fixed-size per-leaf buffers of any kind, and feature-width bitsets in the walk.

## What stays

- The co-occurrence layout (`subset_tables.hpp`, shared with the quadrature kernel): tables
  of the subsets that co-occur on some path, in lexicographic order, with a flat open-addressing
  index. Measured against the alternatives this week: per-tree dense is 30x larger on boosters,
  per-leaf row maps blow memory on deep forests.
- The per-order mode: an order-2 or order-3 table that holds more than
  `_STRUCTURAL_COMPLETE_ABOVE` (0.5) of all `C(n, s)` subsets is completed, and subsets in a
  complete table are positioned by lexicographic rank instead of a probe. Decided once in
  preprocessing, invisible to both fills. The threshold stays; the row-budget guard goes with
  the budget. Completion is skipped when `C(n, s)` would not fit the int32 row limit.
- The cohort walk: one DFS per tree carrying the cohort of reference rows that reached the
  current `(node, E, R)` state, partitioning at each split. This is the source of every
  speedup the PR has.
- `layout_to_dict` and `predict_ensemble_sum`.

## Stages

Each stage leaves a building, passing extension. Order chosen so that the risky walk rewrite
comes after the pure deletions.

### 1. Weights as a per-call array (`weights.hpp`)

`build_weight_table(index, max_order, max_path_width, n_features, custom) -> std::vector<double>`
indexed `[((e * stride + r) * k1 + s_cap_e) * k1 + s_cap_r]`, `stride = max_path_width + 1`,
entries with `e + r > max_path_width` left zero. Every entry is evaluated once with
`weight_func`; a custom table (from Python's `weight_fn`) is copied into the same shape. The
kernel receives one float64 array and has no `CUSTOM` branch. `WeightCache` is deleted.

### 2. Leaf update (`leaf_update.hpp`)

```cpp
struct LeafEvent { const int32_t *feats; const uint8_t *in_e; int n; int e; int r; double scaled_value; };
template <int ORDER, bool DENSE2, bool DENSE3> struct LeafUpdate { void operator()(const LeafEvent&); };
```

`feats` sorted ascending, `in_e[i]` 1 for E and 0 for R. Orders 1-3 are unrolled with the
addressing mode a template parameter, so the inner loops have no branches; orders 4 and up use
the recursive enumeration with probes. Rows come from `layout.row(key, s)` or from the
lexicographic rank; the output offset per order from the layout. A dispatch function picks
the instantiation from `(max_order, layout.dense2, layout.dense3)` once per call.

### 3. Single-reference fill (`single_reference_fill.hpp`)

For one reference row `r` and explain point `x`, E and R at a leaf are determined by the
leaf's position: at each node on its path with feature `f`, if `x` and `r` route the same way
and the leaf lies on that side, `f` joins neither set; if they route the same way and the leaf
lies on the other side, the leaf is unreachable; if they diverge, `f` joins E when the leaf is
on `x`'s side and R when it is on `r`'s side. One DFS per tree carrying the path, no cohort,
no stack of bitsets. The boolean proxy tree (ProxySHAP: `r = 0`, `x = 1`) is the special case
where every node diverges. Binding: `fill_single_reference(ensemble, x, r, layout, weights, out)`.

This fill is also the oracle for the cohort fill: the cohort result over a background
dataset equals the average of single-reference results over its rows.

### 4. Cohort fill (`cohort_fill.hpp`)

The walk as today, with three changes:

- Frames are `(node, begin, end, depth, feature, flag)` with `flag ∈ {E, R, agree}`; the E/R
  sets of a frame are the deltas along its ancestor chain, reconstructed through a shared
  per-DFS path array indexed by depth (LIFO order keeps every ancestor's delta in place). No
  bitsets, no allocation per frame, no feature-count dependence.
- Sorted leaf features come from the layout: preprocessing records, per leaf, its distinct
  path features in ascending order and, per path node, the position of its feature in that
  order. At a leaf event the E/R flags are scattered into those positions, so the event is
  sorted without sorting. (Preprocessing walks every path already.)
- The partition keeps reading reference values (no per-explainer routing cache: the
  explainer must stay cheap when built for a single call, as in ProxySHAP).

Binding: `fill_cohort(ensemble, reference_rows, x, layout, weights, out)`. Thread-local
output blocks while `threads × rows` is small; atomic adds into the shared output above that,
so there is no row cliff.

### 5. Layout changes (`subset_tables.hpp`)

- 128-bit keys: the base-n encoding kept, as two 64-bit words, so `n^order` fits for any
  practical `n` and order (`n ≤ 2^32` at order 4, `2^25` at 5, `2^21` at 6). The index stores
  both words per slot; `row` compares both. `keys_fit` stays as the 2^127 check, applied to
  `min(max_order, widest path)` so a high order on shallow trees is never refused; when the
  keys still overflow, both kernels fall back to the same binary search of the sorted
  tables (`TableSearch`), mirroring the quadrature kernel's original fallback.
- The geometry stays derived from `counts` (and `n_features`) on both sides; Python keeps
  only the `block_bits` and offset rules in `subset_layout.py` and never re-derives keys.
- The numpy oracle in the tests mirrors the key as a Python int (no width concern).

### 6. Python (`interventional/computer.py`)

The explainer builds the layout in the constructor (needed by every call), keeps the ensemble
block, and in `explain_function` computes the weight array and calls `fill_single_reference`
when it holds one reference row and `fill_cohort` otherwise. `_explain_structural`, the
sparse branch, `_use_sparse_path`, `_STRUCTURAL_MAX_ROWS` and `_complete_above` are removed.
`_STRUCTURAL_COMPLETE_ABOVE` stays as the single policy constant. ProxySHAP's route is
unchanged in what it calls; it lands on the single-reference fill by construction.

### 7. Tests

- Existing: the subset-table tests adapt to the returned geometry; the route tests
  (`_use_sparse_path`, `_STRUCTURAL_MAX_ROWS`) are replaced.
- New: cohort equals the average of single-reference results (every index, orders 1-4, deep
  and shallow trees, categorical with NaN); the 70-feature chain tree on both fills; a wide
  model at order 5 (keys beyond 64 bits); ProxySHAP's boolean tree against the numpy reference.
- Gate: `tests/.../test_correct_calculation.py` (numpy references) and the main-versus-branch
  harness on the 13-model grid plus the edge grid, zero mismatches.

### 8. Benchmarks

Rerun the PR figures (summary grid, feature sweep, random-forest figure) and the deep-tree
cells. Acceptance: no cell slower than the current branch beyond noise, order-1/2 forest cells
back at main parity (the generic leaf update was the cause of the 0.8-0.9x), ProxySHAP
scenario unchanged (sub-millisecond either way).

## Open decisions

- Thread-local blocks versus atomics: the switch size (start at 32 MB of blocks).
- Whether orders 4+ get a ranked mode later (needs a general lexicographic rank; not now).
- Whether `layout_to_dict` should skip exact zeros for both explainers (interventional does,
  quadrature does not; unchanged here).

## What was built

Files in `src/shapiq/tree/interventional/cext/`:

| file | role |
|---|---|
| `cext.cc` | bindings only: `fill_cohort`, `fill_single_reference`, `predict_ensemble_sum`, plus the shared `preprocess_subset_tables` and `layout_to_dict`; argument parsing (`FillCall`), the thread strategy (`run_fill`) |
| `leaf_update.hpp` | `Layout`, `LeafEvent`, `LeafUpdate<ORDER_CLASS, DENSE2, DENSE3, ATOMIC>`, `dispatch_leaf_update` |
| `cohort_fill.hpp` | the walk over a background dataset: frames `(node, begin, end, depth, feature, flag)`, path array by depth, per-feature membership |
| `single_reference_fill.hpp` | the one-row recursion: E/R from the leaf's position; `PathScratch`, `gather_leaf_event` (shared with the cohort fill) |
| `weights.cpp` | the index weight functions and `build_weight_table` (per call, `e + r <= longest path`) |
| `utils.cpp` | `IndexType`, `DecisionType`, `Tree` |

Removed: `interventional.cpp` (per-reference DFS, flat map), `BitSet`, `StackFrame`,
`SparseInteractionMap`, `WeightCache`, `compute_interactions_batched_sparse`,
`compute_interactions_cohort`, the Python route selection (`_STRUCTURAL_MAX_ROWS`,
`_use_sparse_path`), the completion guard.

Deviations from the plan above, with the reason:

- The walk keeps reading reference values in its partition instead of precomputed routing
  bits: the explainer must stay cheap when built for one call (ProxySHAP), and the proxy
  scenario is dominated by Python either way (tens of microseconds in C++).
- Sorted leaf features are produced by an insertion merge of the path entries at the leaf
  (`gather_leaf_event`) rather than by per-leaf sorted-path tables from preprocessing; paths
  are short and this keeps preprocessing to the subset layout.
- The thread strategy is the planned one: private blocks up to 32 MB in total, atomic adds
  above (verified against one thread on 60 MB layouts).
- Keys are 128 bits as planned (`Key128` in `subset_tables.hpp`: mixed-radix multiply-add
  and divmod, two int64 words per index slot, `key_hash` over both words); the quadrature
  kernel and the numpy oracle follow. `keys_fit` now means "below 2^127" for
  `min(max_order, widest path)`: order 16 on 255 features with 8-feature paths still
  overflows; no index is built then and both kernels search the sorted tables instead
  (`subset_tables::TableSearch`, used by `LeafUpdate::probe_add` and
  `QuadWorkspace::merged_position`): a gallop from a per-order cursor plus a bisection, which
  works because one leaf event, like one quadrature edge, emits its interactions in
  increasing order. The two routes are mirrored: index when the keys fit, search otherwise,
  with nothing refused.
- A complete order-2/3 table has no index entries (an empty 2-slot block in the geometry):
  both kernels address it by lexicographic rank (`LeafUpdate` and `QuadWorkspace::merged_rank`).
  Completion is skipped when `C(n, s)` would exceed the int32 row limit.
- The unrolled orders 1-3 of the leaf update keep int64 key arithmetic below 2^21 features
  (where `n^3 < 2^63`) and route wider models to the general enumeration.

## Review pass (2026-10-09)

An independent read of the two kernels after the restructure found one regression and one
older bug, both fixed, plus cleanups:

- High `max_order` on shallow trees was refused once `n_features ** max_order` reached the
  key limit although no such subsets exist; the key check now uses `min(max_order, widest
  path)` (`max_path_width` in the header; both kernel bindings trust the collector's index).
- `InterventionalTreeSHAPIQ(weight_fn=...)` dropped the terms with interaction size equal to
  `|E| + |R|` (present on main too); fixed and tested against the built-in SII.
- Complete order-2/3 tables keep an empty two-slot index block (both kernels rank them); one
  thread writes straight into the output; per-call scratch instead of thread-local state; the
  dispatch raises instead of silently computing order 1; dense flags without int64 overflow;
  the strided-output writeback is resolved; `member[]` is cleared along the path.
- Tests added: keys with a non-zero high word, every ranked/probed instantiation of orders 3
  and 4, the shallow-tree high-order case, the search fallback of both kernels (the
  quadrature kernel called without an index; the interventional fills with the index removed
  and on an overflowing order-8 chain), custom weights versus SII.
