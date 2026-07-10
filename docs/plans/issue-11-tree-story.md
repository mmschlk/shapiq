# Issue 11 — The tree story: model-specific games and closed-form explainers

Status: **slice 1 landed 2026-07-10** — pure-Python interventional path; slice 2 is the C
extension.

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
converted scikit-learn models.

## Slice 2 — the C extension (next)

Bring v1's `interventional/cext` (interventional.cpp, weights.cpp) over, adapted:

- The kernel's weight tables are exactly the `omega`/coefficient tables the Python path
  computes from `index.derivative_weights` — the adaptation is feeding declared index
  weights instead of v1's name-keyed tables (v1's CUSTOM weight_fn path becomes "any
  cardinal index").
- Keep the Python path as the reference; the cext registers as a faster implementation of
  the same `tree_explanation` arm, with bit-parity tests against the Python path.
- Build machinery: setup.py compilation + cibuildwheel; v2's pyproject still carries the
  stale v1 test-command referencing `shapiq.tree.*.cext` — replace it when the cext lands.
- v1's dense-flatten vs sparse switch (`max_order > 3` or > 1e6 dense results) informs
  whether the cext returns dense blocks or sparse pairs.

## Later slices

- `PathDependentTreeGame` as the sibling game (TreeSHAP path-dependent semantics; weights
  from node sample counts — `TreeModel` will need `node_sample_weight` back).
- More converters: xgboost / lightgbm / catboost (v1 has C++ parsers for their dumps).
- Background-dataset baselines (v1 supports reference data, not just a point): a batch of
  baselines maps naturally onto explanation targets.
- Linear TreeSHAP as another registered family if wanted.
