"""The interventional tree game."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from shapiq._shape import validate_n_players
from shapiq.games._base import Game
from shapiq.games._values import to_host_array
from shapiq.trees._model import TreeModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import Array

    from shapiq.coalitions import CoalitionArray


class LeafConstraints(NamedTuple):
    """Interventional reachability constraints of one tree's leaves.

    A leaf is reached by a coalition exactly when every ``present`` player is
    in the coalition and no ``absent`` player is; its row of ``values`` then
    enters the game value. Features on which the explained inputs and the
    baseline route the same way constrain nothing and appear in neither mask.
    The arrays are host NumPy: tree explainers consume them in exact
    ``float64``, while game evaluation runs on JAX copies made once at
    construction.
    """

    present: np.ndarray
    absent: np.ndarray
    values: np.ndarray


class InterventionalTreeGame(Game["Array"]):
    """Tree-ensemble game replacing absent players' features with a baseline.

    The interventional semantics of baseline masking, computed exactly on the
    tree structure: present players take the explained ``inputs``' feature
    values, absent players the ``baseline``'s, and the game value is the
    ensemble prediction at that mixed point. Because a tree routes any mixed
    point to exactly one leaf, the game reduces to per-leaf reachability
    constraints (``leaf_constraints``), which also give tree explainers their
    closed forms — the game type is what selects the tree-explanation
    semantics (a path-dependent sibling game is the planned alternative).
    Leaf values may carry trailing axes (class probabilities), which become
    the game's ``value_shape``.

    Example:
        >>> game = InterventionalTreeGame(to_tree_model(model), inputs=x, baseline=background)
        >>> explanation = TreeExplainer(game, SII(order=2)).explain()
    """

    def __init__(
        self,
        trees: TreeModel | Sequence[TreeModel],
        *,
        inputs: object,
        baseline: object,
    ) -> None:
        """Initialize the game and precompute the leaf constraints.

        Args:
            trees: One tree or a sequence of trees whose values add up to the
                ensemble prediction.
            inputs: Feature values of the explained point, shape
                ``(n_players,)``; any array backend (NumPy, JAX, torch) is
                copied to the host for exact split routing.
            baseline: Feature values of the interventional reference point,
                shape ``(n_players,)``; converted like ``inputs``.

        Raises:
            ValueError: If no tree is passed, the trees disagree on their
                value shape, a tree uses a feature beyond the inputs, or the
                inputs and baseline shapes do not line up.
        """
        tree_tuple = (trees,) if isinstance(trees, TreeModel) else tuple(trees)
        if not tree_tuple:
            msg = "the game needs at least one tree"
            raise ValueError(msg)
        inputs_array = to_host_array(inputs, np.float64)
        baseline_array = to_host_array(baseline, np.float64)
        if inputs_array.ndim != 1:
            msg = "inputs must be one explained point with shape (n_players,)"
            raise ValueError(msg)
        if baseline_array.shape != inputs_array.shape:
            msg = (
                f"baseline with shape {baseline_array.shape} does not pair with "
                f"inputs of shape {inputs_array.shape}"
            )
            raise ValueError(msg)
        self.n_players = validate_n_players(int(inputs_array.shape[0]))
        self.target_shape = ()
        value_shape = tree_tuple[0].value_shape
        for tree in tree_tuple:
            if tree.value_shape != value_shape:
                msg = "all trees must agree on their value shape"
                raise ValueError(msg)
            if tree.max_feature >= self.n_players:
                msg = (
                    f"a tree splits on feature {tree.max_feature} but the inputs "
                    f"carry only {self.n_players} features"
                )
                raise ValueError(msg)
        self.value_shape = value_shape
        self.trees = tree_tuple
        self.inputs = inputs_array
        self.baseline = baseline_array
        self.leaf_constraints = tuple(
            _leaf_constraints(tree, inputs_array, baseline_array, self.n_players)
            for tree in tree_tuple
        )
        # evaluation state: the whole ensemble concatenates into one constraint
        # set, stored pre-transposed so a coalition batch reaches every leaf of
        # every tree in a single (batch, players) @ (players, leaves) pass;
        # membership counting is exact in int32, values follow the default
        # JAX precision
        present = np.concatenate([leaves.present for leaves in self.leaf_constraints])
        absent = np.concatenate([leaves.absent for leaves in self.leaf_constraints])
        values = np.concatenate([leaves.values for leaves in self.leaf_constraints])
        self._present_by_player = jnp.asarray(present.T, dtype=jnp.int32)
        self._absent_by_player = jnp.asarray(absent.T, dtype=jnp.int32)
        self._required = jnp.asarray(present.sum(axis=1), dtype=jnp.int32)
        self._leaf_values = jnp.asarray(values)

    def _call(self, coalitions: CoalitionArray) -> Array:
        """Sum reachable leaf values per coalition in one ensemble-wide pass."""
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.int32)
        reached = (masks @ self._present_by_player == self._required) & (
            masks @ self._absent_by_player == 0
        )
        return jnp.tensordot(reached.astype(self._leaf_values.dtype), self._leaf_values, axes=1)


def _leaf_constraints(
    tree: TreeModel,
    inputs: np.ndarray,
    baseline: np.ndarray,
    n_players: int,
) -> LeafConstraints:
    """Extract per-leaf present/absent constraints by routing both points.

    Depth-first traversal: where the explained inputs and the baseline route
    a split the same way, the feature constrains nothing; where they differ,
    the inputs' branch requires the feature present and the baseline's branch
    requires it absent. Branches whose requirement contradicts an earlier
    constraint on the path are unreachable and pruned.
    """
    present_rows: list[np.ndarray] = []
    absent_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    stack: list[tuple[int, frozenset[int], frozenset[int]]] = [(0, frozenset(), frozenset())]
    while stack:
        node, present, absent = stack.pop()
        if tree.children_left[node] == tree.children_right[node]:
            present_row = np.zeros(n_players, dtype=bool)
            present_row[list(present)] = True
            absent_row = np.zeros(n_players, dtype=bool)
            absent_row[list(absent)] = True
            present_rows.append(present_row)
            absent_rows.append(absent_row)
            value_rows.append(tree.values[node])
            continue
        feature = int(tree.features[node])
        threshold = tree.thresholds[node]
        inputs_child = (
            tree.children_left[node] if inputs[feature] <= threshold else tree.children_right[node]
        )
        baseline_child = (
            tree.children_left[node]
            if baseline[feature] <= threshold
            else tree.children_right[node]
        )
        if inputs_child == baseline_child:
            stack.append((int(inputs_child), present, absent))
            continue
        if feature not in absent:  # the inputs' branch needs the feature present
            stack.append((int(inputs_child), present | {feature}, absent))
        if feature not in present:  # the baseline's branch needs it absent
            stack.append((int(baseline_child), present, absent | {feature}))
    return LeafConstraints(
        present=np.asarray(present_rows),
        absent=np.asarray(absent_rows),
        values=np.asarray(value_rows),
    )
