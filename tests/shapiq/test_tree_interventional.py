"""Tests for the interventional tree game and the closed-form tree explainer."""

from __future__ import annotations

from itertools import combinations, product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    BII,
    BV,
    CHII,
    FSII,
    SII,
    STII,
    SV,
    CallableGame,
    CoMoebius,
    DenseCoalitionArray,
    ExactExplainer,
    InterventionalTreeGame,
    Moebius,
    Regression,
    TreeExplainer,
    TreeModel,
    UnsupportedGameError,
    WeightedBII,
    to_tree_model,
)

N_PLAYERS = 4


def stump_tree():
    """A depth-3 tree over features 0, 1, 2 (feature 3 stays unused)."""
    # node0 splits f0; node1 (left) splits f1 into leaves 3/4; node2
    # (right) splits f2 into leaves 5/6
    return TreeModel(
        children_left=[1, 3, 5, -1, -1, -1, -1],
        children_right=[2, 4, 6, -1, -1, -1, -1],
        features=[0, 1, 2, -2, -2, -2, -2],
        thresholds=[0.5, 0.5, 0.5, np.nan, np.nan, np.nan, np.nan],
        values=[0.0, 0.0, 0.0, 1.0, 3.0, -2.0, 5.0],
    )


def second_tree():
    """A stump on feature 3 with a subtree the points route identically."""
    return TreeModel(
        children_left=[1, 3, -1, -1, -1],
        children_right=[2, 4, -1, -1, -1],
        features=[3, 0, -2, -2, -2],
        thresholds=[0.5, 10.0, np.nan, np.nan, np.nan],  # both points go left at node 1
        values=[0.0, 0.0, 7.0, 0.25, 0.0],
    )


INPUTS = np.asarray([1.0, 1.0, 1.0, 1.0])
BASELINE = np.asarray([0.0, 0.0, 0.0, 0.0])


def tree_game(trees=None):
    trees = [stump_tree(), second_tree()] if trees is None else trees
    return InterventionalTreeGame(trees, inputs=INPUTS, baseline=BASELINE)


def predict(trees, point):
    total = 0.0
    for tree in trees:
        node = 0
        while tree.children_left[node] != -1:
            feature = int(tree.features[node])
            node = int(
                tree.children_left[node]
                if point[feature] <= tree.thresholds[node]
                else tree.children_right[node]
            )
        total = total + tree.values[node]
    return total


def all_coalitions():
    rows = jnp.asarray(list(product([False, True], repeat=N_PLAYERS)))
    return DenseCoalitionArray(rows)


def test_game_values_match_routing_the_mixed_point():
    trees = [stump_tree(), second_tree()]
    game = tree_game(trees)
    values = game(all_coalitions())
    for row, mask in enumerate(product([False, True], repeat=N_PLAYERS)):
        mixed = np.where(np.asarray(mask), INPUTS, BASELINE)
        assert jnp.allclose(values[row], predict(trees, mixed), atol=1e-6)


def test_grand_and_empty_are_the_two_predictions():
    trees = [stump_tree(), second_tree()]
    game = tree_game(trees)
    ends = game(DenseCoalitionArray(jnp.asarray([[False] * N_PLAYERS, [True] * N_PLAYERS])))
    assert jnp.allclose(ends[0], predict(trees, BASELINE), atol=1e-6)
    assert jnp.allclose(ends[1], predict(trees, INPUTS), atol=1e-6)


CARDINAL_INDICES = [
    SV(),
    BV(),
    SII(order=2),
    BII(order=3),
    WeightedBII(p=0.3, order=2),
    CHII(order=2),
    STII(order=2),
    Moebius(),
    CoMoebius(),
]


@pytest.mark.parametrize("index", CARDINAL_INDICES, ids=repr)
def test_closed_form_matches_the_exact_explainer(index):
    game = tree_game()
    closed_form = TreeExplainer(game, index).explain()
    exact = ExactExplainer(game, index).explain()
    assert jnp.allclose(closed_form.baseline, exact.baseline, atol=1e-5)
    min_size = index.min_interaction_size
    order = game.n_players if index.order is None else index.order
    for size in range(min_size, order + 1):
        for interaction in combinations(range(N_PLAYERS), size):
            assert jnp.allclose(
                closed_form(interaction),
                exact(interaction),
                atol=1e-4,
            ), f"mismatch at {interaction}"


def test_batched_lookups_match_the_exact_explainer():
    game = tree_game()
    sparse = TreeExplainer(game, SII(order=2)).explain()
    dense = ExactExplainer(game, SII(order=2)).explain()
    singles = jnp.asarray([[player] for player in range(N_PLAYERS)])
    assert jnp.allclose(sparse(singles), dense(singles), atol=1e-4)
    pairs = jnp.asarray(list(combinations(range(N_PLAYERS), 2)))
    assert jnp.allclose(sparse(pairs), dense(pairs), atol=1e-4)


def test_explanations_are_sparse_over_the_tree_support():
    explanation = TreeExplainer(tree_game(), SII(order=2)).explain()
    # features 1 and 2 sit in different branches of the first tree and never
    # co-occur on a path, so their interaction carries no mass
    assert (1, 2) not in explanation.attributions
    assert jnp.allclose(explanation((1, 2)), 0.0)
    assert explanation.interaction_index == "SII"


def test_vector_leaf_values_flow_into_the_value_shape():
    two_class = TreeModel(
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        features=[0, -2, -2],
        thresholds=[0.5, np.nan, np.nan],
        values=[[0.0, 0.0], [0.8, 0.2], [0.1, 0.9]],
    )
    game = InterventionalTreeGame(two_class, inputs=INPUTS, baseline=BASELINE)
    assert game.value_shape == (2,)
    explanation = TreeExplainer(game, SV()).explain()
    exact = ExactExplainer(game, SV()).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(explanation((player,)), exact((player,)), atol=1e-5)
    assert explanation((0,)).shape == (2,)


def test_entry_gates_keep_teaching():
    quadratic = CallableGame(
        fn=lambda c: jnp.sum(jnp.asarray(c.to_dense(), dtype=jnp.float32), axis=-1),
        n_players=N_PLAYERS,
    )
    with pytest.raises(UnsupportedGameError, match="registered games are InterventionalTreeGame"):
        TreeExplainer(quadratic, SV())
    with pytest.raises(TypeError, match="discrete-derivative"):
        TreeExplainer(tree_game(), FSII(order=2))

    class MyTreeGame(InterventionalTreeGame):
        pass

    # a subclassed tree game inherits the closest registered ancestor's closed form
    lookalike = MyTreeGame([stump_tree()], inputs=INPUTS, baseline=BASELINE)
    reference = InterventionalTreeGame([stump_tree()], inputs=INPUTS, baseline=BASELINE)
    assert jnp.allclose(
        TreeExplainer(lookalike, SV()).explain()((0,)),
        TreeExplainer(reference, SV()).explain()((0,)),
        atol=0,
    )


def test_tree_model_validation_teaches():
    with pytest.raises(ValueError, match="leaf exactly when"):
        TreeModel(
            children_left=[1, -1, -1],
            children_right=[-1, -1, -1],
            features=[0, -2, -2],
            thresholds=[0.5, np.nan, np.nan],
            values=[0.0, 1.0, 2.0],
        )
    with pytest.raises(ValueError, match="feature 5"):
        InterventionalTreeGame(
            TreeModel(
                children_left=[1, -1, -1],
                children_right=[2, -1, -1],
                features=[5, -2, -2],
                thresholds=[0.5, np.nan, np.nan],
                values=[0.0, 1.0, 2.0],
            ),
            inputs=INPUTS,
            baseline=BASELINE,
        )


def test_to_tree_model_passes_hand_built_trees_through():
    tree = stump_tree()
    assert to_tree_model(tree) == (tree,)
    assert to_tree_model([tree, tree]) == (tree, tree)
    with pytest.raises(TypeError, match="no tree conversion"):
        to_tree_model(object())


def test_sklearn_trees_convert_and_explain():
    pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415 - requires sklearn

    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, N_PLAYERS))
    labels = features[:, 0] * features[:, 1] + features[:, 2]
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(features, labels)
    trees = to_tree_model(model)
    x = features[0]
    reference = features.mean(axis=0)
    game = InterventionalTreeGame(trees, inputs=x, baseline=reference)
    ends = game(DenseCoalitionArray(jnp.asarray([[False] * N_PLAYERS, [True] * N_PLAYERS])))
    assert jnp.allclose(ends[1], model.predict(x.reshape(1, -1))[0], atol=1e-5)
    assert jnp.allclose(ends[0], model.predict(reference.reshape(1, -1))[0], atol=1e-5)
    closed_form = TreeExplainer(game, SII(order=2)).explain()
    exact = ExactExplainer(game, SII(order=2)).explain()
    for size in (1, 2):
        for interaction in combinations(range(N_PLAYERS), size):
            assert jnp.allclose(closed_form(interaction), exact(interaction), atol=1e-4)


def test_sklearn_classifiers_become_probability_games():
    # classifier leaves store class fractions, so the converted game IS
    # predict_proba and one explanation covers every class
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415 - requires sklearn

    rng = np.random.default_rng(3)
    features = rng.normal(size=(300, N_PLAYERS))
    labels = np.digitize(features[:, 0] * features[:, 1] + features[:, 2], [-0.5, 0.5])
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    model.fit(features, labels)
    game = InterventionalTreeGame(
        to_tree_model(model), inputs=features[0], baseline=features.mean(axis=0)
    )
    assert game.value_shape == (3,)
    masks = list(product([False, True], repeat=N_PLAYERS))
    values = game(DenseCoalitionArray(jnp.asarray(masks)))
    mixed = np.where(np.asarray(masks), features[0], features.mean(axis=0))
    assert jnp.allclose(values, model.predict_proba(mixed), atol=1e-5)
    # the closed form explains all classes at once and stays efficient per class
    explanation = TreeExplainer(game, SV()).explain()
    total = sum(jnp.asarray(explanation((player,))) for player in range(N_PLAYERS))
    assert jnp.allclose(total, values[-1] - values[0], atol=1e-5)


def test_converted_trees_survive_revalidation():
    # converters build trees through the trusted fast path; rebuilding every
    # tree through the validating constructor must accept it and agree
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415 - requires sklearn

    rng = np.random.default_rng(2)
    features = rng.normal(size=(150, N_PLAYERS))
    model = RandomForestRegressor(n_estimators=4, max_depth=4, random_state=0).fit(
        features, features[:, 0] * features[:, 1]
    )
    trusted = to_tree_model(model)
    revalidated = tuple(
        TreeModel(
            children_left=tree.children_left,
            children_right=tree.children_right,
            features=tree.features,
            thresholds=tree.thresholds,
            values=tree.values,
        )
        for tree in trusted
    )
    coalitions = all_coalitions()
    inputs, baseline = features[0], features.mean(axis=0)
    trusted_game = InterventionalTreeGame(trusted, inputs=inputs, baseline=baseline)
    revalidated_game = InterventionalTreeGame(revalidated, inputs=inputs, baseline=baseline)
    assert jnp.allclose(trusted_game(coalitions), revalidated_game(coalitions), atol=1e-6)


def test_sklearn_forests_scale_to_the_mean_prediction():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415 - requires sklearn

    rng = np.random.default_rng(1)
    features = rng.normal(size=(150, N_PLAYERS))
    labels = features[:, 0] - 2.0 * features[:, 3]
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0).fit(
        features, labels
    )
    trees = to_tree_model(model)
    assert len(trees) == 5
    x = features[1]
    game = InterventionalTreeGame(trees, inputs=x, baseline=features.mean(axis=0))
    grand = game(DenseCoalitionArray(jnp.asarray([[True] * N_PLAYERS])))
    assert jnp.allclose(grand[0], model.predict(x.reshape(1, -1))[0], atol=1e-5)
    shapley = TreeExplainer(game, SV()).explain()
    exact = ExactExplainer(game, SV()).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(shapley((player,)), exact((player,)), atol=1e-4)


def test_any_array_backend_constructs_the_same_game():
    reference = tree_game()(all_coalitions())
    jax_game = InterventionalTreeGame(
        [stump_tree(), second_tree()],
        inputs=jnp.asarray(INPUTS),
        baseline=jnp.asarray(BASELINE),
    )
    list_game = InterventionalTreeGame(
        [stump_tree(), second_tree()],
        inputs=[1.0] * N_PLAYERS,
        baseline=[0.0] * N_PLAYERS,
    )
    assert jnp.allclose(jax_game(all_coalitions()), reference, atol=1e-6)
    assert jnp.allclose(list_game(all_coalitions()), reference, atol=1e-6)
    # tree structure itself accepts non-NumPy arrays and normalizes to host
    jax_tree = TreeModel(
        children_left=jnp.asarray([1, -1, -1]),
        children_right=jnp.asarray([2, -1, -1]),
        features=jnp.asarray([0, -2, -2]),
        thresholds=jnp.asarray([0.5, np.nan, np.nan]),
        values=jnp.asarray([0.0, 1.0, 4.0]),
    )
    assert isinstance(jax_tree.thresholds, np.ndarray)
    assert jax_tree.thresholds.dtype == np.float64
    game = InterventionalTreeGame(jax_tree, inputs=INPUTS, baseline=BASELINE)
    ends = game(DenseCoalitionArray(jnp.asarray([[False] * N_PLAYERS, [True] * N_PLAYERS])))
    assert jnp.allclose(ends, jnp.asarray([1.0, 4.0]), atol=1e-6)


def test_torch_tensors_construct_trees_and_games():
    torch = pytest.importorskip("torch")
    tree = TreeModel(
        children_left=torch.tensor([1, -1, -1]),
        children_right=torch.tensor([2, -1, -1]),
        features=torch.tensor([0, -2, -2]),
        thresholds=torch.tensor([0.5, torch.nan, torch.nan]),
        values=torch.tensor([0.0, 1.0, 4.0]),
    )
    assert isinstance(tree.values, np.ndarray)
    game = InterventionalTreeGame(
        tree,
        inputs=torch.ones(N_PLAYERS),
        baseline=torch.zeros(N_PLAYERS),
    )
    ends = game(DenseCoalitionArray(jnp.asarray([[False] * N_PLAYERS, [True] * N_PLAYERS])))
    assert jnp.allclose(ends, jnp.asarray([1.0, 4.0]), atol=1e-6)
    explanation = TreeExplainer(game, SV()).explain()
    assert jnp.allclose(explanation((0,)), 3.0, atol=1e-6)


def test_outputs_follow_the_default_float_dtype():
    # under x64 the game and the explanation come out in float64; the closed
    # forms always run in float64 on the host, only the re-entry dtype moves
    with jax.enable_x64():
        game = tree_game()
        values = game(all_coalitions())
        assert values.dtype == jnp.float64
        explanation = TreeExplainer(game, SII(order=2)).explain()
        assert explanation((0,)).dtype == jnp.float64
        assert explanation.baseline.dtype == jnp.float64
        assert explanation((1, 2)).dtype == jnp.float64  # the zero default too
    default_game = tree_game()
    assert default_game(all_coalitions()).dtype == jnp.float32
    default_explanation = TreeExplainer(default_game, SII(order=2)).explain()
    assert default_explanation((0,)).dtype == jnp.float32


@pytest.mark.filterwarnings("ignore::shapiq.errors.SamplingStallWarning")
def test_sampling_explainers_consume_the_tree_game():
    game = tree_game()
    approximator = Regression(game, SV(), random_state=0, deduplicate=True)
    estimate = approximator.estimate(2 + 12)
    closed_form = TreeExplainer(game, SV()).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(estimate[(player,)], closed_form((player,)), atol=1e-4)
