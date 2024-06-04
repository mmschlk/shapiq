"""This test module contains all tests regarding the InteractionValues dataclass."""

import os
from copy import copy, deepcopy

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset


@pytest.mark.parametrize(
    "index, n, min_order, max_order, estimation_budget, estimated",
    [
        ("STII", 5, 1, 2, 100, True),
        ("FSII", 5, 1, 2, 100, True),
        ("k-SII", 5, 1, 2, 100, True),
        ("SII", 5, 1, 2, 100, False),
        ("something", 5, 1, 2, 100, False),  # expected to fail with ValueError
    ],
)
def test_initialization(index, n, min_order, max_order, estimation_budget, estimated):
    """Tests the initialization of the InteractionValues dataclass."""
    interaction_lookup = {interaction: i for i, interaction in enumerate(powerset(range(n), 1, 2))}
    values = np.random.rand(len(interaction_lookup))
    baseline_value = 2.0
    if index == "something":
        with pytest.warns(UserWarning):
            interaction_values = InteractionValues(
                values=values,
                index=index,
                n_players=n,
                min_order=min_order,
                max_order=max_order,
                interaction_lookup=interaction_lookup,
                estimation_budget=estimation_budget,
                estimated=estimated,
                baseline_value=baseline_value,
            )
    else:
        interaction_values = InteractionValues(
            values=values,
            index=index,
            n_players=n,
            min_order=min_order,
            max_order=max_order,
            interaction_lookup=interaction_lookup,
            estimation_budget=estimation_budget,
            estimated=estimated,
            baseline_value=baseline_value,
        )
    assert interaction_values.index == index
    assert interaction_values.n_players == n
    assert interaction_values.min_order == min_order
    assert interaction_values.max_order == max_order
    assert np.all(interaction_values.values == values)
    assert interaction_values.estimation_budget == estimation_budget
    assert interaction_values.estimated == estimated
    assert interaction_values.interaction_lookup == interaction_lookup

    # test dict_values property
    assert interaction_values.dict_values == {
        interaction: value for interaction, value in zip(interaction_lookup, values)
    }

    # check that default values are set correctly
    interaction_values_2 = InteractionValues(
        values=np.random.rand(len(interaction_lookup)),
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        baseline_value=baseline_value,
    )
    assert interaction_values_2.estimation_budget is None  # default value is None
    assert interaction_values_2.estimated is True  # default value is True
    assert interaction_values_2.interaction_lookup == interaction_lookup  # automatically generated

    # check the string representations (not semantics)
    assert isinstance(str(interaction_values), str)
    assert isinstance(repr(interaction_values), str)
    assert repr(interaction_values) != str(interaction_values)

    # check equality
    interaction_values_copy = copy(interaction_values)
    interaction_values_deepcopy = deepcopy(interaction_values)
    assert interaction_values == interaction_values_copy
    assert interaction_values == interaction_values_deepcopy
    assert interaction_values != interaction_values_2

    try:
        assert interaction_values == 1  # expected to fail with TypeError
    except TypeError:
        pass

    # check that the hash is correct
    assert hash(interaction_values) == hash(interaction_values_copy)
    assert hash(interaction_values) == hash(interaction_values_deepcopy)
    assert hash(interaction_values) != hash(interaction_values_2)

    # check getitem
    assert interaction_values[(0,)] == interaction_values.values[0]
    assert interaction_values[(1,)] == interaction_values.values[1]
    assert interaction_values[(0, 1)] == interaction_values.values[n]  # first 2nd order is at n
    assert interaction_values[(1, 0)] == interaction_values.values[n]  # order does not matter

    # check getitem with invalid interaction (not in interaction_lookup)
    assert interaction_values[(100, 101)] == 0  # invalid interaction is 0

    # test getitem with integer as input
    assert interaction_values[0] == interaction_values.values[0]
    assert interaction_values[-1] == interaction_values.values[-1]

    # test __len__
    assert len(interaction_values) == len(interaction_values.values)

    # test baseline value
    assert interaction_values.baseline_value == baseline_value
    # expected behavior of interactions is 0 for emptyset
    assert interaction_values[()] == 0


def test_add():
    """Tests the __add__ method of the InteractionValues dataclass."""
    index = "SII"
    n = 5
    min_order = 1
    max_order = 2
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = np.random.rand(len(interaction_lookup))
    values_copy = deepcopy(values)
    interaction_values_first = InteractionValues(
        values=values,
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # test adding scalar values
    interaction_values_added = interaction_values_first + 1
    assert np.all(interaction_values_added.values == interaction_values_first.values + 1)
    interaction_values_added = 1 + interaction_values_first
    assert np.all(interaction_values_added.values == interaction_values_first.values + 1)
    interaction_values_added = interaction_values_first + 1.0
    assert np.all(interaction_values_added.values == interaction_values_first.values + 1.0)

    # test adding InteractionValues (without modifying the original)
    interaction_values_added = interaction_values_first + interaction_values_first
    assert np.all(interaction_values_added.values == 2 * interaction_values_first.values)
    assert np.all(interaction_values_first.values == values_copy)  # original is not modified

    # test adding InteractionValues with different indices
    interaction_values_second = InteractionValues(
        values=values,
        index="STII",
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )
    with pytest.raises(ValueError):
        interaction_values_first + interaction_values_second

    # test adding InteractionValues with different interactions
    n_players_second = n + 1
    interaction_lookup_second = {
        interaction: i
        for i, interaction in enumerate(powerset(range(n_players_second), min_order, max_order))
    }
    values_second = np.random.rand(len(interaction_lookup_second))
    interaction_values_second = InteractionValues(
        values=values_second,
        index=index,
        n_players=n_players_second,
        min_order=min_order,
        max_order=max_order + 1,
        interaction_lookup=interaction_lookup_second,
        baseline_value=0.0,
    )

    # test adding InteractionValues with different interactions
    interaction_values_added = interaction_values_first + interaction_values_second
    assert interaction_values_added.n_players == n + 1  # is the maximum of the two
    assert interaction_values_added.min_order == min_order
    assert interaction_values_added.max_order == max_order + 1  # is the maximum of the two
    # check weather interactions present in both InteractionValues are added
    assert (
        interaction_values_added[(0,)]
        == interaction_values_first[(0,)] + interaction_values_second[(0,)]
    )
    # check weather the interactions that were not present in the first InteractionValues are added
    assert interaction_values_added[(5,)] == interaction_values_second[(5,)]

    # raise TypeError
    with pytest.raises(TypeError):
        interaction_values_first + "string"


def test_sub():
    """Tests the __sub__ method of the InteractionValues dataclass."""
    index = "SII"
    n = 5
    min_order = 1
    max_order = 2
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = np.random.rand(len(interaction_lookup))
    values_copy = deepcopy(values)
    interaction_values_first = InteractionValues(
        values=values,
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # test subtracting scalar values
    interaction_values_sub = interaction_values_first - 1
    assert np.all(interaction_values_sub.values == interaction_values_first.values - 1)
    interaction_values_sub = 1 - interaction_values_first
    assert np.all(interaction_values_sub.values == 1 - interaction_values_first.values)

    # test subtracting InteractionValues (without modifying the original)
    interaction_values_sub = interaction_values_first - interaction_values_first
    assert np.all(interaction_values_sub.values == 0)
    assert np.all(interaction_values_first.values == values_copy)  # original is not modified


def test_mul():
    """Tests the __mul__ method of the InteractionValues dataclass."""
    index = "SII"
    n = 5
    min_order = 1
    max_order = 2
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = np.random.rand(len(interaction_lookup))
    interaction_values_first = InteractionValues(
        values=values,
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # test adding scalar values
    interaction_values_mul = interaction_values_first * 2
    assert np.all(interaction_values_mul.values == 2 * interaction_values_first.values)
    interaction_values_mul = 2 * interaction_values_first
    assert np.all(interaction_values_mul.values == 2 * interaction_values_first.values)
    interaction_values_mul = interaction_values_first * 2.0
    assert np.all(interaction_values_mul.values == 2.0 * interaction_values_first.values)


def test_sum():
    """Tests the sum method of the InteractionValues dataclass."""
    index = "SII"
    n = 5
    min_order = 1
    max_order = 2
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = np.random.rand(len(interaction_lookup))
    interaction_values = InteractionValues(
        values=values,
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    assert np.isclose(sum(interaction_values), np.sum(interaction_values.values))


def test_n_order_transform():
    """Tests the n_order_transform method of the InteractionValues dataclass."""
    index = "SII"
    n = 5
    min_order = 1
    max_order = 3
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = np.random.rand(len(interaction_lookup))
    interaction_values = InteractionValues(
        values=values,
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # test n_order_transform order 1
    interaction_values_transformed = interaction_values.get_n_order_values(1)
    assert interaction_values_transformed.shape == (n,)
    assert interaction_values_transformed[3] == interaction_values[(3,)]

    # test n_order_transform order 2
    interaction_values_transformed = interaction_values.get_n_order_values(2)
    assert interaction_values_transformed.shape == (n, n)
    assert interaction_values_transformed[3, 4] == interaction_values[(3, 4)]

    # test n_order_transform order 3
    interaction_values_transformed = interaction_values.get_n_order_values(3)
    assert interaction_values_transformed.shape == (n, n, n)
    assert interaction_values_transformed[0, 3, 4] == interaction_values[(0, 3, 4)]
    assert interaction_values_transformed[4, 3, 0] == interaction_values[(0, 3, 4)]
    assert interaction_values_transformed[0, 4, 3] == interaction_values[(0, 3, 4)]
    assert interaction_values_transformed[4, 0, 3] == interaction_values[(0, 3, 4)]

    with pytest.raises(ValueError):
        _ = interaction_values.get_n_order_values(0)


def test_sparsify():
    """Tests the sparsify function of the InteractionValues dataclass."""

    # parameters
    values = np.array([1, 1e-1, 1e-3, 1e-4, 1, 1e-4, 1])
    n_players = 7
    interaction_lookup = {(i,): i for i in range(len(values))}
    original_length = len(values)

    # create InteractionValues object
    interaction_values = InteractionValues(
        values=values,
        index="SV",
        n_players=n_players,
        min_order=1,
        max_order=1,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # test before sparsify
    assert len(interaction_values.values) == original_length
    assert np.all(interaction_values.values == values)
    assert interaction_values[(3,)] == values[3]  # will be removed
    assert interaction_values[(4,)] == values[4]
    assert interaction_values[(5,)] == values[5]  # will be removed
    assert interaction_values[(6,)] == values[6]
    assert (3,) in interaction_values.interaction_lookup
    assert (5,) in interaction_values.interaction_lookup

    # sparsify
    threshold = 1e-3
    interaction_values.sparsify(threshold=threshold)

    # test after sparsify
    assert len(interaction_values.values) == original_length - 2  # two are removed
    assert interaction_values[(3,)] != values[3]  # removed
    assert interaction_values[(5,)] != values[5]  # removed
    assert interaction_values[(3,)] == 0  # removed
    assert interaction_values[(5,)] == 0  # removed
    assert interaction_values[(4,)] == values[4]  # not removed
    assert interaction_values[(6,)] == values[6]  # not removed
    assert (3,) not in interaction_values.interaction_lookup  # removed
    assert (5,) not in interaction_values.interaction_lookup  # removed

    # sparsify again
    threshold = 0.9
    interaction_values.sparsify(threshold=threshold)

    # test after sparsify
    assert len(interaction_values.values) == 3
    assert interaction_values[(4,)] == values[4]  # not removed
    assert interaction_values[(6,)] == values[6]  # not removed
    assert interaction_values[(0,)] == values[0]  # not removed


def test_top_k():
    """Tests the top-k selection of the InteractionValues dataclass."""

    # parameters
    values = np.array([1, 2, 3, 4, 5, 6, 8, 7, 9, 10])
    n_players = 10
    interaction_lookup = {(i,): i for i in range(len(values))}
    original_length = len(values)

    # create InteractionValues object
    interaction_values = InteractionValues(
        values=values,
        index="SV",
        n_players=n_players,
        min_order=1,
        max_order=1,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # test before top-k
    assert len(interaction_values.values) == original_length
    assert np.all(interaction_values.values == values)
    for i in range(n_players):
        assert interaction_values[(i,)] == values[i]

    print(interaction_values)

    # top-k
    k = 3
    top_k_interaction, sorted_top_k_interactions = interaction_values.get_top_k(
        k=k, as_interaction_values=False
    )

    assert len(top_k_interaction) == len(sorted_top_k_interactions) == k
    assert sorted_top_k_interactions[0] == ((9,), 10)
    assert sorted_top_k_interactions[1] == ((8,), 9)
    assert sorted_top_k_interactions[2] == ((6,), 8)

    assert (9,) in top_k_interaction
    assert (8,) in top_k_interaction
    assert (6,) in top_k_interaction

    # test with k > len(values)
    k = 20
    top_k_interaction, sorted_top_k_interactions = interaction_values.get_top_k(
        k=k, as_interaction_values=False
    )
    assert len(top_k_interaction) == len(sorted_top_k_interactions) == original_length

    # test with k = 0
    k = 0
    top_k_interaction, sorted_top_k_interactions = interaction_values.get_top_k(
        k=k, as_interaction_values=False
    )
    assert len(top_k_interaction) == len(sorted_top_k_interactions) == 0


def test_from_dict():
    """Tests the from_dict method of the InteractionValues dataclass."""

    # parameters
    values = np.array([1, 2, 3, 4, 5, 6, 8, 7, 9, 10])
    n_players = 10
    interaction_lookup = {(i,): i for i in range(len(values))}

    # create InteractionValues object
    interaction_values = InteractionValues(
        values=values,
        index="SV",
        n_players=n_players,
        min_order=1,
        max_order=1,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # create dict
    interaction_values_dict = interaction_values.to_dict()
    assert np.equal(interaction_values_dict["values"], values).all()
    assert interaction_values_dict["index"] == "SV"
    assert interaction_values_dict["n_players"] == n_players
    assert interaction_values_dict["min_order"] == 1
    assert interaction_values_dict["max_order"] == 1
    assert interaction_values_dict["interaction_lookup"] == interaction_lookup
    assert interaction_values_dict["baseline_value"] == 0.0

    # create InteractionValues object from dict
    interaction_values_from_dict = InteractionValues.from_dict(interaction_values_dict)

    assert interaction_values_from_dict == interaction_values


@pytest.mark.parametrize("as_pickle", [True, False])
def test_save_and_load(as_pickle):
    """Tests the save and load functions of the InteractionValues dataclass."""

    # parameters
    values = np.array([1, 2, 3, 4, 5, 6, 8, 7, 9, 10])
    n_players = 10
    interaction_lookup = {(i,): i for i in range(len(values))}
    original_length = len(values)

    # create InteractionValues object
    interaction_values = InteractionValues(
        values=values,
        index="SV",
        n_players=n_players,
        min_order=1,
        max_order=1,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    # save and load
    path = "test_interaction_values"
    if not as_pickle:
        path += ".npz"
    interaction_values.save(path, as_pickle=as_pickle)

    # see if file exists
    assert os.path.exists(path)

    # test cls load
    loaded_interaction_values = InteractionValues.load(path)
    assert len(loaded_interaction_values.values) == original_length
    assert np.all(loaded_interaction_values.values == values)
    for i in range(n_players):
        assert loaded_interaction_values[(i,)] == values[i]

    # test function load
    loaded_interaction_values = InteractionValues.load_interaction_values(path)
    assert len(loaded_interaction_values.values) == original_length
    assert np.all(loaded_interaction_values.values == values)
    for i in range(n_players):
        assert loaded_interaction_values[(i,)] == values[i]

    # remove file
    os.remove(path)

    # test if file is removed
    assert not os.path.exists(path)


def test_plot():
    """Tests the plot methods in InteractionValues."""
    n = 5
    min_order = 1
    max_order = 2
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = np.random.rand(len(interaction_lookup))
    interaction_values = InteractionValues(
        values=values,
        index="SII",
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    _ = interaction_values.plot_network()
    _ = interaction_values.plot_network(feature_names=["a" for _ in range(n)])
    _ = interaction_values.plot_stacked_bar()
    _ = interaction_values.plot_stacked_bar(feature_names=["a" for _ in range(n)])

    n = 5
    min_order = 1
    max_order = 1
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = np.random.rand(len(interaction_lookup))
    interaction_values = InteractionValues(
        values=values,
        index="SII",
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )
    with pytest.raises(ValueError):
        _ = interaction_values.plot_network()
    with pytest.raises(ValueError):
        _ = interaction_values.plot_network(feature_names=["a" for _ in range(n)])
    _ = interaction_values.plot_stacked_bar()
    _ = interaction_values.plot_stacked_bar(feature_names=["a" for _ in range(n)])
