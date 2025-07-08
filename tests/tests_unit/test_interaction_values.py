"""This test module contains all tests regarding the InteractionValues dataclass."""

from __future__ import annotations

import contextlib
import pathlib
from copy import copy, deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues, aggregate_interaction_values
from shapiq.utils import powerset
from tests.fixtures.interaction_values import (
    get_mock_interaction_value,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("index", "n", "min_order", "max_order", "estimation_budget", "estimated"),
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
    assert interaction_values.dict_values == dict(zip(interaction_lookup, values, strict=False))

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
    assert interaction_values == interaction_values_copy
    assert interaction_values != interaction_values_2

    with contextlib.suppress(TypeError):
        assert interaction_values == 1  # expected to fail with TypeError

    # check that the hash is correct
    assert hash(interaction_values) == hash(interaction_values_copy)
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

    # check setitem
    interaction_values[(0,)] = 999_999
    assert interaction_values[(0,)] == 999_999

    # check setitem with integer as input
    interaction_values[0] = 111_111
    assert interaction_values[0] == 111_111

    # check setitem raises error for invalid interaction
    with pytest.raises(KeyError):
        interaction_values[(100, 101)] = 0

    # test __len__
    assert len(interaction_values) == len(interaction_values.values)

    # test baseline value
    assert interaction_values.baseline_value == baseline_value
    # test baseline value initialization
    with pytest.raises(TypeError):
        InteractionValues(
            values=values,
            index=index,
            n_players=n,
            min_order=min_order,
            max_order=max_order,
            interaction_lookup=interaction_lookup,
            baseline_value="None",
        )
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


def test_abs():
    """Tests the abs method of the InteractionValues dataclass."""
    index = "SII"
    n = 5
    min_order = 1
    max_order = 2
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    values = (-1) * np.random.rand(len(interaction_lookup))
    interaction_values = InteractionValues(
        values=values,
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=0.0,
    )

    assert np.all(abs(interaction_values).values == abs(interaction_values.values))


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

    # top-k
    k = 3
    top_k_interaction, sorted_top_k_interactions = interaction_values.get_top_k(
        k=k,
        as_interaction_values=False,
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
        k=k,
        as_interaction_values=False,
    )
    assert len(top_k_interaction) == len(sorted_top_k_interactions) == original_length

    # test with k = 0
    k = 0
    top_k_interaction, sorted_top_k_interactions = interaction_values.get_top_k(
        k=k,
        as_interaction_values=False,
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

    _ = interaction_values.plot_network(show=False)
    _ = interaction_values.plot_network(show=False, feature_names=["a" for _ in range(n)])
    _ = interaction_values.plot_stacked_bar(show=False)
    _ = interaction_values.plot_stacked_bar(show=False, feature_names=["a" for _ in range(n)])

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
        _ = interaction_values.plot_network(show=False)
    with pytest.raises(ValueError):
        _ = interaction_values.plot_network(show=False, feature_names=["a" for _ in range(n)])
    _ = interaction_values.plot_stacked_bar(show=False)
    _ = interaction_values.plot_stacked_bar(show=False, feature_names=["a" for _ in range(n)])


@pytest.mark.parametrize("subset_players", [[0, 1], [0, 1, 3, 4]])
def test_subset(subset_players):
    """Test Subset function."""
    n = 7
    min_order = 1
    max_order = 3
    values = np.random.rand(2**n - 1)
    interaction_lookup = {
        interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
    }
    interaction_values = InteractionValues(
        values=values,
        index="SII",
        max_order=max_order,
        n_players=n,
        min_order=min_order,
        interaction_lookup=interaction_lookup,
        estimated=False,
        estimation_budget=0,
        baseline_value=0.0,
    )

    n_players_in_subset = len(subset_players)
    subset_interaction_values = interaction_values.get_subset(subset_players)

    assert subset_interaction_values.n_players == n - n_players_in_subset
    assert all(
        all(p in subset_players for p in key)
        for key in subset_interaction_values.interaction_lookup
    )
    assert len(subset_interaction_values.values) == len(
        subset_interaction_values.interaction_lookup,
    )
    assert interaction_values.baseline_value == subset_interaction_values.baseline_value
    assert subset_interaction_values.min_order == interaction_values.min_order
    assert subset_interaction_values.max_order == interaction_values.max_order
    assert subset_interaction_values.estimated == interaction_values.estimated
    assert subset_interaction_values.estimation_budget == interaction_values.estimation_budget
    assert subset_interaction_values.index == interaction_values.index

    # check that all values are correct
    for interaction in powerset(subset_players, max_size=max_order, min_size=min_order):
        old_value = interaction_values[interaction]
        new_value = subset_interaction_values[interaction]
        assert old_value == new_value


@pytest.mark.parametrize("aggregation", ["sum", "mean", "median", "max", "min"])
def test_aggregation(aggregation):
    """Tests the aggregation of InteractionValues."""
    n_objects = 3
    n, min_order, max_order = 5, 1, 3
    interaction_values_list = []
    for _ in range(n_objects):
        values = np.random.rand(2**n - 1)
        interaction_lookup = {
            interaction: i for i, interaction in enumerate(powerset(range(n), min_order, max_order))
        }
        interaction_values = InteractionValues(
            values=values,
            index="SII",
            max_order=max_order,
            n_players=n,
            min_order=min_order,
            interaction_lookup=interaction_lookup,
            estimated=False,
            estimation_budget=0,
            baseline_value=0.0,
        )
        interaction_values_list.append(interaction_values)

    aggregated_interaction_values = aggregate_interaction_values(
        interaction_values_list,
        aggregation=aggregation,
    )

    assert isinstance(aggregated_interaction_values, InteractionValues)
    assert aggregated_interaction_values.index == "SII"
    assert aggregated_interaction_values.n_players == n
    assert aggregated_interaction_values.min_order == min_order
    assert aggregated_interaction_values.max_order == max_order

    # check that all interactions are equal to the expected value
    for interaction in powerset(range(n), 1, n):
        aggregated_value = np.array(
            [interaction_values[interaction] for interaction_values in interaction_values_list],
        )
        if aggregation == "sum":
            expected_value = np.sum(aggregated_value)
        elif aggregation == "mean":
            expected_value = np.mean(aggregated_value)
        elif aggregation == "median":
            expected_value = np.median(aggregated_value)
        elif aggregation == "max":
            expected_value = np.max(aggregated_value)
        elif aggregation == "min":
            expected_value = np.min(aggregated_value)
        assert aggregated_interaction_values[interaction] == expected_value

    # test aggregate from InteractionValues object
    aggregated_from_object = interaction_values_list[0].aggregate(
        aggregation=aggregation,
        others=interaction_values_list[1:],
    )
    assert isinstance(aggregated_from_object, InteractionValues)
    assert aggregated_from_object == aggregated_interaction_values  # same values
    assert aggregated_from_object is not aggregated_interaction_values  # but different objects


def test_docs_aggregation_function():
    """Tests the aggregation function in the InteractionValues dataclass like in the docs."""
    iv1 = InteractionValues(
        values=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        index="SII",
        n_players=3,
        min_order=1,
        max_order=2,
        interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
        baseline_value=0.0,
    )

    # this does not contain the (1, 2) interaction (i.e. is 0)
    iv2 = InteractionValues(
        values=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
        index="SII",
        n_players=3,
        min_order=1,
        max_order=2,
        interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4},
        baseline_value=1.0,
    )

    # test sum
    aggregated_interaction_values = aggregate_interaction_values([iv1, iv2], aggregation="sum")
    assert pytest.approx(aggregated_interaction_values[(0,)]) == 0.3
    assert pytest.approx(aggregated_interaction_values[(1,)]) == 0.5
    assert pytest.approx(aggregated_interaction_values[(1, 2)]) == 0.6
    assert pytest.approx(aggregated_interaction_values.baseline_value) == 1.0

    # test mean
    aggregated_interaction_values = aggregate_interaction_values([iv1, iv2], aggregation="mean")
    assert pytest.approx(aggregated_interaction_values[(0,)]) == 0.15
    assert pytest.approx(aggregated_interaction_values[(1,)]) == 0.25
    assert pytest.approx(aggregated_interaction_values[(1, 2)]) == 0.3
    assert pytest.approx(aggregated_interaction_values.baseline_value) == 0.5

    with pytest.raises(ValueError):
        _ = aggregate_interaction_values([iv1, iv2], aggregation="invalid")


def test_get_n_order_error_all_none(iv_10_all):
    """Tests if `get_n_order` raises ValueError if all parameters are `None`."""
    iv = iv_10_all
    with pytest.raises(ValueError):
        iv.get_n_order()  # all parameters are None


def test_get_n_order_error_min_larger_max(iv_10_all):
    """Tests if `get_n_order` raises ValueError if min_order > max_order."""
    iv = iv_10_all
    with pytest.raises(ValueError):
        iv.get_n_order(min_order=2, max_order=1)  # min > max

    with pytest.raises(ValueError):
        iv.get_n_order(order=3, min_order=3, max_order=2)  # min > max even with order


def test_get_n_order_min_max_overrides_order(iv_10_all):
    """Tests that min_order and max_order overrides order."""
    iv = iv_10_all
    min_order = 0
    max_order = 5
    iv_new = iv.get_n_order(order=2, min_order=min_order, max_order=max_order)
    assert iv_new.min_order == min_order
    assert iv_new.max_order == max_order
    assert all(
        min_order <= len(interaction) <= max_order for interaction in iv_new.interaction_lookup
    )


@pytest.mark.parametrize(("min_order", "max_order"), [(None, 3), (2, None)])
def test_get_n_order_single_bound(min_order, max_order, iv_10_all):
    """Tests behavior when only min or max is provided."""
    iv = iv_10_all
    iv_new = iv.get_n_order(min_order=min_order, max_order=max_order)

    if min_order is None:
        min_order = iv.min_order
    if max_order is None:
        max_order = iv.max_order

    assert iv_new.min_order == min_order
    assert iv_new.max_order == max_order
    assert all(min_order <= len(inter) <= max_order for inter in iv_new.interaction_lookup)


def test_get_n_order_empty_result(iv_10_all):
    """Test that get_n_order returns an empty InteractionValues if no interactions match."""
    iv = iv_10_all
    # Choose min/max such that no interaction can match
    iv_new = iv.get_n_order(min_order=11, max_order=15)
    assert len(iv_new.interaction_lookup) == 0
    assert iv_new.values.size == 0

    # choose order to be larger than max_order
    iv_new = iv.get_n_order(order=11)
    assert len(iv_new.interaction_lookup) == 0
    assert iv_new.values.size == 0


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
def test_get_n_order_with_only_order_param(
    order: int,
    iv_10_all: InteractionValues,
    iv_300_300_0_300: InteractionValues,
):
    """Tests that get_n_order returns only the specified order if only order is given as a parameter."""
    for iv in [iv_10_all, iv_300_300_0_300]:
        iv_new = iv.get_n_order(order=order)
        assert isinstance(iv_new, InteractionValues)
        assert iv_new.min_order == order
        assert iv_new.max_order == order

        # check that the order is correct
        assert all(len(interaction) == order for interaction in iv_new.interaction_lookup)

        # check that all interactions from the original are present
        assert all(
            interaction in iv_new.interaction_lookup
            for interaction in iv.interaction_lookup
            if len(interaction) == order
        )

        # check that all values are correct
        assert all(
            iv_new[interaction] == iv[interaction] for interaction in iv_new.interaction_lookup
        )


@pytest.mark.parametrize(("min_order", "max_order"), [(0, 1), (0, 2), (2, 3), (3, 4), (4, 4)])
def test_get_n_order_with_only_min_max_param(
    min_order: int,
    max_order: int,
    iv_10_all: InteractionValues,
    iv_300_300_0_300: InteractionValues,
):
    """Tests that get_n_order returns the correct interactions when only min and max are given."""
    for iv in [iv_10_all, iv_300_300_0_300]:
        iv_new = iv.get_n_order(min_order=min_order, max_order=max_order)
        assert isinstance(iv_new, InteractionValues)
        assert iv_new.min_order == min_order
        assert iv_new.max_order == max_order

        # check that the order is correct
        assert all(
            min_order <= len(interaction) <= max_order for interaction in iv_new.interaction_lookup
        )

        # check that all interactions from the original are present
        assert all(
            interaction in iv_new.interaction_lookup
            for interaction in iv.interaction_lookup
            if min_order <= len(interaction) <= max_order
        )

        # check that all values are correct
        assert all(
            iv_new[interaction] == iv[interaction] for interaction in iv_new.interaction_lookup
        )


def test_copy_behaviour():
    """Tests that InteractionValues objects are copied correctly."""
    from copy import copy, deepcopy

    # check that copy and deepcopy both work and create a copyied object
    for copy_method in [copy, deepcopy]:
        original = get_mock_interaction_value(n_players=10, n_interactions=20)
        copied = copy_method(original)
        interaction = next(iter(copied.interaction_lookup))  # we use this interaction to test later

        # Structural equality: values, lookup, etc.
        assert isinstance(copied, InteractionValues)
        assert np.array_equal(original.values, copied.values)
        assert original.interaction_lookup == copied.interaction_lookup
        assert original.n_players == copied.n_players
        assert original.min_order == copied.min_order
        assert original.max_order == copied.max_order
        assert original.index == copied.index
        assert original.baseline_value == copied.baseline_value
        assert original.estimated == copied.estimated
        assert original[interaction] == copied[interaction]
        assert hash(original) == hash(copied)
        assert original == copied

        # Independence: changing one doesn't affect the other
        copied.values[0] += 1.0
        assert not np.array_equal(original.values, copied.values), "Values should be independent"
        copied.interaction_lookup[interaction] = 999
        assert original.interaction_lookup != copied.interaction_lookup, "Lookup should be different"  # fmt: skip
        assert original.interaction_lookup[interaction] != copied.interaction_lookup[interaction]
        assert hash(original) != hash(copied)
        assert original != copied, "Objects should be different"


class TestSavingInteractionValues:
    """Tests the saving and loading of InteractionValues."""

    @pytest.mark.parametrize("iv_str", ("iv_7_all", "iv_300_300_0_300"))
    def test_save_and_load_json(self, iv_str: str, tmp_path: Path, request):
        """Tests saving and loading of InteractionValues using a temp path."""
        path = tmp_path / pathlib.Path(f"test_interaction_values_{iv_str}.json")
        iv: InteractionValues = request.getfixturevalue(iv_str)

        iv.save(path)
        assert path.exists()
        loaded_iv = InteractionValues.load(path)
        assert loaded_iv == iv  # check if loaded InteractionValues is equal to original
        loaded_iv_json = InteractionValues.from_json_file(path)
        assert loaded_iv_json == iv

    def test_deprecation_warning_in_save(self, iv_7_all: InteractionValues, tmp_path: Path):
        """Tests that old methods work but also warn with deprecation."""
        path = tmp_path / pathlib.Path("test_interaction_values")
        with pytest.warns(DeprecationWarning):
            iv_7_all.save(path, as_pickle=True)
            iv = InteractionValues.load(path)
            assert iv == iv_7_all

        with pytest.warns(DeprecationWarning):
            iv_7_all.save(path, as_npz=True)
            iv = InteractionValues.load(path.with_suffix(".npz"))
            assert iv == iv_7_all
