"""This test module contains all tests regarding the InteractionValues dataclass."""
from copy import copy, deepcopy

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset


@pytest.mark.parametrize(
    "index, n, min_order, max_order, estimation_budget, estimated",
    [
        ("SII", 5, 1, 2, 100, True),
        ("STI", 5, 1, 2, 100, True),
        ("FSI", 5, 1, 2, 100, True),
        ("k-SII", 5, 1, 2, 100, True),
        ("SII", 5, 1, 2, 100, False),
        ("something", 5, 1, 2, 100, False),  # expected to fail with ValueError
    ],
)
def test_initialization(index, n, min_order, max_order, estimation_budget, estimated):
    """Tests the initialization of the InteractionValues dataclass."""
    interaction_lookup = {interaction: i for i, interaction in enumerate(powerset(range(n), 1, 2))}
    values = np.random.rand(len(interaction_lookup))
    try:
        interaction_values = InteractionValues(
            values=values,
            index=index,
            n_players=n,
            min_order=min_order,
            max_order=max_order,
            interaction_lookup=interaction_lookup,
            estimation_budget=estimation_budget,
            estimated=estimated,
        )
    except ValueError:
        if index == "something":
            return
        raise
    assert interaction_values.index == index
    assert interaction_values.n_players == n
    assert interaction_values.min_order == min_order
    assert interaction_values.max_order == max_order
    assert np.all(interaction_values.values == values)
    assert interaction_values.estimation_budget == estimation_budget
    assert interaction_values.estimated == estimated
    assert interaction_values.interaction_lookup == interaction_lookup

    # check that default values are set correctly
    interaction_values_2 = InteractionValues(
        values=np.random.rand(len(interaction_lookup)),
        index=index,
        n_players=n,
        min_order=min_order,
        max_order=max_order,
    )
    assert interaction_values_2.estimation_budget is None  # default value is None
    assert interaction_values_2.estimated is True  # default value is True
    assert interaction_values_2.interaction_lookup == interaction_lookup  # automatically generated

    # check the string representations (not semantics)
    str(interaction_values)
    repr(interaction_values)

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
        index="STI",
        n_players=n,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
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
    )
    with pytest.warns(UserWarning):
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
    )

    # test adding scalar values
    interaction_values_mul = interaction_values_first * 2
    assert np.all(interaction_values_mul.values == 2 * interaction_values_first.values)
    interaction_values_mul = 2 * interaction_values_first
    assert np.all(interaction_values_mul.values == 2 * interaction_values_first.values)
    interaction_values_mul = interaction_values_first * 2.0
    assert np.all(interaction_values_mul.values == 2.0 * interaction_values_first.values)
