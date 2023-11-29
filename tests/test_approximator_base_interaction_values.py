"""This test module contains all tests regarding the InteractionValues dataclass."""
from copy import copy, deepcopy

import numpy as np
import pytest

from approximator._base import InteractionValues
from utils import powerset


@pytest.mark.parametrize(
    "index, n, min_order, max_order, estimation_budget, estimated",
    [
        ("SII", 5, 1, 2, 100, True),
        ("STI", 5, 1, 2, 100, True),
        ("FSI", 5, 1, 2, 100, True),
        ("nSII", 5, 1, 2, 100, True),
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
