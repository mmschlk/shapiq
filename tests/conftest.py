"""This test module contains all fixtures for all tests of shapiq."""

import numpy as np
import pytest

# Note to developers: If you add a new model fixture, make sure to add it to the

pytest_plugins = [
    "tests.fixtures.games",
    "tests.fixtures.models",
    "tests.fixtures.data",
]


@pytest.fixture
def mae_loss():
    """Returns the mean absolute error loss function."""
    from sklearn.metrics import mean_absolute_error

    return mean_absolute_error


@pytest.fixture
def interaction_values_list():
    """Returns a list of three InteractionValues objects."""
    rng = np.random.RandomState(42)

    from shapiq.interaction_values import InteractionValues
    from shapiq.utils import powerset

    n_objects = 3
    n_players = 5
    min_order = 0
    max_order = n_players
    iv_list = []
    for i in range(n_objects):
        values = []
        interaction_lookup = {}
        for i, interaction in enumerate(
            powerset(range(n_players), min_size=min_order, max_size=max_order)
        ):
            interaction_lookup[interaction] = i
            values.append(rng.uniform(0, 1))
        values = np.array(values)
        iv = InteractionValues(
            n_players=n_players,
            values=values,
            baseline_value=float(values[interaction_lookup[tuple()]]),
            index="Moebius",
            interaction_lookup=interaction_lookup,
            max_order=max_order,
            min_order=min_order,
        )
        iv_list.append(iv)
    return iv_list
