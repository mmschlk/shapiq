"""This test module contains all fixtures for all tests of shapiq."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import InteractionValues

from .fixtures.models import (
    TABULAR_MODEL_FIXTURES,
    TABULAR_TENSORFLOW_MODEL_FIXTURES,
    TABULAR_TORCH_MODEL_FIXTURES,
    TREE_MODEL_FIXTURES,
)

ALL_MODEL_FIXTURES = (
    TABULAR_MODEL_FIXTURES
    + TREE_MODEL_FIXTURES
    + TABULAR_TENSORFLOW_MODEL_FIXTURES
    + TABULAR_TORCH_MODEL_FIXTURES
)


pytest_plugins = [
    "tests.fixtures.games",
    "tests.fixtures.models",
    "tests.fixtures.data",
    "tests.fixtures.interaction_values",
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
    for _ in range(n_objects):
        values = []
        interaction_lookup = {}
        for i, interaction in enumerate(
            powerset(range(n_players), min_size=min_order, max_size=max_order),
        ):
            interaction_lookup[interaction] = i
            values.append(rng.uniform(0, 1))
        values = np.array(values)
        iv = InteractionValues(
            n_players=n_players,
            values=values,
            baseline_value=float(values[interaction_lookup[()]]),
            index="Moebius",
            interaction_lookup=interaction_lookup,
            max_order=max_order,
            min_order=min_order,
        )
        iv_list.append(iv)
    return iv_list


@pytest.fixture(scope="module")
def example_values():
    return InteractionValues(
        n_players=4,
        values=np.array(
            [
                0.0,  # ()
                -0.2,  # (1)
                0.4,  # (2)
                0.2,  # (3)
                -0.1,  # (4)
                0.2,  # (1, 2)
                -0.2,  # (1, 3)
                0.2,  # (1, 4)
                0.2,  # (2, 3)
                -0.2,  # (2, 4)
                0.2,  # (3, 4)
                0.4,  # (1, 2, 3)
                0.0,  # (1, 2, 4)
                0.0,  # (1, 3, 4)
                0.0,  # (2, 3, 4)
                -0.1,  # (1, 2, 3, 4)
            ],
            dtype=float,
        ),
        index="k-SII",
        interaction_lookup={
            (): 0,
            (1,): 1,
            (2,): 2,
            (3,): 3,
            (4,): 4,
            (1, 2): 5,
            (1, 3): 6,
            (1, 4): 7,
            (2, 3): 8,
            (2, 4): 9,
            (3, 4): 10,
            (1, 2, 3): 11,
            (1, 2, 4): 12,
            (1, 3, 4): 13,
            (2, 3, 4): 14,
            (1, 2, 3, 4): 15,
        },
        baseline_value=0,
        min_order=0,
        max_order=4,
    )
