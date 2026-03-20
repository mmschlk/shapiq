"""Fixtures for the tests."""

from __future__ import annotations

import pytest
from shapiq.datasets import load_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from shapiq_benchmark.typing import TabularDataSet

TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED = 42


@pytest.fixture(scope="session")
def california_housing() -> TabularDataSet:
    """Returns the California Housing dataset split into training and testing sets."""
    x_data, y_data = load_california_housing(to_numpy=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
    )
    return TabularDataSet(x_train, y_train, x_test, y_test)


@pytest.fixture(scope="session")
def california_housing_dt(california_housing: TabularDataSet) -> DecisionTreeRegressor:
    """Returns a DecisionTreeRegressor fitted on the California Housing dataset."""
    x_train = california_housing.x_train
    y_train = california_housing.y_train
    model = DecisionTreeRegressor(max_depth=6, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    return model
