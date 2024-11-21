"""This test module contains all tests for the conditional imputer module of the shapiq package."""

import numpy as np
import pytest

from shapiq.games.imputer import ConditionalImputer


def test_conditional_imputer_init():
    """Test the initialization of the conditional imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    rng = np.random.default_rng(42)
    data = rng.random((100, 3))
    x = rng.random((1, 3))

    imputer = ConditionalImputer(
        model=model,
        data=data,
        x=x,
        sample_size=9,
        random_state=42,
    )
    assert np.array_equal(imputer._x, x)
    assert imputer.sample_size == 9
    assert imputer.random_state == 42
    assert imputer.n_features == 3

    # test raise warning with non generative method
    with pytest.raises(ValueError):
        _ = ConditionalImputer(
            model=model,
            data=data,
            x=x,
            sample_size=9,
            random_state=42,
            method="not_generative",
        )

    # test with conditional sample size higher than 2**n_features
    with pytest.warns(UserWarning):
        imputer = ConditionalImputer(
            model=model,
            data=data,
            x=x,
            sample_size=1,
            conditional_budget=2 ** data.shape[1] + 1,  # budget for warning here
            random_state=42,
            conditional_threshold=0.5,  # increases the conditional samples drawn
        )
        coalitions = np.zeros((1, data.shape[1]), dtype=bool)
        imputer(coalitions)


def test_conditional_imputer_value_function():
    def model(x: np.ndarray) -> np.ndarray:
        return np.asarray([np.random.uniform(0, 1) for _ in range(x.shape[0])])

    data = np.random.rand(100, 3)
    x = np.random.rand(1, 3)

    imputer = ConditionalImputer(
        model=model,
        data=data,
        x=x,
        sample_size=11,
        random_state=42,
    )

    imputed_values = imputer(np.array([[True, False, True], [False, True, False]]))
    assert len(imputed_values) == 2
