"""This test module contains all tests for the conditional imputer module of the shapiq package."""

import numpy as np
import pytest

from shapiq.games.imputer import ConditionalImputer


def test_conditional_imputer_init():
    """Test the initialization of the conditional imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = np.random.rand(10, 3)
    x = np.random.rand(1, 3)

    imputer = ConditionalImputer(
        model=model,
        data=data,
        x=x,
        sample_size=9,
        random_state=42,
    )
    assert np.array_equal(imputer._x, x)
    assert imputer.sample_size == 9
    assert imputer._random_state == 42
    assert imputer._n_features == 3

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
