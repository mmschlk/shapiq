"""Recontextualized games: one context per coalition, the explained rows restricted alike."""

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.game.discretederivative import subset_patterns
from shapiq.players import FinitePlayers
from shapiq.value import VectorValue
from shapiq.xai import RecontextualizedGame

pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, LogisticRegression

COALITIONS = subset_patterns(3)
RNG = np.random.default_rng(0)
X_TRAIN = RNG.normal(size=(40, 3))
Y_TRAIN = X_TRAIN @ np.array([1.0, -2.0, 0.5]) + X_TRAIN[:, 0] * X_TRAIN[:, 2]
X_EXPLAIN = RNG.normal(size=(2, 3))


def recontextualized(model, x_train, y_train, x_explain):
    """Return the prediction for every coalition and row, the model refit on its features.

    Without features, every row gets the mean target, or a classifier the class frequencies.
    """
    values = []
    for coalition in np.asarray(COALITIONS):
        if coalition.any():
            fitted = model.fit(x_train[:, coalition], y_train)
            predict = getattr(fitted, "predict_proba", fitted.predict)
            values.append(predict(x_explain[:, coalition]))
        elif hasattr(model, "predict_proba"):
            frequencies = (y_train[:, None] == np.unique(y_train)).mean(axis=0)
            values.append(np.tile(frequencies, (len(x_explain), 1)))
        else:
            values.append(np.full(len(x_explain), y_train.mean()))
    return np.stack(values)  # (n_coalitions, n_rows, ...)


def test_regression_values():
    """Is every coalition's value the prediction of the model contextualized on its features?"""
    model = LinearRegression()
    game = RecontextualizedGame(
        model, FinitePlayers(3), X_EXPLAIN, X_TRAIN, (3,), y_background=Y_TRAIN
    )
    values = game(COALITIONS[:, None, :])
    want = recontextualized(LinearRegression(), X_TRAIN, Y_TRAIN, X_EXPLAIN)
    np.testing.assert_allclose(values, want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(values[0], np.mean(Y_TRAIN), rtol=1e-6)  # no features: the mean
    assert not hasattr(model, "coef_")  # the user's model is never fitted


def test_classifier_and_dataframe():
    """Do classifiers give vector values, and DataFrames keep their columns through selection?"""
    pd = pytest.importorskip("pandas")
    columns = ["a", "b", "c"]
    game = RecontextualizedGame(
        LogisticRegression(),
        FinitePlayers(3),
        pd.DataFrame(X_EXPLAIN, columns=columns),
        pd.DataFrame(X_TRAIN, columns=columns),
        (3,),
        y_background=Y_TRAIN > 0,
    )
    values = game[1](COALITIONS)
    assert isinstance(values, VectorValue)
    want = recontextualized(LogisticRegression(), X_TRAIN, Y_TRAIN > 0, X_EXPLAIN)[:, 1]
    np.testing.assert_allclose(values.values, want, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(values.values[0], [np.mean(Y_TRAIN <= 0), np.mean(Y_TRAIN > 0)])
    # Coalitions in any order and with repeats give the values of their rows.
    shuffled = jnp.asarray(COALITIONS)[jnp.array([5, 0, 5, 7])]
    np.testing.assert_allclose(game[1](shuffled).values, want[[5, 0, 5, 7]], rtol=1e-4, atol=1e-6)
