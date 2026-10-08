"""Tabular global explanation games: how much of a model's behavior a set of features explains."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions, resolve_predict_function

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Loss, LossName

__all__ = ["TabularGlobalExplanation"]

_LOSSES: dict[LossName, Loss] = {
    "mse": lambda reference, prediction: float(np.mean((reference - prediction) ** 2)),
    "mae": lambda reference, prediction: float(np.mean(np.abs(reference - prediction))),
}


class TabularGlobalExplanation(Game):
    r"""The tabular global explanation game (SAGE-like): the loss a coalition of features explains.

    For evaluation rows :math:`X` and replacement rows :math:`R` (each feature independently
    permuted, i.e. drawn from the product of the marginals), let :math:`X_S` keep the features in
    :math:`S` from :math:`X` and take the others from :math:`R`. The value of a coalition is the
    negative loss between the model's predictions on the full and on the masked rows:

    .. math::
        v(S) = -\ell\big(f(X), f(X_S)\big)

    Hence :math:`v(N) = 0`, and the normalized game :math:`v(S) - v(\emptyset)` is the loss
    reduction explained by :math:`S` (higher is better). The evaluation and replacement rows are
    drawn once at construction, so the game is deterministic.

    Attributes:
        class_index: The explained class for classifiers, ``None`` for regressors or callables.
        x_eval: The evaluation rows.
        x_replacement: The replacement rows.
        empty_value: The value of the empty coalition before centering (all features replaced).

    Examples:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
        >>> game = TabularGlobalExplanation(model, data=X[:100], loss="mse")
        >>> game.n_players
        5
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: np.ndarray,
        *,
        class_index: int | None = None,
        loss: LossName | Loss = "mse",
        n_samples: int = 100,
        random_state: int = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the global explanation game.

        Args:
            model: A fitted scikit-learn compatible model, or a callable mapping a
                ``(n_samples, n_features)`` matrix to ``(n_samples,)`` predictions.
            data: The data the rows are drawn from, of shape ``(n_samples, n_features)``.
            class_index: The explained class for classifiers (its probability is compared).
                Defaults to ``None``, which means class ``1`` for classifiers.
            loss: ``"mse"``, ``"mae"``, or a callable ``loss(reference, prediction) -> float``.
                Defaults to ``"mse"``.
            n_samples: The number of evaluation rows. Defaults to ``100``.
            random_state: The seed of the evaluation and replacement rows. Defaults to ``42``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        data = np.asarray(data)
        self._predict, self.class_index = resolve_predict_function(model, class_index)
        if callable(loss):
            self._loss = loss
        elif loss in _LOSSES:
            self._loss = _LOSSES[loss]
        else:
            msg = f"Unknown loss {loss!r}. Choose 'mse', 'mae', or pass a callable."
            raise ValueError(msg)

        rng = np.random.default_rng(random_state)
        n_rows = min(n_samples, data.shape[0])
        self.x_eval = data[rng.choice(data.shape[0], size=n_rows, replace=False)]
        self.x_replacement = np.empty_like(self.x_eval)
        for feature in range(data.shape[1]):
            rows = rng.choice(data.shape[0], size=n_rows, replace=False)
            self.x_replacement[:, feature] = data[rows, feature]
        self._reference_predictions = np.asarray(self._predict(self.x_eval), dtype=float)

        self.empty_value = float(self.value_function(np.zeros((1, data.shape[1]), dtype=bool))[0])
        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the negative loss between full and masked predictions."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            masked = np.where(coalition, self.x_eval, self.x_replacement)
            predictions = np.asarray(self._predict(masked), dtype=float)
            values[i] = -self._loss(self._reference_predictions, predictions)
        return values
