"""The interventional game of a tree model (the game explained by interventional TreeSHAP)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions, resolve_class_index

from ._output import model_output

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues

__all__ = ["InterventionalTreeGame"]


class InterventionalTreeGame(Game):
    r"""The interventional (marginal) game of a model over a reference dataset.

    The value of a coalition :math:`S` is the mean model output over the reference rows
    :math:`z`, where the features in :math:`S` are replaced by the explained point :math:`x`:

    .. math::
        v(S) = \frac{1}{|Z|} \sum_{z \in Z} f(x_S, z_{\bar S})

    For tree models, this is the game that interventional TreeSHAP-IQ explains, so its exact values
    are available through :class:`~shapiq.tree.interventional.InterventionalTreeSHAPIQ`. The game
    evaluates the model's own predictions in the output space of that explainer: raw margins
    (log-odds) for gradient boosting classifiers (scikit-learn, XGBoost, LightGBM, CatBoost), class
    probabilities for other classifiers, and predictions for regressors. Class ``0`` of a binary
    gradient boosting classifier is the negated margin of class ``1``. The game is not normalized
    by default.

    Attributes:
        model: The model.
        reference_data: The reference (background) rows.
        x: The explained point.
        class_index: The explained class for classifiers, ``None`` for regressors.

    Examples:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
        >>> game = InterventionalTreeGame(model, reference_data=X[:50], x=X[0])
        >>> game.n_players
        5
        >>> bool(np.isclose(game(game.grand_coalition)[0], model.predict(X[:1])[0]))
        True
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        reference_data: np.ndarray,
        x: np.ndarray,
        *,
        class_index: int | None = None,
        normalize: bool = False,
    ) -> None:
        """Initialize the interventional game.

        Args:
            model: The fitted model.
            reference_data: The reference rows of shape ``(n_reference, n_features)``.
            x: The explained point of shape ``(n_features,)``.
            class_index: The explained class for classifiers. Defaults to ``None``, which means
                class ``1`` for classifiers (the convention of the shapiq explainers).
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``False``.
        """
        self.model = model
        self.reference_data = np.asarray(reference_data)
        self.x = np.asarray(x).reshape(-1)
        self.class_index = resolve_class_index(model, class_index)
        n_players = self.x.shape[0]
        empty_value = float(self._evaluate(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(n_players, normalize=normalize, normalization_value=empty_value)

    def _evaluate(self, coalitions: CoalitionMatrix) -> GameValues:
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            data = np.where(coalition, self.x, self.reference_data)
            values[i] = float(np.mean(model_output(self.model, data, self.class_index)))
        return values

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the mean model output over the reference data for the coalitions."""
        return self._evaluate(as_bool_coalitions(coalitions))
