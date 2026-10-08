"""Uncertainty explanation games: which features drive a random forest's predictive uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import entropy

from shapiq_games.local_xai import TabularLocalExplanation

if TYPE_CHECKING:
    from shapiq.typing import FloatVector
    from shapiq_games.typing import ImputerName, Uncertainty

__all__ = ["UncertaintyExplanation"]


def _uncertainty_function(forest: Any, uncertainty: Uncertainty):  # noqa: ANN202, ANN401
    """Return a function computing the entropy-based uncertainty of a random forest classifier."""
    trees = list(forest.estimators_)

    def _predict(x: np.ndarray) -> FloatVector:
        probabilities = np.stack(
            [tree.predict_proba(x) for tree in trees]
        )  # (trees, rows, classes)
        total = entropy(probabilities.mean(axis=0), axis=1, base=2)
        if uncertainty == "total":
            return total
        aleatoric = entropy(probabilities, axis=2, base=2).mean(axis=0)
        return aleatoric if uncertainty == "aleatoric" else total - aleatoric

    return _predict


class UncertaintyExplanation(TabularLocalExplanation):
    """The uncertainty explanation game of a random forest classifier.

    The value of a coalition is the imputed predictive uncertainty of the forest at ``x`` when only
    the features in the coalition are known: the total uncertainty (entropy of the mean class
    probabilities), the aleatoric part (mean entropy of the trees), or the epistemic part (their
    difference). Absent features are imputed as in :class:`~shapiq_games.local_xai.TabularLocalExplanation`.

    Attributes:
        uncertainty: The explained kind of uncertainty.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        >>> game = UncertaintyExplanation(model, data=X[:50], x=X[0], uncertainty="epistemic")
        >>> game.n_players
        5
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: np.ndarray,
        x: int | np.ndarray = 0,
        *,
        uncertainty: Uncertainty = "total",
        imputer: ImputerName = "marginal",
        sample_size: int = 100,
        random_state: int = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the uncertainty explanation game.

        Args:
            model: A fitted ``RandomForestClassifier``.
            data: The background data of shape ``(n_samples, n_features)``.
            x: The explained point, or its index in ``data``. Defaults to ``0``.
            uncertainty: ``"total"``, ``"aleatoric"``, or ``"epistemic"``. Defaults to ``"total"``.
            imputer: ``"marginal"``, ``"conditional"``, or ``"baseline"``.
            sample_size: The number of background rows the marginal imputer averages over.
            random_state: The seed of the imputer. Defaults to ``42``.
            normalize: Whether to center the game. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        from sklearn.ensemble import RandomForestClassifier

        if not isinstance(model, RandomForestClassifier):
            msg = f"Expected a fitted RandomForestClassifier, got {type(model).__name__}."
            raise TypeError(msg)
        if uncertainty not in ("total", "aleatoric", "epistemic"):
            msg = f"uncertainty must be 'total', 'aleatoric', or 'epistemic', got {uncertainty!r}."
            raise ValueError(msg)
        self.uncertainty = uncertainty
        self.forest = model
        super().__init__(
            _uncertainty_function(model, uncertainty),
            data,
            x,
            imputer=imputer,
            sample_size=sample_size,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )
