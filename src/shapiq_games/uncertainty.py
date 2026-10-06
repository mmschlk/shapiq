"""Uncertainty explanation games: which features drive a random forest's predictive uncertainty."""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
from scipy.stats import entropy

from shapiq_games._base import resolve_x
from shapiq_games._setup import configure
from shapiq_games.local_xai import LocalExplanation

__all__ = ["UncertaintyExplanation"]

type Uncertainty = Literal["total", "aleatoric", "epistemic"]


def _uncertainty_function(forest: Any, uncertainty: Uncertainty):  # noqa: ANN202, ANN401
    """Return a function computing the entropy-based uncertainty of a random forest classifier."""
    trees = list(forest.estimators_)

    def _predict(x: np.ndarray) -> np.ndarray:
        probabilities = np.stack(
            [tree.predict_proba(x) for tree in trees]
        )  # (trees, rows, classes)
        total = entropy(probabilities.mean(axis=0), axis=1, base=2)
        if uncertainty == "total":
            return total
        aleatoric = entropy(probabilities, axis=2, base=2).mean(axis=0)
        return aleatoric if uncertainty == "aleatoric" else total - aleatoric

    return _predict


class UncertaintyExplanation(LocalExplanation):
    """The uncertainty explanation game of a random forest classifier.

    The value of a coalition is the imputed predictive uncertainty of the forest at ``x`` when only
    the features in the coalition are known: the total uncertainty (entropy of the mean class
    probabilities), the aleatoric part (mean entropy of the trees), or the epistemic part (their
    difference). Absent features are imputed as in :class:`~shapiq_games.local_xai.LocalExplanation`.

    Attributes:
        uncertainty: The explained kind of uncertainty.
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: np.ndarray,
        x: int | np.ndarray = 0,
        *,
        uncertainty: Uncertainty = "total",
        imputer: Literal["marginal", "conditional", "baseline"] = "marginal",
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

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        *,
        dataset: str,
        x: int = 0,
        uncertainty: Uncertainty = "total",
        imputer: Literal["marginal", "conditional", "baseline"] = "marginal",
        n_background: int = 100,
        random_state: int = 42,
        test_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
        dataset_params: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> Self:
        """Build the game for a random forest trained on a registered classification dataset.

        Args:
            dataset: The name of a classification dataset.
            x: The index of the explained point in the test split. Defaults to ``0``.
            uncertainty: ``"total"``, ``"aleatoric"``, or ``"epistemic"``.
            imputer: ``"marginal"``, ``"conditional"``, or ``"baseline"``.
            n_background: The number of background rows (from the training split).
            random_state: The seed of the split, the forest, the background, and the imputer.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            model_params: Hyperparameters of the forest.
            dataset_params: Parameters of synthetic datasets.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        setup = configure(
            dataset=dataset,
            model="random_forest",
            random_state=random_state,
            test_size=test_size,
            model_params=model_params,
            dataset_params=dataset_params,
        )
        split = setup.split
        if split.task != "classification":
            msg = f"UncertaintyExplanation needs a classification dataset, got '{dataset}'."
            raise ValueError(msg)
        rng = np.random.default_rng(random_state)
        n_rows = min(n_background, split.x_train.shape[0])
        rows = np.sort(rng.choice(split.x_train.shape[0], size=n_rows, replace=False))
        game = cls(
            setup.model,
            split.x_train[rows],
            resolve_x(x, split.x_test),
            uncertainty=uncertainty,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
        )
        return game._set_config(
            **setup.config,
            x=x,
            uncertainty=uncertainty,
            imputer=imputer,
            n_background=n_background,
            normalize=normalize,
        )
