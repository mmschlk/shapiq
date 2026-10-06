"""Global feature importance games: how much of a model's behavior a coalition of features explains."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from shapiq.game import Game
from shapiq_games._base import (
    ConfigMixin,
    as_bool_coalitions,
    make_predict_function,
    resolve_class_index,
)
from shapiq_games._setup import configure

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["GlobalExplanation"]

_LOSSES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mse": lambda reference, prediction: float(np.mean((reference - prediction) ** 2)),
    "mae": lambda reference, prediction: float(np.mean(np.abs(reference - prediction))),
}


class GlobalExplanation(ConfigMixin, Game):
    r"""The global explanation game (SAGE-like): the loss a coalition of features explains.

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
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: np.ndarray,
        *,
        class_index: int | None = None,
        loss: Literal["mse", "mae"] | Callable[[np.ndarray, np.ndarray], float] = "mse",
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
        self.class_index: int | None = None
        if callable(model) and not hasattr(model, "predict"):
            self._predict: Callable[[np.ndarray], np.ndarray] = model
        else:
            self.class_index = resolve_class_index(model, class_index)
            self._predict = make_predict_function(model, self.class_index)
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

        empty_value = float(self._evaluate(np.zeros((1, data.shape[1]), dtype=bool))[0])
        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=empty_value,
            verbose=verbose,
        )

    def _evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            masked = np.where(coalition, self.x_eval, self.x_replacement)
            predictions = np.asarray(self._predict(masked), dtype=float)
            values[i] = -self._loss(self._reference_predictions, predictions)
        return values

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the negative loss between full and masked predictions."""
        return self._evaluate(as_bool_coalitions(coalitions))

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        model: str = "random_forest",
        class_index: int | None = None,
        loss: Literal["mse", "mae"] = "mse",
        n_samples: int = 100,
        random_state: int = 42,
        test_size: float = 0.2,
        preset: str | None = None,
        model_params: dict[str, Any] | None = None,
        dataset_params: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> Self:
        """Build the game for a registered dataset; rows are drawn from the test split.

        Args:
            dataset: The dataset name.
            model: The model name. Defaults to ``"random_forest"``.
            class_index: The explained class for classifiers (``None`` means class ``1``).
            loss: ``"mse"`` or ``"mae"``. Defaults to ``"mse"``.
            n_samples: The number of evaluation rows. Defaults to ``100``.
            random_state: The seed of the split, the model, and the rows.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            preset: The hyperparameter preset of the model (``"tuned"`` or ``None``).
            model_params: Hyperparameters of the model.
            dataset_params: Parameters of synthetic datasets.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        setup = configure(
            dataset=dataset,
            model=model,
            random_state=random_state,
            test_size=test_size,
            preset=preset,
            model_params=model_params,
            dataset_params=dataset_params,
        )
        game = cls(
            setup.model,
            setup.split.x_test,
            class_index=class_index,
            loss=loss,
            n_samples=n_samples,
            random_state=random_state,
            normalize=normalize,
        )
        return game._set_config(
            **setup.config,
            class_index=class_index,
            loss=loss,
            n_samples=n_samples,
            normalize=normalize,
        )
