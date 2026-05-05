"""Local XAI benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Protocol, cast

import numpy as np
from shapiq.typing import IndexType, Model
from shapiq_games.benchmark.local_xai.base import LocalExplanation

from .base import Benchmark
from .computers import LocalXAIComputer
from .setup import load_from_str, infer_data_type

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues


class _PredictProbaModel(Protocol):
    def predict_proba(self, x_data: np.ndarray) -> np.ndarray: ...


class _PredictModel(Protocol):
    def predict(self, x_data: np.ndarray) -> np.ndarray: ...


def _build_predict_fn(
    model: object, data_type: str, class_label: int | None
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a prediction function for the given model and data type."""
    if data_type == "classification" and hasattr(model, "predict_proba"):
        model_proba = cast(_PredictProbaModel, model)
        label = 0 if class_label is None else class_label

        def _predict(x_data: np.ndarray) -> np.ndarray:
            return model_proba.predict_proba(x_data)[:, label]

        return _predict

    model_predict = cast(_PredictModel, model)

    def _predict(x_data: np.ndarray) -> np.ndarray:
        return model_predict.predict(x_data)

    return _predict


class LocalXAIBench(Benchmark[IndexType]):
    """Benchmark for local explanation games."""

    def __init__(
        self,
        data: str | np.ndarray,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        *,
        x_explain: int | None = 0,
        class_index: int | None = 1,
        random_state: int | None = 42,
        imputer: str = "marginal",
    ) -> None:
        """Initialize the benchmark by loading data and model and fitting the model.

        Args:
            data: Dataset identifier (e.g. "adult_census") or a NumPy array containing the data.
            model: Model identifier (e.g. "decision_tree") or a fitted model object.
            x_explain: Instance to explain.
            class_index: Class index for classification models.
            random_state: Random state used for data split and model init.
            imputer: Imputation method to use in the LocalExplanation game.
        """
        if isinstance(data, str) and isinstance(model, str):
            self.dataset, self.model = load_from_str(
                data, model, benchmark_type="local_xai", random_state=random_state
            )
            self.x_train: np.ndarray = np.asarray(self.dataset.x_train)
        elif isinstance(data, np.ndarray) and not isinstance(model, str):
            self.dataset = None
            self.x_train = np.asarray(data)
            self.model = model
        else:
            raise ValueError(
                "Invalid combination of data and model arguments. Please provide either both as strings or both as objects."
            )

        data_type: Literal["classification", "regression"] | None = None
        if self.dataset:
            data_type = self.dataset.data_type
        if data_type is None:
            data_type = infer_data_type(self.model)

        if data_type == "regression":
            class_index = None

        predict_fn = _build_predict_fn(self.model, data_type, class_index)
        x_index = 0 if x_explain is None else x_explain
        if not isinstance(x_index, int):
            msg = "x_explain must be an int index."
            raise ValueError(msg)
        self._game = LocalExplanation(
            data=self.x_train,
            model=predict_fn,
            x=self.x_train[x_index],
            imputer=imputer,
            random_state=random_state,
            verbose=False,
        )
        self._computer = LocalXAIComputer(self._game)

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the LocalXAIBench computer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional budget for computation.
        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.exact_values(index=index, order=order, budget=budget)

    @property
    def game(self) -> LocalExplanation:
        """Game instance used by the LocalXAIBench."""
        return self._game

    @property
    def computer(self) -> LocalXAIComputer[IndexType]:
        """Ground truth computer used by the LocalXAIBench."""
        return self._computer
