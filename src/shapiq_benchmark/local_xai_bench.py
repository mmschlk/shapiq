"""Local XAI benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

from shapiq.typing import IndexType, Model
from shapiq_games.benchmark.local_xai.base import LocalExplanation

from .base import Benchmark
from .computers import LocalXAIComputer
from .setup import infer_data_type, load_from_str

if TYPE_CHECKING:
    from collections.abc import Callable


class _PredictProbaModel(Protocol):
    def predict_proba(self, x_data: np.ndarray) -> np.ndarray: ...


class _PredictModel(Protocol):
    def predict(self, x_data: np.ndarray) -> np.ndarray: ...


def _build_predict_fn(
    model: object, data_type: str, class_label: int | None
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a prediction function for the given model and data type."""
    if data_type == "classification" and hasattr(model, "predict_proba"):
        model_proba = cast("_PredictProbaModel", model)
        label = 0 if class_label is None else class_label

        def _predict(x_data: np.ndarray) -> np.ndarray:
            return model_proba.predict_proba(x_data)[:, label]

        return _predict

    model_predict = cast("_PredictModel", model)

    def _predict(x_data: np.ndarray) -> np.ndarray:
        return model_predict.predict(x_data)

    return _predict


class LocalXAIBench(Benchmark[IndexType]):
    """Benchmark for local explanation games."""

    def _load_and_set_dataset_and_model(
        self,
        data: str | np.ndarray,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        data_type: Literal["classification", "regression"] | None = None,
        *,
        benchmark_type: str,
        random_state: int | None,
        class_index: int | None,
        **kwargs: object,
    ) -> tuple[int | None, Literal["classification", "regression"]]:
        """Load and set dataset and model based on the provided arguments.

        Args:
            data: Dataset identifier or a NumPy array containing the data.
            model: Model identifier or a fitted model object.
            data_type: Type of data (e.g. "classification", "regression").
            benchmark_type: Type of benchmark to load (e.g. "local_xai").
            random_state: Random state used for data split and model init.
            class_index: Class index for classification models.
            **kwargs: Additional keyword arguments for model building.
        """
        if isinstance(data, str) and isinstance(model, str):
            self.dataset, self.model = load_from_str(
                data,
                model,
                benchmark_type=benchmark_type,
                random_state=random_state,
                **kwargs,
            )
            self.data: np.ndarray = np.asarray(self.dataset.x_test)
        elif isinstance(data, np.ndarray) and not isinstance(model, str):
            if data_type is None:
                msg = "data_type (classification or regression) must be provided when data is a NumPy array."
                raise ValueError(msg)
            self.dataset = None
            self.data = np.asarray(data)
            self.model = model
        else:
            msg = (
                "Invalid combination of data and model arguments. "
                "Please provide either both as strings or both as objects."
            )
            raise TypeError(msg)

        if self.dataset:
            data_type = self.dataset.data_type
        if data_type is None:
            data_type = infer_data_type(self.model)

        if data_type == "regression":
            class_index = None

        return class_index, data_type

    @staticmethod
    def _resolve_x_explain(x_explain: int | None) -> int:
        x_index = 0 if x_explain is None else x_explain
        if not isinstance(x_index, int):
            msg = "x_explain must be an int index."
            raise TypeError(msg)
        return x_index

    def __init__(
        self,
        data: str | np.ndarray,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        data_type: Literal["classification", "regression"] | None = None,
        *,
        x_explain: int | None = 0,
        class_index: int | None = 1,
        random_state: int | None = 42,
        imputer: str = "marginal",
        **kwargs: object,
    ) -> None:
        """Initialize the benchmark by loading data and model and fitting the model.

        Args:
            data: Dataset identifier (e.g. "adult_census") or a NumPy array containing the data.
            model: Model identifier (e.g. "decision_tree") or a fitted model object.
            data_type: Type of data ("classification" or "regression"), or None to infer. Must be provided if data is a NumPy array.
            x_explain: Instance to explain.
            class_index: Class index for classification models.
            random_state: Random state used for data split and model init.
            imputer: Imputation method to use in the LocalExplanation game.
            **kwargs: Additional keyword arguments for model building.
        """
        class_index, data_type = self._load_and_set_dataset_and_model(
            data,
            model,
            benchmark_type="local_xai",
            random_state=random_state,
            class_index=class_index,
            data_type=data_type,
            **kwargs,
        )

        predict_fn = _build_predict_fn(self.model, data_type, class_index)
        x_index = self._resolve_x_explain(x_explain)
        self._game = LocalExplanation(
            data=self.data,
            model=predict_fn,
            x=self.data[x_index],
            imputer=imputer,
            random_state=random_state,
            verbose=False,
        )
        self._computer = LocalXAIComputer(self._game)
