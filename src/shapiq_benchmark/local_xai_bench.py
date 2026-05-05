"""Local XAI benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, get_args

import numpy as np
from shapiq.typing import IndexType, Model
from shapiq_games.benchmark.local_xai.base import LocalExplanation

from .base import Benchmark
from .computers import LocalXAIComputer
from .setup import load_from_str, infer_data_type
from .bench_types import BenchmarkDataset

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues


def _build_predict_fn(
    model: object, data_type: str, class_label: int | None
) -> Callable[[np.ndarray], np.ndarray]:
    if data_type == "classification" and hasattr(model, "predict_proba"):
        label = 0 if class_label is None else class_label

        def _predict(x_data: np.ndarray) -> np.ndarray:
            return model.predict_proba(x_data)[:, label]  # ty: ignore[call-arg]

        return _predict

    def _predict(x_data: np.ndarray) -> np.ndarray:
        return model.predict(x_data)  # ty: ignore[call-arg]

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
            imputer: Imputer strategy used by the LocalExplanation game.
        """
        if isinstance(data, str) and isinstance(model, str):
            self.dataset, self.model = load_from_str(
                data, model, benchmark_type="local_xai", random_state=random_state
            )
            self.x_train = self.dataset.x_train
        elif isinstance(data, np.ndarray) and not isinstance(model, str):
            self.dataset = None
            self.x_train = data
            self.model = model
        else:
            raise ValueError(
                "Invalid combination of data and model arguments. Please provide either both as strings or both as objects."
            )

        data_type: Literal["classification", "regression"] = None
        if self.dataset:
            data_type = self.dataset.data_type
        if data_type is None:
            data_type = infer_data_type(self.model)

        if data_type == "regression":
            class_index = None

        predict_fn = _build_predict_fn(self.model, data_type, class_index)
        self._game = LocalExplanation(
            data=self.x_train,
            model=predict_fn,
            x=(self.x_train[x_explain]),
            imputer=imputer,
            random_state=random_state,
            verbose=False,
        )
        self._computer = LocalXAIComputer(self._game)

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute exact interaction values using the benchmark computer."""
        return self._computer.exact_values(index=index, order=order)

    @property
    def game(self) -> LocalExplanation:
        """Game instance used by this benchmark."""
        return self._game

    @property
    def computer(self) -> LocalXAIComputer[IndexType]:
        """Ground truth computer used by this benchmark."""
        return self._computer
