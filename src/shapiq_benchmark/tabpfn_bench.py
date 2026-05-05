"""TabPFN benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, get_args

import numpy as np
from shapiq.explainer.utils import get_predict_function_and_model_type
from shapiq.imputer.tabpfn_imputer import TabPFNImputer
from shapiq.typing import IndexType, Model

from .base import Benchmark
from .bench_types import BenchmarkDataset
from .computers import TabPFNComputer
from .setup import load_from_str

if TYPE_CHECKING:
    from shapiq import InteractionValues


class TabPFNBench(Benchmark[IndexType]):
    """Benchmark for TabPFN-based explanations."""

    def __init__(
        self,
        data: str | tuple[np.ndarray, np.ndarray],
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        *,
        labels: tuple[np.ndarray, np.ndarray] | None = None,
        x_explain: int | None = 0,
        class_index: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        """Initialize the benchmark by loading data and initializing a TabPFN imputer.

        Args:
            data: Dataset identifier or a tuple ``(x_train, x_test)``.
            model: "tabpfn" or a TabPFN model instance (classifier or regressor).
            labels: Labels for ``data`` if ``data`` is a tuple, provided as
                ``(y_train, y_test)``.
            x_explain: Index or vector to explain.
            class_index: Class index for classification models.
            random_state: Random state for train/test split if needed.
        """
        self.dataset: BenchmarkDataset | None = None
        if isinstance(data, str) and isinstance(model, str):
            self.dataset, self.model = load_from_str(
                data, model, benchmark_type="tabpfn", random_state=random_state
            )
            if labels is not None:
                msg = "Labels must not be provided when data is a dataset string."
                raise ValueError(msg)
            x_train = self.dataset.x_train
            y_train = self.dataset.y_train
            x_test = self.dataset.x_test
            y_test = self.dataset.y_test

        elif isinstance(data, tuple) and not isinstance(model, str):
            if labels is None:
                msg = "When data is a tuple, labels must be provided as (y_train, y_test)."
                raise ValueError(msg)
            x_train, x_test = data
            y_train, y_test = labels
            if x_train.shape[0] != y_train.shape[0]:
                msg = "x_train and y_train must have the same number of samples."
                raise ValueError(msg)
            if x_test.shape[0] != y_test.shape[0]:
                msg = "x_test and y_test must have the same number of samples."
                raise ValueError(msg)
            if x_train.shape[1] != x_test.shape[1]:
                msg = "x_train and x_test must have the same number of features."
                raise ValueError(msg)
        else:
            msg = (
                "Invalid combination of data and model arguments. "
                "Please provide either both as strings or both as objects."
            )
            raise ValueError(msg)

        predict_function, _ = get_predict_function_and_model_type(
            self.model, class_index=class_index
        )

        self.model._shapiq_predict_function = predict_function 

        imputer = TabPFNImputer(
            model=self.model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
        )
        imputer.fit(np.asarray(x_train[x_explain]))

        self._game = imputer
        self._computer = TabPFNComputer(self._game)

    def exact_values(self, index: IndexType, order: int, budget: int=1000) -> InteractionValues:
        """Compute exact interaction values using the benchmark computer."""
        return self._computer.exact_values(index=index, order=order, budget=budget) #TODO check if others also need budget

    @property
    def game(self) -> TabPFNImputer:
        """Game instance used by this benchmark."""
        return self._game

    @property
    def computer(self) -> TabPFNComputer:
        """Ground truth computer used by this benchmark."""
        return self._computer
