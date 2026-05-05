"""Pathdependent (tree) benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, get_args

import numpy as np
import pandas as pd
from shapiq.typing import IndexType, Model
from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

from .base import Benchmark
from .computers import PathdependentComputer
from .setup import (
    infer_data_type,
    load_from_str,
)
from .bench_types import BenchmarkDataset

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues


class PathdependentBench(Benchmark[IndexType]):
    """Benchmark for tree-based pathdependent explanations."""

    def __init__(
        self,
        data: str | np.ndarray,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        *,
        x_explain: int | None = 0,
        class_index: int | None = 1,
        random_state: int | None = 42,
    ) -> None:
        """Initialize the benchmark by loading data/model and fitting the model.

        Args:
            data: Dataset identifier (e.g. "adult_census") or a NumPy array containing the data.
            model: Model identifier (e.g. "decision_tree") or a fitted model object.
            x_explain: Instance to explain.
            class_index: Class index for classification models.
            random_state: Random state used for data split and model init.
        """
        if isinstance(data, str) and isinstance(model, str):
            self.dataset, self.model = load_from_str(
                data, model, benchmark_type="pathdependent", random_state=random_state
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
        else:
            data_type = infer_data_type(self.model)

        if data_type == "regression":
            class_index = None

        self._game = TreeSHAPIQXAI(
            x=(self.x_train[x_explain]),
            tree_model=self.model,
            class_label=class_index,
            verbose=False,
        )
        self._computer = PathdependentComputer(self._game)

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute exact interaction values using the benchmark computer."""
        return self._computer.exact_values(index=index, order=order)

    @property
    def game(self) -> TreeSHAPIQXAI:
        """Game instance used by this benchmark."""
        return self._game

    @property
    def computer(self) -> PathdependentComputer[IndexType]:
        """Ground truth computer used by this benchmark."""
        return self._computer
