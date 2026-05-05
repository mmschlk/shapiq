"""Pathdependent (tree) benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

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

if TYPE_CHECKING:
    from shapiq import InteractionValues


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
        """Initialize the Pathdependent Benchmark by loading data and model and fitting the model.

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
        else:
            data_type = infer_data_type(self.model)

        if data_type == "regression":
            class_index = None

        x_index = 0 if x_explain is None else x_explain
        if not isinstance(x_index, int):
            msg = "x_explain must be an int index."
            raise ValueError(msg)

        self._game = TreeSHAPIQXAI(
            x=self.x_train[x_index],
            tree_model=self.model,
            class_label=class_index,
            verbose=False,
        )
        self._computer = PathdependentComputer(self._game)

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the PathdependentBench computer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional budget for computation.
        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.exact_values(index=index, order=order, budget=budget)

    @property
    def game(self) -> TreeSHAPIQXAI:
        """Game instance used by the Pathdependent Benchmark."""
        return self._game

    @property
    def computer(self) -> PathdependentComputer[IndexType]:
        """Ground truth computer used by the Pathdependent Benchmark."""
        return self._computer
