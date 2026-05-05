"""Interventional benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from shapiq.typing import IndexType, Model
from shapiq.tree.interventional.game import InterventionalGame

from .base import Benchmark
from .computers import InterventionalComputer
from .setup import (
    load_from_str,
    infer_data_type,
)

if TYPE_CHECKING:
    from shapiq import InteractionValues


class InterventionalBench(Benchmark[IndexType]):
    """Benchmark for interventional tree-based explanations."""

    def __init__(
        self,
        data: str | np.ndarray,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        *,
        x_explain: int | None = 0,
        class_index: int | None = 1,
        random_state: int | None = 42,
    ) -> None:
        """Initialize the benchmark by loading data and model and fitting the model.

        Args:
            data: Dataset identifier (e.g. "adult_census") or a NumPy array containing the data.
            model: Model identifier (e.g. "decision_tree") or a fitted model object.
            x_explain: Instance to explain.
            class_index: Class index for classification models.
            random_state: Random state used for data split and model init.
        """
        if isinstance(data, str) and isinstance(model, str):
            self.dataset, self.model = load_from_str(
                data, model, benchmark_type="interventional", random_state=random_state
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

        self._game = InterventionalGame(
            model=self.model,
            reference_data=self.x_train,
            target_instance=self.x_train[x_index],
            class_index=class_index,
        )
        self._computer = InterventionalComputer(self._game)

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the InterventionalBench computer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional Budget for computation.
        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.exact_values(index=index, order=order, budget=budget)

    @property
    def game(self) -> InterventionalGame:
        """Game instance used by the Interventioanl Benchmark."""
        return self._game

    @property
    def computer(self) -> InterventionalComputer[IndexType]:
        """Ground truth computer used by the Interventioanl Benchmark."""
        return self._computer
