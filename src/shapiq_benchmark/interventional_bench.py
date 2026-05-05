"""Interventional benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, get_args

import numpy as np
from shapiq.typing import IndexType, Model
from shapiq_games.benchmark.interventionaltreeshapiq_xai.base import InterventionalGame

from .base import Benchmark
from .computers import InterventionalComputer
from .setup import (
    AllSupportedDatasets,
    SupportedModelsInterventional,
    load_from_str,
    infer_data_type,
)
from .bench_types import BenchmarkDataset

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues




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
        """Initialize the benchmark by loading data/model and fitting the model.

        Args:
                data_str: Dataset identifier (e.g. "adult_census").
                model_str: Model identifier (e.g. "decision_tree").
                x_explain: Instance to explain.
                class_index: Class index for classification models.
                random_state: Random state used for data split and model init.
        """
        if isinstance(data, str) and isinstance(model, str):
            self.dataset, self.model = load_from_str(
                data, model, benchmark_type="interventional", random_state=random_state
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

        self._game = InterventionalGame(
            model=self.model,
            reference_data=self.x_train,
            target_instance=(self.x_train[x_explain]),
            class_index=class_index,
        )
        self._computer = InterventionalComputer(self._game)

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute exact interaction values using the benchmark computer."""
        return self._computer.exact_values(index=index, order=order)

    @property
    def game(self) -> InterventionalGame:
        """Game instance used by this benchmark."""
        return self._game

    @property
    def computer(self) -> InterventionalComputer[IndexType]:
        """Ground truth computer used by this benchmark."""
        return self._computer
