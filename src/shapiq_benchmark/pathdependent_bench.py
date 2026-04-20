"""Pathdependent (tree) benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapiq.typing import IndexType
from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

from .base import Benchmark, GroundTruthComputer
from .computers import PathdependentComputer
from .setup import load_data_from_str, load_model_from_str
from .bench_types import BenchmarkDataset

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues


class PathdependentBench(Benchmark[IndexType]):
    """Benchmark for tree-based pathdependent explanations."""

    def __init__(
        self,
        data_str: str,
        model_str: str,
        *,
        class_index: int | None = 1,
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
        test_size: float = 0.2,
        n_estimators: int = 10,
    ) -> None:
        """Initialize the benchmark by loading data/model and fitting the model.

        Args:
            data_str: Dataset identifier (e.g. "adult_census").
            model_str: Model identifier (e.g. "decision_tree").
            class_index: Class index for classification models.
            normalize: Whether to normalize game values.
            verbose: Whether to enable verbose output in the game.
            random_state: Random state used for data split and model init.
            test_size: Fraction of data used for testing.
            n_estimators: Number of estimators for random forest models.
        """
        self.dataset: BenchmarkDataset = load_data_from_str(
            data_str,
            random_state=random_state,
            test_size=test_size,
        )
        self.model = load_model_from_str(
            model_str,
            self.dataset,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        self.model.fit(self.dataset.x_train, self.dataset.y_train)

        if self.dataset.data_type == "regression":
            class_index = None
            
        self._game = TreeSHAPIQXAI(
            x=self.dataset.x_explain,
            tree_model=self.model,
            class_label=class_index,
            normalize=normalize,
            verbose=verbose,
        )
        self._computer = PathdependentComputer(self._game)

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute exact interaction values using the benchmark computer."""
        return self._computer.exact_values(index=index, order=order)
    
    def get_dataset(self) -> BenchmarkDataset:
        """Get the dataset used in this benchmark."""
        return self.dataset 

    @property
    def game(self) -> Game:
        """Game instance used by this benchmark."""
        return self._game

    @property
    def computer(self) -> GroundTruthComputer[IndexType]:
        """Ground truth computer used by this benchmark."""
        return self._computer
