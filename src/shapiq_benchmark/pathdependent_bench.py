"""Pathdependent (tree) benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING


from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

from .computers import PathdependentComputer
from .local_xai_bench import LocalXAIBench

if TYPE_CHECKING:
    from collections.abc import Callable
    import numpy as np
    from shapiq import InteractionValues
    from shapiq.typing import IndexType, Model


class PathdependentBench(LocalXAIBench):
    """Benchmark for tree-based pathdependent explanations."""

    def __init__(
        self,
        data: str | np.ndarray,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        *,
        x_explain: int | None = 0,
        class_index: int | None = 1,
        random_state: int | None = 42,
        **kwargs: object,
    ) -> None:
        """Initialize the Pathdependent Benchmark by loading data and model and fitting the model.

        Args:
            data: Dataset identifier (e.g. "adult_census") or a NumPy array containing the data.
            model: Model identifier (e.g. "decision_tree") or a fitted model object.
            x_explain: Instance to explain.
            class_index: Class index for classification models.
            random_state: Random state used for data split and model init.
            **kwargs: Additional keyword arguments for model building.
        """
        class_index, _data_type = self._load_dataset_and_model(
            data,
            model,
            benchmark_type="pathdependent",
            random_state=random_state,
            class_index=class_index,
            **kwargs,
        )

        x_index = self._resolve_x_explain(x_explain)

        self._game = TreeSHAPIQXAI(
            x=self.data[x_index],
            tree_model=self.model,
            class_label=class_index,
            verbose=False,
        )
        self._computer = PathdependentComputer(self._game)

    def exact_values(
        self, index: IndexType, order: int, **kwargs: object
    ) -> InteractionValues:
        """Compute exact interaction values using the PathdependentBench computer.

        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            **kwargs: Additional keyword arguments for computation.

        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.exact_values(index=index, order=order, **kwargs)

    @property
    def game(self) -> TreeSHAPIQXAI:
        """Game instance used by the Pathdependent Benchmark."""
        return self._game

    @property
    def computer(self) -> PathdependentComputer[IndexType]:
        """Ground truth computer used by the Pathdependent Benchmark."""
        return self._computer
