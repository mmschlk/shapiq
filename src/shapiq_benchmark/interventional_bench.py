"""Interventional benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING


from shapiq.tree.interventional.game import InterventionalGame


from .computers import InterventionalComputer
from .local_xai_bench import LocalXAIBench

if TYPE_CHECKING:
    from collections.abc import Callable
    import numpy as np
    from shapiq import InteractionValues
    from shapiq.typing import IndexType, Model


class InterventionalBench(LocalXAIBench):
    """Benchmark for interventional tree-based explanations."""

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
        """Initialize the benchmark by loading data and model and fitting the model.

        Args:
            data: Dataset identifier (e.g. "adult_census") or a NumPy array containing the data.
            model: Model identifier (e.g. "decision_tree") or a fitted model object.
            x_explain: Instance to explain.
            class_index: Class index for classification models.
            random_state: Random state used for data split and model init.
            **kwargs: Additional keyword arguments for model building.
        """
        class_index, _ = self._load_dataset_and_model(
            data,
            model,
            benchmark_type="interventional",
            random_state=random_state,
            class_index=class_index,
            **kwargs,
        )

        x_index = self._resolve_x_explain(x_explain)

        self._game = InterventionalGame(
            model=self.model,
            reference_data=self.data,
            target_instance=self.data[x_index],
            class_index=class_index,
        )
        self._computer = InterventionalComputer(self._game)

    def exact_values(self, index: IndexType, order: int, **kwargs: object) -> InteractionValues:
        """Compute exact interaction values using the InterventionalBench computer.

        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            **kwargs: Additional keyword arguments for computation.

        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.exact_values(index=index, order=order, **kwargs)

    @property
    def game(self) -> InterventionalGame:
        """Game instance used by the Interventioanl Benchmark."""
        return self._game

    @property
    def computer(self) -> InterventionalComputer[IndexType]:
        """Ground truth computer used by the Interventioanl Benchmark."""
        return self._computer
