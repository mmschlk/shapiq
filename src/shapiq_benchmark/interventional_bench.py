"""Interventional benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from shapiq.tree.interventional.game import InterventionalGame

from .computers import InterventionalComputer
from .local_xai_bench import LocalXAIBench

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq.typing import Model


class InterventionalBench(LocalXAIBench):
    """Benchmark for interventional tree-based explanations."""

    def __init__(
        self,
        data: str | np.ndarray,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        data_type: Literal["classification", "regression"] | None = None,
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
            data_type: Type of data ("classification" or "regression"), or None to infer. Must be provided if data is a NumPy array.
            x_explain: Instance to explain.
            class_index: Class index for classification models.
            random_state: Random state used for data split and model init.
            **kwargs: Additional keyword arguments for model building.
        """
        class_index, _ = self._load_and_set_dataset_and_model(
            data,
            model,
            data_type=data_type,
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
