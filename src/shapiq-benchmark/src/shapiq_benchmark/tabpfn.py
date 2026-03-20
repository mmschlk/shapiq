from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from shapiq import Game, TabPFNImputer
from shapiq.explainer.tabpfn import TabPFNExplainer

from .base import Benchmark, GroundTruthComputer

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import IndexType


class TabPFNBenchmark(Benchmark):
    def __init__(self, tabpfn_model, data, labels, x_explain, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tabpfn_model = tabpfn_model
        self.x_explain = x_explain
        computer = TabPFNComputer(
            tabpfn_model=self.tabpfn_model,
            data=deepcopy(data),
            labels=deepcopy(labels),
            x_explain=self.x_explain,
        )
        game = TabPFNGame()
        super().__init__(game=game, computer=computer)


class TabPFNComputer(GroundTruthComputer):
    def __init__(self, tabpfn_model, data, labels, x_explain) -> None:
        self.tabpfn_model = tabpfn_model
        self.data = data
        self.labels = labels
        self.x_explain = x_explain

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        # Implement the logic to compute exact values using TabPFN
        explainer = TabPFNExplainer(
            model=self.tabpfn_model,
            data=self.data,
            labels=self.labels,
            index=index,
            approximator="auto",
            max_order=order,
        )
        return explainer.explain(self.x_explain)


class TabPFNGame(Game):
    def __init__(
        self,
        tabpfn_model,
        data,
        labels,
        x_explain,
        *,
        x_test=None,
        empty_prediction=None,
        normalize=True,
        class_label: int | None = None,
    ) -> None:
        self.tabpfn_model = tabpfn_model
        n_samples = data.shape[0]
        x_train = data
        y_train = labels

        if x_test is None and empty_prediction is None:
            sections = [int(0.8 * n_samples)]
            x_train, x_test = np.split(data, sections)
            y_train, _ = np.split(labels, sections)

        if x_test is None:
            x_test = x_train  # is not used in the TabPFNImputer if empty_prediction is set

        self.imputer = TabPFNImputer(
            model=self.tabpfn_model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            empty_prediction=empty_prediction,
        )
        super().__init__(
            n_players=data.shape[1],
            normalize=normalize,
            class_label=class_label,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return self.imputer(coalitions)
