"""Implementation of the benchmark for KNNExplainer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from ._util import keep_first_n
from .base import KNNBenchmarkBase

if TYPE_CHECKING:
    import numpy.typing as npt

    from shapiq.typing import GameValues


class KNNExplainerXAI(KNNBenchmarkBase):
    """Benchmark game for the KNNExplainer."""

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> GameValues:
        utilities = np.zeros(coalitions.shape[0])

        for i, coalition in enumerate(coalitions):
            coalition_sorted = coalition[self.sortperm]
            coalition_k_nearest = keep_first_n(coalition_sorted, n=self.k)
            # TODO(Zaphoood): Fix divisor for case N < k  # noqa: TD003
            utilities[i] = (
                np.sum(self.y_train_sorted[coalition_k_nearest] == self.class_index) / self.k
            )

        return utilities
