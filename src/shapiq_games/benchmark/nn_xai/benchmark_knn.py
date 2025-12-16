"""Implementation of the benchmark for KNNExplainer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from ._util import keep_first_n
from .base import NNBenchmarkBase

if TYPE_CHECKING:
    import numpy.typing as npt


class KNNExplainerXAI(NNBenchmarkBase):
    """Benchmark game for the KNNExplainer."""

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        utilities = np.zeros(coalitions.shape[0])

        for coalition in coalitions:
            coalition_first_k = keep_first_n(coalition, n=self.k)
            # TODO(Zaphoood): Handle case where N < k  # noqa: TD003
            utility = np.sum(self.y_train_sorted[coalition_first_k] == self.class_index) / self.k
            coalition_tuple = tuple(sorted(self.sortperm[coalition]))
            utilities[coalition_tuple] = utility

        return utilities
