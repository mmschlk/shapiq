"""Unsupervised data games: how much statistical dependence a coalition of features shares."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer

from shapiq.game import Game
from shapiq_games._base import ConfigMixin, as_bool_coalitions
from shapiq_games._setup import configure

__all__ = ["UnsupervisedData", "total_correlation"]


def total_correlation(data: np.ndarray) -> float:
    r"""Return the total correlation of discrete data.

    .. math::
        TC(X_1, \dots, X_k) = \sum_i H(X_i) - H(X_1, \dots, X_k)

    Args:
        data: Discrete data of shape ``(n_samples, n_features)``.

    Returns:
        The total correlation in nats (zero for a single feature).
    """
    if data.shape[1] < 2:
        return 0.0
    marginal = sum(
        float(entropy(np.unique(data[:, i], return_counts=True)[1])) for i in range(data.shape[1])
    )
    joint = float(entropy(np.unique(data, axis=0, return_counts=True)[1]))
    return marginal - joint


class UnsupervisedData(ConfigMixin, Game):
    """The unsupervised data game: the total correlation of a coalition of features.

    The players are the features, discretized into equal-width bins. The value of a coalition is
    the total correlation of its features, which is zero for the empty coalition and for single
    features (Balestra et al., 2022).

    Attributes:
        data_discrete: The discretized data.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        n_bins: int = 20,
        verbose: bool = False,
    ) -> None:
        """Initialize the unsupervised data game.

        Args:
            data: The data of shape ``(n_samples, n_features)``.
            n_bins: The number of equal-width bins per feature. Defaults to ``20``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        data = np.asarray(data, dtype=float)
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None
        )
        self.data_discrete = np.column_stack(
            [
                discretizer.fit_transform(data[:, [i]]).ravel().astype(int)
                for i in range(data.shape[1])
            ]
        )
        super().__init__(data.shape[1], normalize=False, verbose=verbose)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the total correlation of the features of each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        return np.array(
            [total_correlation(self.data_discrete[:, coalition]) for coalition in coalitions]
        )

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        n_samples: int | None = None,
        n_bins: int = 20,
        random_state: int = 42,
        dataset_params: dict[str, Any] | None = None,
    ) -> Self:
        """Build the game on a registered dataset.

        Args:
            dataset: The dataset name.
            n_samples: Use a seeded sample of this many rows (``None`` for all).
            n_bins: The number of equal-width bins per feature. Defaults to ``20``.
            random_state: The seed of the row sample.
            dataset_params: Parameters of synthetic datasets.

        Returns:
            The configured game.
        """
        setup = configure(
            dataset=dataset, model=None, random_state=random_state, dataset_params=dataset_params
        )
        data = setup.split.dataset.x
        if n_samples is not None and n_samples < data.shape[0]:
            rng = np.random.default_rng(random_state)
            data = data[np.sort(rng.choice(data.shape[0], size=n_samples, replace=False))]
        game = cls(data, n_bins=n_bins)
        return game._set_config(
            dataset=dataset,
            dataset_params=setup.config["dataset_params"],
            random_state=random_state,
            n_samples=n_samples,
            n_bins=n_bins,
        )
