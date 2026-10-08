"""Metrics comparing estimated interaction values with the ground truth.

All metrics compare the interactions of order ``1`` to the ground truth's ``max_order`` (or of a
single order). The order-0 term is excluded: it is the value of the empty coalition, not an
attribution. Higher is better for the ranking metrics and for faithfulness; lower is better for
the error metrics.

An interaction that an :class:`~shapiq.InteractionValues` does not store is 0. The metrics only
visit the interactions where the ground truth or the estimate is nonzero, so their cost grows with
the stored interactions, not with the ``C(n, k)`` possible ones: a sparse game with hundreds of
players is compared as quickly as a small one.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau, spearmanr

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues

__all__ = ["RANK_TOLERANCE", "compare", "error_metrics", "faithfulness", "ranking_metrics"]

RANK_TOLERANCE = 1e-6
"""Values closer than this times the largest absolute ground-truth value rank as equal."""


def _aligned(
    ground_truth: InteractionValues,
    estimate: InteractionValues,
    order: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the values where the ground truth or the estimate is nonzero, and their total count.

    The count is that of all compared interactions, including those that are 0 in both.
    """
    if order is None:
        min_size, max_size = 1, ground_truth.max_order
    elif 1 <= order <= ground_truth.max_order:
        min_size = max_size = order
    else:
        msg = f"order must be between 1 and the ground truth's max_order {ground_truth.max_order}."
        raise ValueError(msg)
    support = sorted(
        {
            interaction
            for values in (ground_truth, estimate)
            for interaction, value in values.dict_values.items()
            if min_size <= len(interaction) <= max_size and value != 0
        },
        key=lambda interaction: (len(interaction), interaction),
    )
    truth = np.array([ground_truth[interaction] for interaction in support], dtype=float)
    estimated = np.array([estimate[interaction] for interaction in support], dtype=float)
    n = ground_truth.n_players
    n_interactions = sum(math.comb(n, size) for size in range(min_size, max_size + 1))
    return truth, estimated, n_interactions


def error_metrics(
    truth: np.ndarray, estimated: np.ndarray, n_interactions: int | None = None
) -> dict[str, float]:
    """Return the squared and absolute errors between two aligned value vectors.

    Args:
        truth: The ground-truth values.
        estimated: The estimated values.
        n_interactions: The number of compared interactions, if the vectors leave out interactions
            that are 0 in both. Defaults to ``None``, the length of the vectors.

    Returns:
        ``mse``, ``mae``, ``sse``, ``sae``, and ``nmse`` (the squared error relative to the squared
        norm of the ground truth; ``nan`` if the ground truth is zero). The means are over all
        ``n_interactions``.
    """
    n_interactions = truth.size if n_interactions is None else n_interactions
    difference = estimated - truth
    sse = float(np.sum(difference**2))
    sae = float(np.sum(np.abs(difference)))
    norm = float(np.sum(truth**2))
    return {
        "mse": _mean(sse, n_interactions),
        "mae": _mean(sae, n_interactions),
        "sse": sse,
        "sae": sae,
        "nmse": sse / norm if norm > 0 else float("nan"),
    }


def _mean(total: float, count: int) -> float:
    """Return ``total / count``; exact for counts beyond the float range (e.g. ``2**1100``)."""
    return float(Fraction(total) / count) if count > 0 else float("nan")


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest absolute values (ties broken by position)."""
    return np.argsort(-np.abs(values), kind="stable")[:k]


def ranking_metrics(truth: np.ndarray, estimated: np.ndarray, k: int = 10) -> dict[str, float]:
    """Return rank agreement metrics between two aligned value vectors.

    Values that differ by less than :data:`RANK_TOLERANCE` times the largest absolute ground-truth
    value rank as equal, so float noise does not order values that are equal (e.g. the many zeros
    of a sparse game). :func:`compare` passes only the interactions where the ground truth or the
    estimate is nonzero: pairs of zeros carry no ranking information.

    Args:
        truth: The ground-truth values.
        estimated: The estimated values.
        k: The number of top interactions (by absolute ground-truth value) for the ``@k`` metrics.

    Returns:
        ``kendall_tau`` and ``spearman`` of all values; ``precision_at_k``, the share of the ``k``
        largest absolute estimates that are among the largest absolute ground-truth values (all
        values tied with the ``k``-th largest included); and ``kendall_tau_at_k``, Kendall's tau
        restricted to those top ground-truth interactions. Correlations of constant vectors are
        ``nan``.
    """
    k = min(k, truth.size)
    if k == 0:
        nan = float("nan")
        return {"kendall_tau": nan, "spearman": nan, "precision_at_k": nan, "kendall_tau_at_k": nan}
    scale = RANK_TOLERANCE * float(np.max(np.abs(truth)))
    if scale > 0:
        truth, estimated = np.round(truth / scale) * scale, np.round(estimated / scale) * scale
    threshold = np.sort(np.abs(truth))[::-1][k - 1]  # the k-th largest absolute ground truth
    top_truth = np.flatnonzero(np.abs(truth) >= threshold)
    top_estimate = _top_k(estimated, k)
    return {
        "kendall_tau": _correlation(kendalltau, truth, estimated),
        "spearman": _correlation(spearmanr, truth, estimated),
        "precision_at_k": float(np.mean(np.abs(truth[top_estimate]) >= threshold)),
        "kendall_tau_at_k": _correlation(kendalltau, truth[top_truth], estimated[top_truth]),
    }


def _correlation(statistic, a: np.ndarray, b: np.ndarray) -> float:  # noqa: ANN001
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(statistic(a, b)[0])


def compare(
    ground_truth: InteractionValues,
    estimate: InteractionValues,
    *,
    k: int = 10,
    order: int | None = None,
) -> dict[str, float]:
    """Compare an estimate with the ground truth.

    Args:
        ground_truth: The exact interaction values.
        estimate: The estimated interaction values.
        k: The number of top interactions for the ``@k`` metrics. Defaults to ``10``.
        order: Compare only the interactions of this order. Defaults to ``None``, which compares
            all orders from ``1`` to the ground truth's ``max_order``.

    Returns:
        The error metrics, averaged over all interactions of the compared orders, and the ranking
        metrics over the interactions where the ground truth or the estimate is nonzero (see
        :func:`error_metrics` and :func:`ranking_metrics`).

    Raises:
        ValueError: If ``order`` is not between ``1`` and the ground truth's ``max_order``.
    """
    truth, estimated, n_interactions = _aligned(ground_truth, estimate, order)
    return {
        **error_metrics(truth, estimated, n_interactions),
        **ranking_metrics(truth, estimated, k=k),
    }


def faithfulness(
    game: Game,
    estimate: InteractionValues,
    *,
    n_samples: int = 1000,
    random_state: int = 0,
) -> float:
    r"""Return how well the estimate reconstructs the game (the R² of the reconstruction).

    Each sampled coalition :math:`S` is reconstructed as
    :math:`\hat v(S) = b + \sum_{T \subseteq S, |T| \geq 1} \hat\phi(T)`, where :math:`b` is the
    estimate's baseline value. Coalitions are sampled uniformly at random (every player present
    with probability one half), or all coalitions are used if there are at most ``n_samples``. The
    R² is not clipped and can be negative.

    Args:
        game: The game.
        estimate: The estimated interaction values.
        n_samples: The number of sampled coalitions. Defaults to ``1000``.
        random_state: The seed of the coalition sample. Defaults to ``0``.

    Returns:
        The coefficient of determination between the game values and the reconstruction.
    """
    n = game.n_players
    if 2**n <= n_samples:  # every coalition, as the binary digits of 0, ..., 2**n - 1
        coalitions = (np.arange(2**n)[:, None] >> np.arange(n) & 1).astype(bool)
    else:
        coalitions = np.random.default_rng(random_state).random((n_samples, n)) < 0.5
    values = game(coalitions)
    reconstruction = np.full(coalitions.shape[0], float(estimate.baseline_value))
    for interaction, value in estimate.dict_values.items():
        if len(interaction) >= 1 and value != 0:
            reconstruction += value * coalitions[:, list(interaction)].all(axis=1)
    residual = float(np.sum((values - reconstruction) ** 2))
    total = float(np.sum((values - values.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")
