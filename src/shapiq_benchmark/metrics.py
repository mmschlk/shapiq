"""Metrics comparing estimated interaction values with the ground truth.

All metrics compare the interactions of order ``1`` to the ground truth's ``max_order`` (or of a
single order). The order-0 term is excluded: it is the value of the empty coalition, not an
attribution. Higher is better for the ranking metrics and for faithfulness; lower is better for
the error metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau, spearmanr

from shapiq.utils import powerset

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues

__all__ = ["RANK_TOLERANCE", "compare", "error_metrics", "faithfulness", "ranking_metrics"]

RANK_TOLERANCE = 1e-6
"""Values closer than this times the largest absolute ground-truth value rank as equal."""


def _aligned(
    ground_truth: InteractionValues,
    estimate: InteractionValues,
    order: int | None,
) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
    """Return the compared interactions and their ground-truth and estimated values."""
    if order is None:
        min_size, max_size = 1, ground_truth.max_order
    elif 1 <= order <= ground_truth.max_order:
        min_size = max_size = order
    else:
        msg = f"order must be between 1 and the ground truth's max_order {ground_truth.max_order}."
        raise ValueError(msg)
    interactions = list(
        powerset(range(ground_truth.n_players), min_size=min_size, max_size=max_size)
    )
    truth = np.array([ground_truth[interaction] for interaction in interactions], dtype=float)
    estimated = np.array([estimate[interaction] for interaction in interactions], dtype=float)
    return interactions, truth, estimated


def error_metrics(truth: np.ndarray, estimated: np.ndarray) -> dict[str, float]:
    """Return the squared and absolute errors between two aligned value vectors.

    Returns:
        ``mse``, ``mae``, ``sse``, ``sae``, and ``nmse`` (the squared error relative to the squared
        norm of the ground truth; ``nan`` if the ground truth is zero).
    """
    difference = estimated - truth
    sse = float(np.sum(difference**2))
    norm = float(np.sum(truth**2))
    return {
        "mse": sse / truth.size,
        "mae": float(np.mean(np.abs(difference))),
        "sse": sse,
        "sae": float(np.sum(np.abs(difference))),
        "nmse": sse / norm if norm > 0 else float("nan"),
    }


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest absolute values (ties broken by position)."""
    return np.argsort(-np.abs(values), kind="stable")[:k]


def ranking_metrics(truth: np.ndarray, estimated: np.ndarray, k: int = 10) -> dict[str, float]:
    """Return rank agreement metrics between two aligned value vectors.

    Values that differ by less than :data:`RANK_TOLERANCE` times the largest absolute ground-truth
    value rank as equal, so float noise does not order values that are equal (e.g. the many zeros
    of a sparse game).

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
        The error metrics and the ranking metrics (see :func:`error_metrics` and
        :func:`ranking_metrics`).

    Raises:
        ValueError: If ``order`` is not between ``1`` and the ground truth's ``max_order``.
    """
    _, truth, estimated = _aligned(ground_truth, estimate, order)
    return {**error_metrics(truth, estimated), **ranking_metrics(truth, estimated, k=k)}


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
    if 2**n <= n_samples:
        coalitions = np.array([[i in s for i in range(n)] for s in powerset(range(n))], dtype=bool)
    else:
        coalitions = np.random.default_rng(random_state).random((n_samples, n)) < 0.5
    values = game(coalitions)
    interactions = [
        (interaction, estimate[interaction])
        for interaction in estimate.interaction_lookup
        if len(interaction) >= 1
    ]
    reconstruction = np.full(coalitions.shape[0], float(estimate.baseline_value))
    for interaction, value in interactions:
        reconstruction += value * coalitions[:, list(interaction)].all(axis=1)
    residual = float(np.sum((values - reconstruction) ** 2))
    total = float(np.sum((values - values.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")
