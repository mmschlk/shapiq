"""Descriptive signed estimator combinations on complete, authenticated panels.

Call once per target and relative budget. Each block is one game/estimator-seed
panel with aligned nonempty coefficients, a dataset ID and a benchmark weight.
The caller owns capability, provenance and score-eligibility selection. Missing
methods must never be filled with zero; sparse *coefficients* may be zero.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def aligned_block(game: dict, rows: Sequence[dict], methods: Sequence[str], weight: float) -> dict:
    """Align successful saved estimates for one game, budget and estimator seed.

    Include every target coordinate, even zeros omitted by sparse estimates/truth.
    Dataset membership comes from the frozen game, never a random coordinate split.
    """
    by_method = {row["method"]: row for row in rows}
    keys = {(row["game_id"], row["budget"], row["seed"]) for row in rows}
    if (
        len(keys) != 1
        or len(by_method) != len(rows)
        or set(by_method) != set(methods)
        or any(row["game_id"] != game["id"] or row["status"] != "ok" for row in rows)
    ):
        message = "Expected one complete successful game/budget/seed panel."
        raise ValueError(message)

    def unpack(vector: dict) -> dict:
        pairs = list(zip(map(tuple, vector["coordinates"]), vector["values"], strict=True))
        if len(dict(pairs)) != len(pairs):
            message = "Duplicate coefficient coordinates."
            raise ValueError(message)
        return {key: float(value) for key, value in pairs if key}

    truth = unpack(game["truth"])
    estimates = [unpack(by_method[method]["estimate"]) for method in methods]
    coordinates = [
        key
        for degree in range(1, game["order"] + 1)
        for key in itertools.combinations(range(game["n_players"]), degree)
    ]
    if not set(truth).union(*estimates) <= set(coordinates):
        message = "Coefficient is outside the game's nonempty target coordinates."
        raise ValueError(message)
    return {
        "id": next(iter(keys)),
        "dataset": game["metadata"]["dataset"],
        "weight": weight,
        "truth": [truth.get(key, 0.0) for key in coordinates],
        "estimates": [[estimate.get(key, 0.0) for estimate in estimates] for key in coordinates],
    }


def fit_panel(methods: Sequence[str], blocks: Sequence[dict], *, folds: int = 5) -> dict:
    """Fit unregularized, signed least squares without an intercept.

    Normalize each block by its reference norm; its squared residual is nMSE.
    Block weights are fixed before fitting and are shared across all methods.
    R² uses weighted, centered normalized reference coefficients. Dataset-grouped
    folds keep every construction and estimator seed of a dataset together.
    All-data coefficients are descriptive; only held-out metrics measure transfer.
    """
    if len(methods) < 2 or len(set(methods)) != len(methods):
        message = "At least two distinct methods are required."
        raise ValueError(message)
    if type(folds) is not int or folds < 2 or not blocks:
        message = "At least two folds and a nonempty panel are required."
        raise ValueError(message)
    xs, ys, weights, groups = [], [], [], []
    identities = set()
    for block in blocks:
        identity = block["id"]
        if identity in identities:
            message = "Duplicate game/seed panel."
            raise ValueError(message)
        identities.add(identity)
        x = np.asarray(block["estimates"], dtype=float)
        y = np.asarray(block["truth"], dtype=float)
        weight = float(block["weight"])
        dataset = block["dataset"]
        if (
            y.ndim != 1
            or not y.size
            or x.shape != (y.size, len(methods))
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isfinite(weight)
            or weight <= 0
            or not isinstance(dataset, str)
            or not dataset
        ):
            message = "Invalid complete panel, dataset or weight."
            raise ValueError(message)
        norm = np.linalg.norm(y)
        if not np.isfinite(norm) or norm <= 0:
            message = "Reference must have finite positive energy."
            raise ValueError(message)
        xs.append(x / norm)
        ys.append(y / norm)
        weights.append(np.full(y.size, weight))
        groups.extend([dataset] * y.size)
    x, y, weight = np.vstack(xs), np.concatenate(ys), np.concatenate(weights)
    # Normalize before summing to avoid overflow for finite, very large weights.
    maximum = float(weight.max())
    weight /= maximum
    weight /= sum(float(block["weight"]) / maximum for block in blocks)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        message = "Normalized coefficients must be finite."
        raise ValueError(message)
    groups = np.asarray(groups)
    datasets = sorted(set(groups), key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    if len(datasets) < 2:
        message = "Held-out evaluation requires at least two datasets."
        raise ValueError(message)
    sqrt_weight = np.sqrt(weight)

    def fit(mask: np.ndarray) -> tuple[np.ndarray, int]:
        coefficients, _, rank, _ = np.linalg.lstsq(
            x[mask] * sqrt_weight[mask, None], y[mask] * sqrt_weight[mask], rcond=None
        )
        return coefficients, int(rank)

    def metrics(prediction: np.ndarray) -> dict:
        error = float(np.dot(weight, (prediction - y) ** 2))
        mean = float(np.dot(weight, y) / weight.sum())
        variance = float(np.dot(weight, (y - mean) ** 2))
        if not np.isfinite([error, variance]).all():
            message = "Combination metrics overflowed."
            raise ValueError(message)
        return {"nmse": error, "r2": 1 - error / variance if variance > 0 else None}

    coefficients, rank = fit(np.ones(y.size, dtype=bool))
    prediction = np.empty_like(y)
    fold_details = []
    for fold in range(min(folds, len(datasets))):
        held_out = datasets[fold :: min(folds, len(datasets))]
        test = np.isin(groups, held_out)
        alpha, fold_rank = fit(~test)
        prediction[test] = x[test] @ alpha
        fold_details.append(
            {"held_out_datasets": held_out, "coefficients": alpha.tolist(), "rank": fold_rank}
        )

    # Rows are coefficient observations and columns are estimators. Center across
    # estimators so PCA describes distances between their normalized error vectors.
    errors = (x - y[:, None]) * sqrt_weight[:, None]
    errors -= errors.mean(axis=1, keepdims=True)
    covariance = errors.T @ errors
    if not np.isfinite(covariance).all():
        message = "Error PCA overflowed."
        raise ValueError(message)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    axes = eigenvectors[:, order[:2]]
    # Remove arbitrary sign flips from otherwise equivalent eigendecompositions.
    for column in range(axes.shape[1]):
        if axes[np.argmax(np.abs(axes[:, column])), column] < 0:
            axes[:, column] *= -1
    positions = axes * np.sqrt(eigenvalues[:2])
    total = float(eigenvalues.sum())
    explained = eigenvalues[:2] / total if total > 0 else np.zeros(2)
    return {
        "methods": list(methods),
        "coefficients": coefficients.tolist(),
        "rank": rank,
        "train": metrics(x @ coefficients),
        "held_out": metrics(prediction),
        "equal_weights": metrics(x.mean(axis=1)),
        "n_panels": len(blocks),
        "n_datasets": len(datasets),
        "folds": fold_details,
        "pca": {
            "points": [
                {"method": method, "x": float(point[0]), "y": float(point[1])}
                for method, point in zip(methods, positions, strict=True)
            ],
            "explained_variance": explained.tolist(),
        },
    }
