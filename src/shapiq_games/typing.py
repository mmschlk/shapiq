"""The types of :mod:`shapiq_games`, next to the core ones in :mod:`shapiq.typing`.

Coalitions and game values use the core types :data:`shapiq.typing.CoalitionMatrix` and
:data:`shapiq.typing.GameValues`. This module holds the choices the games offer (as ``Literal``
types, which :mod:`shapiq_benchmark.setups` also uses to check its fields) and the shapes of the
functions they take.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from shapiq.typing import FloatVector

__all__ = [
    "ClipModel",
    "ClusterMethod",
    "ClusterScore",
    "ConfoundingMode",
    "Fill",
    "ImageModel",
    "ImputerName",
    "LossName",
    "MaskStrategy",
    "Metric",
    "MetricName",
    "PlayerSizes",
    "PredictFunction",
    "Task",
    "Uncertainty",
]

type Task = Literal["classification", "regression"]
"""The supervised learning task of a model or dataset."""

type PredictFunction = Callable[[np.ndarray], FloatVector]
"""Maps a ``(n_samples, n_features)`` matrix to one output per row."""

type Metric = Callable[[np.ndarray, np.ndarray], float]
"""Scores predictions as ``metric(y_true, y_pred)``; higher is better."""

type MetricName = Literal["accuracy", "r2", "neg_mse", "neg_mae"]
"""The built-in metrics of the games that train models."""

type LossName = Literal["mse", "mae"]
"""The built-in losses of :class:`~shapiq_games.TabularGlobalExplanation`; lower is better."""

type ImputerName = Literal["marginal", "conditional", "baseline"]
"""How :class:`~shapiq_games.TabularLocalExplanation` removes features."""

type Uncertainty = Literal["total", "aleatoric", "epistemic"]
"""The uncertainty :class:`~shapiq_games.TabularUncertaintyExplanation` explains."""

type PlayerSizes = Literal["uniform", "increasing", "random"]
"""How :class:`~shapiq_games.DatasetValuation` splits the training data into players."""

type ClusterMethod = Literal["kmeans", "agglomerative"]
"""The clustering algorithm of :class:`~shapiq_games.ClusterExplanation`."""

type ClusterScore = Literal["calinski_harabasz", "silhouette"]
"""The clustering score of :class:`~shapiq_games.ClusterExplanation`."""

type ConfoundingMode = Literal["signed", "abs", "sq"]
"""How the confounding games aggregate the bias of the treatment effect."""

type MaskStrategy = Literal["mask", "remove"]
"""How :class:`~shapiq_games.SentimentAnalysis` and the transformer image games remove tokens:
replace them with a mask token, or drop them from the sequence."""

type ImageModel = Literal[
    "vit_9_patches",
    "vit_16_patches",
    "vit_36_patches",
    "vit_144_patches",
    "dinov2_16_patches",
    "dinov2_20_patches",
    "dinov2_25_patches",
    "resnet_18",
]
"""The built-in classifiers of :class:`~shapiq_games.ImageClassifier`."""

type Fill = Literal["mean", "gray", "black", "blur"]
"""How the image games fill removed regions in image space."""

type ClipModel = Literal["clip_vit_b16", "clip_vit_b32"]
"""The CLIP models of :class:`~shapiq_games.ImageTextSimilarity`."""
