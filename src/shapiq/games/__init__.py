"""Game construction abstractions."""

from __future__ import annotations

from shapiq.games._algebra import SumGame
from shapiq.games._base import Game, LinkFunction, Model
from shapiq.games._callable import CallableGame
from shapiq.games._masked import MaskedGame
from shapiq.games._masked_predictor import MaskedPredictor, ModelMaskedPredictor
from shapiq.games._measures import (
    Measure,
    product_measure,
    soft_shapley_measure,
    uniform_measure,
)
from shapiq.games._parametric import ParametricGame
from shapiq.games._projection import (
    all_coalitions,
    banzhaf_values,
    fidelity,
    project,
    shapley_values,
    to_basis,
)
from shapiq.games._values import to_values
from shapiq.games.maskers import (
    BaselineMasker,
    Masker,
    SuperpixelMasker,
    TokenMasker,
    grid_labels,
)

__all__ = [
    "BaselineMasker",
    "CallableGame",
    "Game",
    "LinkFunction",
    "MaskedGame",
    "MaskedPredictor",
    "Masker",
    "Measure",
    "Model",
    "ModelMaskedPredictor",
    "ParametricGame",
    "SumGame",
    "SuperpixelMasker",
    "TokenMasker",
    "all_coalitions",
    "banzhaf_values",
    "fidelity",
    "grid_labels",
    "product_measure",
    "project",
    "shapley_values",
    "soft_shapley_measure",
    "to_basis",
    "to_values",
    "uniform_measure",
]
