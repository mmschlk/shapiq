from __future__ import annotations

from typing import Any

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import setup_approximator
from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.game_theory.indices import is_empty_value_the_baseline
from shapiq.interaction_values import InteractionValues

from .imputer import ImageImputer

from .architecture import ModelArchitectureStrategy
from .utils import ImageLike

ImageExplainerIndices = ExplainerIndices


class ImageExplainer(Explainer):
    """Explainer for vision models."""

    def __init__(
        self,
        model_architecture: ModelArchitectureStrategy,
        data: ImageLike,
        *,
        imputer: ImageImputer | None = None,
        index: ExplainerIndices = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model_architecture.model, data=None, index=index, max_order=max_order)

        if isinstance(imputer, ImageImputer):
            self._imputer = imputer
        else:
            self._imputer = ImageImputer(
                model_architecture=model_architecture,
                image=data,
                batch_size=batch_size,
            )
            
        self._n_features: int = self._imputer.n_features

        self._approximator = setup_approximator(
            approximator="auto",
            index=index,
            max_order=self.max_order,
            n_players=self._n_features,
            random_state=random_state,
        )

    def explain_function(
        self, x: ImageLike| None, *, budget: int = 64
    ) -> InteractionValues:      
        interaction_values = self._approximator.approximate(budget=budget, game=self._imputer)
        interaction_values.baseline_value = self.baseline_value
        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        return interaction_values

    @property
    def baseline_value(self) -> float:
        return self._imputer.empty_prediction