"""Explainer for vision models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import setup_approximator
from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.game_theory.indices import is_empty_value_the_baseline

from .imputer import ImageImputer

if TYPE_CHECKING:
    import numpy as np

    from shapiq.approximator.base import Approximator
    from shapiq.interaction_values import InteractionValues

    from .architecture import ModelArchitectureStrategy
    from .utils import ImageLike

ImageExplainerIndices = ExplainerIndices


class ImageExplainer(Explainer):
    """Explainer for vision models based on Shapley interaction values.

    Args:
        model_architecture: A configured
            :class:`~shapiq.vision.architecture.ModelArchitectureStrategy`
            (e.g. :class:`~shapiq.vision.architecture.CNNArchitecture` or
            :class:`~shapiq.vision.architecture.TransformerArchitecture`).
            This object owns the model, the player strategy, and the masking
            strategy. Sensible defaults are chosen automatically if no custom
            strategies are passed to the architecture constructor.
        data: The image to explain. Accepts a numpy ``(H, W, C)`` or
            ``(H, W)`` array, a PIL :class:`~PIL.Image.Image`, or a PyTorch
            ``(C, H, W)`` / ``(H, W, C)`` tensor. Batched 4-D inputs are not
            supported.
        imputer: An already-constructed :class:`~shapiq.vision.imputer.ImageImputer`.
        index: The Shapley-interaction index to compute. Any value accepted by
            :func:`~shapiq.explainer.configuration.setup_approximator` is
            valid, e.g. ``"k-SII"`` (default), ``"SII"``, ``"STI"``,
            ``"FSI"``, ``"FSII"``.
        max_order: Maximum interaction order to compute. Defaults to ``2``
            (pairwise interactions).
        random_state: Seed for the approximator's random number generator.
            Defaults to ``None``.
        batch_size: Number of coalitions forwarded to the model per call.
            Defaults to ``32``.
        **kwargs: Additional keyword arguments.

    Example:
        >>> from shapiq.vision.architecture import CNNArchitecture, TransformerArchitecture
        >>> from shapiq.vision.explainer import ImageExplainer

        >>> # --- CNN (ResNet-style) ---
        >>> arch = CNNArchitecture(model=my_resnet)
        >>> explainer = ImageExplainer(model_architecture=arch, data=my_image)
        >>> iv = explainer.explain_function(x=None, budget=256)

        >>> # --- ViT ---
        >>> arch = TransformerArchitecture(model=my_vit, vit_processor=processor)
        >>> explainer = ImageExplainer(model_architecture=arch, data=my_image,
                                   index="SII", max_order=2)
        >>> iv = explainer.explain_function(x=None, budget=512)

        .. note::
            PyTorch must be installed.
    """

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
        """Initialize an image explainer with a model architecture and imputer."""
        if isinstance(imputer, ImageImputer):
            _imputer: ImageImputer = imputer
        else:
            _imputer: ImageImputer = ImageImputer(
                model_architecture=model_architecture,
                image=data,
                batch_size=batch_size,
            )

        super().__init__(
            model=_imputer.value_function,
            data=None,
            index=index,
            max_order=max_order,
            **kwargs,
        )
        self._imputer: ImageImputer = _imputer

        self._n_features: int = self._imputer.n_features

        self._approximator: Approximator = setup_approximator(
            approximator="auto",
            index=index,
            max_order=self.max_order,
            n_players=self._n_features,
            random_state=random_state,
        )

    def explain_function(
        self,
        x: np.ndarray | None,
        *args: Any,  # noqa: ARG002 — required to match base class signature
        **kwargs: Any,
    ) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x (np.ndarray | None): Image to be explained. If not passed, the explainer will use the image passed during initialization.
            Accepts PIL Image, numpy array ``(H, W, C)`` or ``(C, H, W)``, or a PyTorch tensor.
            *args: Unused in this implementation.
            **kwargs: Optional keyword arguments. Supported keys:
                - ``budget`` (int): Maximum number of model evaluations.
                Defaults to ``64``.

        Returns:
            InteractionValues: The interaction values of the prediction.
        """
        budget: int = kwargs.get("budget", 64)
        if x is not None:
            self._imputer.fit(x)
        interaction_values = self._approximator.approximate(budget=budget, game=self._imputer)
        interaction_values.baseline_value = self.baseline_value

        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        return interaction_values

    @property
    def baseline_value(self) -> float:
        """Return the model prediction for the empty coalition."""
        return self._imputer.empty_prediction

    @property
    def imputer(self) -> ImageImputer:
        """The image imputer used by this explainer."""
        return self._imputer
