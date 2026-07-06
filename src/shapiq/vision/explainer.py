"""Explainer for vision models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import ValidApproximatorTypes, setup_approximator
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
ImageExplainerApproximators = ValidApproximatorTypes


class ImageExplainer(Explainer):
    """Explainer for vision models based on Shapley interaction values.

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
        model: ModelArchitectureStrategy,
        data: ImageLike,
        *,
        class_index: int | None = None,
        imputer: ImageImputer | None = None,
        approximator: (
            Literal["auto"] | ImageExplainerApproximators | Approximator[ImageExplainerIndices]
        ) = "auto",
        index: ExplainerIndices = "SV",
        max_order: int = 1,
        random_state: int | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize an image explainer.

        Args:
            model: A configured
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

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to the highest predicted class for the image.

            imputer: An already-constructed :class:`~shapiq.vision.imputer.ImageImputer`.

            approximator: An :class:`~shapiq.approximator.Approximator` object to use for the
                explainer or a literal string from
                ``["auto", "spex", "montecarlo", "svarm", "permutation", "regression", "proxyshap", "proxyspex"]``. Defaults to ``"auto"``
                which automatically selects a :class:`~shapiq.approximator.Approximator`
                based on the selected index and max_order.

            index: The Shapley-interaction index to compute. Any value accepted by
                :func:`~shapiq.explainer.configuration.setup_approximator` is
                valid, e.g.  ``SV``, ``"k-SII"``, ``"SII"``, ``"STI"``,
                ``"FSI"``, ``"FSII"``.

            max_order: Maximum interaction order to compute. Defaults to ``1``.

            random_state: Seed for the approximator's random number generator.
                Defaults to ``None``.

            batch_size: Number of coalitions forwarded to the model per call.
                Defaults to ``32``.

            **kwargs: Additional keyword arguments.
        """
        if isinstance(imputer, ImageImputer):
            _imputer: ImageImputer = imputer
        else:
            _imputer: ImageImputer = ImageImputer(
                model_architecture=model,
                image=data,
                batch_size=batch_size,
                class_index=class_index,
            )

        super().__init__(
            model=_imputer.value_function,
            data=None,
            class_index=class_index,
            index=index,
            max_order=max_order,
            **kwargs,
        )
        self._imputer: ImageImputer = _imputer

        self._n_features: int = self._imputer.n_features
        self._approximator_spec = approximator
        self._random_state = random_state

        self._approximator: Approximator = setup_approximator(
            approximator=approximator,
            index=index,
            max_order=self.max_order,
            n_players=self._n_features,
            random_state=random_state,
        )

    def explain_function(  # type: ignore[override]
        self,
        x: np.ndarray | None,
        budget: int,
        *,
        random_state: int | None = None,
    ) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x (np.ndarray | None): Image to be explained. If not passed, the explainer will use the image passed during initialization.
                Accepts PIL Image, numpy array (H, W, C) or (C, H, W), or a PyTorch tensor.

            budget (int): The budget to use for the approximation. It indicates how many coalitions are
                sampled, thus high values indicate more accurate approximations, but induce higher
                computational costs.

            random_state: The random state to re-initialize Imputer and Approximator with.
                Defaults to ``None``, which will not set a random state.

        Returns:
            An object of class :class:`~shapiq.interaction_values.InteractionValues` containing
            the computed interaction values.
        """
        if x is not None:
            self._imputer.fit(x)
            if self._imputer.n_features != self._n_features:
                self._rebuild_approximator()

        self.set_random_state(random_state)

        interaction_values = self._approximator.approximate(budget=budget, game=self._imputer)
        interaction_values.baseline_value = self.baseline_value

        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        return interaction_values

    def _rebuild_approximator(self) -> None:
        """Rebuild the approximator after the imputer's player count changed.

        The imputer is a fixed-size game, so refitting to a new image can change
        ``n_features`` (e.g. SLIC returns a different superpixel count). Reusing the
        original approximator would then desync it from the game.
        """
        from shapiq.approximator.base import Approximator

        if isinstance(self._approximator_spec, Approximator):
            msg = (
                "Cannot reuse this explainer across images with different player counts "
                f"({self._n_features} -> {self._imputer.n_features}): a pre-built approximator "
                "was supplied and cannot be resized. Construct a new ImageExplainer for this "
                "image, or pass approximator='auto' (or a string) so it can be rebuilt."
            )
            raise TypeError(msg)

        self._n_features = self._imputer.n_features
        self._approximator = setup_approximator(
            approximator=self._approximator_spec,
            index=self.index,
            max_order=self.max_order,
            n_players=self._n_features,
            random_state=self._random_state,
        )

    @property
    def baseline_value(self) -> float:
        """Return the model prediction for the empty coalition."""
        return self._imputer.empty_prediction

    @property
    def imputer(self) -> ImageImputer:
        """The image imputer used by this explainer."""
        return self._imputer
