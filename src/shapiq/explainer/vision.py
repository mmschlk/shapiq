"""Vision Explainer for shapiq.

The :class:`VisionExplainer` explains vision-language model predictions using
Shapley interaction values. It wraps the :mod:`shapiq.imputer.vision` pipeline
(VisionImputerFactory → VisionLanguageGame) and exposes the standard
``Explainer`` API::

    from shapiq import Explainer

    explainer = Explainer(model, data=image, text="a dog", processor=processor)
    iv = explainer.explain(budget=1024)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import setup_approximator
from shapiq.explainer.custom_types import ExplainerIndices

if TYPE_CHECKING:
    import numpy as np

    from shapiq.approximator.base import Approximator
    from shapiq.imputer.vision import (
        MaskerConfig,
        SegmenterConfig,
        VisionLanguageGame,
    )
    from shapiq.interaction_values import InteractionValues

    from .tabular import TabularExplainerApproximators

VisionExplainerIndices = ExplainerIndices
"""Valid index types for the VisionExplainer."""


class VisionExplainer(Explainer):
    """Vision Explainer for HuggingFace vision-language models (CLIP, SigLIP).

    Builds a :class:`~shapiq.imputer.vision.VisionLanguageGame` via the
    :class:`~shapiq.imputer.vision.VisionImputerFactory`, then uses a shapiq
    approximator to compute Shapley interaction values.

    Usage via auto-dispatch::

        from shapiq import Explainer

        explainer = Explainer(
            model=clip_model,
            data=image,           # PIL Image
            text="a black dog",   # required for VLMs
            processor=processor,
        )
        iv = explainer.explain(budget=2048)

    Or directly::

        from shapiq.explainer.vision import VisionExplainer

        explainer = VisionExplainer(
            model=clip_model,
            data=image,
            text="a black dog",
            processor=processor,
        )
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: Any = None,  # noqa: ANN401
        *,
        text: str = "",
        processor: Any | None = None,  # noqa: ANN401
        segmenter_config: SegmenterConfig | None = None,
        masker_config: MaskerConfig | None = None,
        batch_size: int = 64,
        class_index: int | None = None,
        approximator: (
            Literal["auto"] | TabularExplainerApproximators | Approximator[VisionExplainerIndices]
        ) = "auto",
        index: VisionExplainerIndices = "SV",
        max_order: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
        use_amp: bool = False,
    ) -> None:
        """Initializes the VisionExplainer.

        Args:
            model: A HuggingFace vision-language model (e.g. ``CLIPModel``,
                ``SiglipModel``).

            data: The input image. Accepted types: ``PIL.Image``,
                ``np.ndarray``, or ``torch.Tensor``.

            text: The text prompt for the VLM (e.g. ``"a photo of a dog"``).

            processor: The HuggingFace processor corresponding to ``model``.
                If ``None``, the explainer attempts to infer it from the model
                name or path.

            segmenter_config: Optional :class:`~shapiq.imputer.vision.SegmenterConfig`
                to override the default segmenter (``"patch"`` for ViT models).

            masker_config: Optional :class:`~shapiq.imputer.vision.MaskerConfig`
                to override the default masker (``"crossmodal_mean"``).

            batch_size: Number of coalitions evaluated per model call.
                Defaults to ``64``.

            class_index: The class index to explain. Ignored for VLMs
                (similarity is a scalar), kept for API compatibility.

            approximator: The approximator to use. Defaults to ``"auto"``,
                which auto-selects based on the index and max_order.

            index: The Shapley interaction index to compute. Defaults to
                ``"SV"``.

            max_order: Maximum interaction order. Defaults to ``1``.

            random_state: Random state for reproducibility.

            verbose: Whether to print progress information.

            use_amp: Whether to use automatic mixed precision (FP16) on CUDA.
        """
        from shapiq.imputer.vision import (
            VisionImputerFactory,
            VisionLanguageGame,
        )

        # 1. Build the VisionImputer pipeline
        if processor is None:
            processor = self._infer_processor(model)

        factory = VisionImputerFactory()
        imputer = factory.build(
            model=model,
            processor=processor,
            input_image=data,
            input_text=text,
            segmenter_config=segmenter_config,
            masker_config=masker_config,
            use_amp=use_amp,
        )

        # 2. Wrap as a shapiq Game
        self._vision_game: VisionLanguageGame = VisionLanguageGame(
            imputer,
            batch_size=batch_size,
            verbose=verbose,
        )

        # 3. Initialise base Explainer (pass the game as "model" — follows
        #    the AgnosticExplainer pattern so the base class does not try
        #    to call predict() on it).
        super().__init__(
            model=self._vision_game,
            data=None,
            class_index=class_index,
            index=index,
            max_order=max_order,
        )

        self._n_features: int = imputer.n_players

        # 4. Set up the approximator
        self._approximator = setup_approximator(
            approximator=approximator,
            index=self.index,
            max_order=self._max_order,
            n_players=self._n_features,
            random_state=random_state,
        )

    # ─── Public API ───────────────────────────────────────────────────────

    @property
    def game(self) -> VisionLanguageGame:
        """The underlying VisionLanguageGame."""
        return self._vision_game

    @property
    def baseline_value(self) -> float:
        """The baseline (empty coalition) value of the game."""
        return self._vision_game.empty_value

    def explain_function(  # type: ignore[override]
        self,
        budget: int,
        *,
        x: np.ndarray | None = None,  # noqa: ARG002
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Explain the model's prediction with Shapley interaction values.

        Args:
            budget: Number of coalition evaluations for the approximator.

            x: Ignored for vision models (the image is fixed at construction
                time). Kept for API compatibility.

            random_state: Optional random state for reproducibility.

            **kwargs: Additional keyword arguments (unused).

        Returns:
            InteractionValues containing the computed Shapley interactions.
        """
        self.set_random_state(random_state=random_state)
        interaction_values = self.approximator(budget=budget, game=self._vision_game)
        interaction_values.baseline_value = self.baseline_value
        return interaction_values

    # ─── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _infer_processor(model: Any) -> Any:  # noqa: ANN401
        """Attempt to auto-load the HuggingFace processor for a given model.

        Uses ``model.name_or_path`` if available, otherwise falls back to
        the model's config ``_name_or_path``.
        """
        from transformers import AutoProcessor

        model_id = getattr(model, "name_or_path", None)
        if not model_id:
            model_id = getattr(getattr(model, "config", None), "_name_or_path", None)
        if model_id:
            return AutoProcessor.from_pretrained(model_id)
        msg = (
            "Could not infer processor from model. Please pass a ``processor`` argument explicitly."
        )
        raise ValueError(msg)
