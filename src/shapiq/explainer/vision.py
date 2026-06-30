"""Vision Language Explainer for shapiq.

The :class:`VisionLanguageExplainer` explains vision-language model predictions using
Shapley interaction values. It wraps the :mod:`shapiq.imputer.vision` pipeline
(VisionImputerFactory → VisionLanguageGame) and exposes the standard
``Explainer`` API::

    from shapiq import Explainer

    explainer = Explainer(model, processor=processor)
    iv = explainer.explain(
        x={"image": image, "text": "a dog"},
        budget=1024,
    )
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

VisionLanguageExplainerIndices = ExplainerIndices
"""Valid index types for the VisionLanguageExplainer."""


class VisionLanguageExplainer(Explainer):
    """Vision Language Explainer for HuggingFace vision-language models (CLIP, SigLIP).

    Builds a :class:`~shapiq.imputer.vision.VisionLanguageGame` via the
    :class:`~shapiq.imputer.vision.VisionImputerFactory`, then uses a shapiq
    approximator to compute Shapley interaction values.

    The image and text prompt are passed at **explain time** via the ``x``
    parameter as a dict with keys ``"image"`` and ``"text"``.

    Usage via auto-dispatch::

        from shapiq import Explainer

        explainer = Explainer(
            model=clip_model,
            processor=processor,
        )
        iv = explainer.explain(
            x={"image": image, "text": "a black dog"},
            budget=2048,
        )

    Or directly::

        from shapiq.explainer.vision import VisionLanguageExplainer

        explainer = VisionLanguageExplainer(
            model=clip_model,
            processor=processor,
        )
        iv = explainer.explain(
            x={"image": image, "text": "a black dog"},
            budget=2048,
        )
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: Any = None,  # noqa: ANN401  # deprecated — use explain(x={...}) instead
        *,
        text: str | None = None,  # deprecated — pass via explain(x={"text": ...}) instead
        processor: Any | None = None,  # noqa: ANN401
        segmenter_config: SegmenterConfig | None = None,
        masker_config: MaskerConfig | None = None,
        batch_size: int = 64,
        class_index: int | None = None,
        approximator: (
            Literal["auto"] | TabularExplainerApproximators | Approximator[VisionLanguageExplainerIndices]
        ) = "auto",
        index: VisionLanguageExplainerIndices = "SV",
        max_order: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the VisionLanguageExplainer.

        Args:
            model: A HuggingFace vision-language model (e.g. ``CLIPModel``,
                ``SiglipModel``).

            data: Deprecated. Ignored for VLM explainers. Pass the image via
                ``explain(x={"image": img, "text": "..."})`` instead.

            text: Deprecated. Ignored for VLM explainers. Pass the text via
                ``explain(x={"text": "..."})`` instead.

            processor: The HuggingFace processor corresponding to ``model``.
                If ``None``, the explainer attempts to infer it from the model
                name or path.

            segmenter_config: Optional :class:`~shapiq.imputer.vision.SegmenterConfig`
                to override the default segmenter (``"patch"`` for ViT models).

            masker_config: Optional :class:`~shapiq.imputer.vision.MaskerConfig`
                to override the default masker (``"crossmodal_mean"``).

            batch_size: Number of coalitions evaluated per model call.
                Defaults to ``64``.

            class_index: Deprecated. Ignored for VLMs (similarity is a scalar).

            approximator: The approximator to use. Defaults to ``"auto"``,
                which auto-selects based on the index and max_order.

            index: The Shapley interaction index to compute. Defaults to
                ``"SV"``.

            max_order: Maximum interaction order. Defaults to ``1``.

            random_state: Random state for reproducibility.

            verbose: Whether to print progress information.
        """

        # Resolve processor if not provided
        if processor is None:
            processor = self._infer_processor(model)

        # Store configuration for lazy game building in explain_function
        self._vision_model = model
        self._vision_processor = processor
        self._segmenter_config = segmenter_config
        self._masker_config = masker_config
        self._batch_size = batch_size
        self._approximator_spec = approximator
        self._verbose = verbose
        self._random_state = random_state

        # Cached game from the most recent explain() call
        self._vision_game: VisionLanguageGame | None = None

        # Initialise the base Explainer.
        super().__init__(
            model=model,
            data=None,
            class_index=class_index,
            index=index,
            max_order=max_order,
        )

        # Approximator is NOT built here — n_players depends on the image
        # and is only known once explain() is called.

    # Public API

    @property
    def game(self) -> VisionLanguageGame:
        """The :class:`~shapiq.imputer.vision.VisionLanguageGame` from the
        last ``explain()`` call.

        Raises:
            RuntimeError: If ``explain()`` has not been called yet.
        """
        if self._vision_game is None:
            msg = (
                "No game available yet. Call ``explain(x={'image': ..., 'text': ...}, budget=...)`` "
                "first."
            )
            raise RuntimeError(msg)
        return self._vision_game

    @property
    def baseline_value(self) -> float:
        """The baseline (empty coalition) value from the last ``explain()``
        call.

        Raises:
            RuntimeError: If ``explain()`` has not been called yet.
        """
        return self.game.empty_value

    def explain_function(  # type: ignore[override]
        self,
        budget: int,
        *,
        x: dict | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> InteractionValues:
        """Explain the model's prediction with Shapley interaction values.

        The ``x`` parameter must be a dict with ``"image"`` and ``"text"``
        keys::

            explainer.explain(
                x={"image": PIL.Image.open("dog.jpg"), "text": "a black dog"},
                budget=2048,
            )

        Args:
            budget: Number of coalition evaluations for the approximator.

            x: A dict with keys ``"image"`` (``PIL.Image``, ``np.ndarray``,
                or ``torch.Tensor``) and ``"text"`` (``str``).

            random_state: Optional random state for reproducibility.

            **kwargs: Additional keyword arguments. Supports ``custom_masks``
                (``np.ndarray | None``) forwarded to ``CustomSegmenter``.

        Returns:
            InteractionValues containing the computed Shapley interactions.
        """
        if x is None or not isinstance(x, dict):
            msg = (
                "``x`` must be a dict with ``'image'`` and ``'text'`` keys, e.g. "
                "``explain(x={'image': img, 'text': 'a dog'}, budget=...)``."
            )
            raise TypeError(msg)

        image = x.get("image")
        text = x.get("text")
        if image is None or text is None:
            msg = "``x`` dict must contain both ``'image'`` and ``'text'`` keys."
            raise ValueError(msg)

        custom_masks = kwargs.get("custom_masks", None)

        # Build the game (VisionImputer + VisionLanguageGame) for this input
        game = self._build_game(image=image, text=text, custom_masks=custom_masks)
        self._vision_game = game

        # Build the approximator — n_players depends on the image/text
        n_players = game.n_players
        _random_state = random_state if random_state is not None else self._random_state
        approximator = setup_approximator(
            approximator=self._approximator_spec,
            index=self._index,
            max_order=self._max_order,
            n_players=n_players,
            random_state=_random_state,
        )
        self._approximator = approximator

        if _random_state is not None:
            approximator.set_random_state(random_state=_random_state)

        interaction_values = approximator(budget=budget, game=game)
        interaction_values.baseline_value = game.empty_value
        return interaction_values

    # Internal helpers

    def _build_game(
        self, image: Any, text: str, custom_masks: np.ndarray | None = None  # noqa: ANN401
    ) -> VisionLanguageGame:
        """Build a :class:`~shapiq.imputer.vision.VisionLanguageGame` for
        the given image and text.

        Args:
            image: Input image (``PIL.Image``, ``np.ndarray``, or ``torch.Tensor``).
            text: Text prompt.
            custom_masks: Optional binary masks of shape ``(N_players, H, W)``
                for use with ``CustomSegmenter``.

        Returns:
            A fully wired ``VisionLanguageGame``.
        """
        from shapiq.imputer.vision import (
            VisionImputerFactory,
            VisionLanguageGame,
        )

        factory = VisionImputerFactory()
        segmenter_kwargs: dict[str, Any] = {}
        if custom_masks is not None:
            segmenter_kwargs["masks"] = custom_masks
        imputer = factory.build(
            model=self._vision_model,
            processor=self._vision_processor,
            input_image=image,
            input_text=text,
            segmenter_config=self._segmenter_config,
            masker_config=self._masker_config,
            **segmenter_kwargs,
        )
        return VisionLanguageGame(
            imputer,
            batch_size=self._batch_size,
            verbose=self._verbose,
        )

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
