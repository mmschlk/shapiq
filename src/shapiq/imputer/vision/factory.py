"""VisionImputerFactory — Central assembly line.

Inspects the model, enriches SegmenterConfig with model metadata,
selects components, and returns a fully wired VisionImputer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import ProcessorOutput
from .imputer import VisionImputer
from .maskers import get_masker
from .maskers.base import MaskerConfig
from .segmenters import get_segmenter
from .segmenters.base import SegmenterConfig

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image
    from transformers import ProcessorMixin

    from shapiq.typing import Model

    from .maskers.base import Masker
    from .segmenters.base import Segmenter


class VisionImputerFactory:
    """Assembles the VisionImputer pipeline from typed configs.

    Usage::

        factory = VisionImputerFactory()
        imputer = factory.build(model, processor, input_image, input_text)

    Or with explicit config::

        seg_cfg = SegmenterConfig(strategy="slic", slic=SlicParams(n_segments=60))
        msk_cfg = MaskerConfig(strategy="crossmodal_mean")
        imputer = factory.build(model, processor, img, txt,
                                segmenter_config=seg_cfg,
                                masker_config=msk_cfg)
    """

    def build(
        self,
        model: Model,
        processor: ProcessorMixin,
        input_image: PIL.Image.Image | np.ndarray,
        input_text: str,
        segmenter_config: SegmenterConfig | None = None,
        masker_config: MaskerConfig | None = None,
        *,
        use_amp: bool = False,
        **segmenter_kwargs: Any,
    ) -> VisionImputer:
        """Assemble the VisionImputer pipeline from typed configs.

        Args:
            model: HuggingFace VLM model.
            processor: HuggingFace processor for the model.
            input_image: Input image (PIL, ndarray, or path).
            input_text: Input text.
            segmenter_config: Segmenter configuration (default: PatchSegmenter).
            masker_config: Masker configuration (default: cross-modal mean).
            use_amp: Enable mixed-precision inference on CUDA.
            **segmenter_kwargs: Extra kwargs forwarded to the segmenter.

        Returns:
            A fully wired VisionImputer instance.
        """
        if segmenter_config is None:
            segmenter_config = SegmenterConfig()
        if masker_config is None:
            masker_config = MaskerConfig()

        # 1. Infer model type
        model_type = self._infer_model_type(model)

        # 2. Extract model dimensions
        image_size, patch_size, n_channels = self._extract_vision_dims(model)
        is_vit = patch_size > 0
        grid_size = image_size // patch_size if is_vit else 0

        # 3. Preprocess once to determine text players
        inputs_dict = self._preprocess(processor, input_image, input_text, model_type)
        n_players_text = self._count_text_players(inputs_dict, model_type)
        text_total_length = inputs_dict["input_ids"].shape[1]

        # 4. Enrich SegmenterConfig with model metadata (common fields)
        segmenter_config.model_type = model_type
        segmenter_config.image_size = image_size
        segmenter_config.n_channels = n_channels
        segmenter_config.n_players_text = n_players_text
        segmenter_config.text_total_length = text_total_length

        # 5. Create Segmenter
        strategy = segmenter_config.strategy

        # Patch-specific enrichment
        if strategy == "patch":
            segmenter_config.patch_size = patch_size
            segmenter_config.grid_size = grid_size
            segmenter_config.n_players_image = grid_size**2 if grid_size > 0 else 0
        # Create segmenter
        if strategy == "slic":
            segmenter_kwargs.setdefault("image_array", input_image)
            segmenter = self._create_segmenter(segmenter_config, **segmenter_kwargs)
        else:
            segmenter = self._create_segmenter(segmenter_config, **segmenter_kwargs)

        segmenter_config.n_players_image = segmenter.get_layout().n_players_image

        # 6. Create Masker
        masker = self._create_masker(masker_config)

        # 7. Build standardised 1-sample inputs
        inputs_original = ProcessorOutput(
            pixel_values=inputs_dict["pixel_values"],
            input_ids=inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            model_type=model_type,
        )

        # 8. Assemble and return
        return VisionImputer(
            model=model,
            processor=processor,
            segmenter=segmenter,
            masker=masker,
            inputs_original=inputs_original,
            inputs_raw=inputs_dict,
            input_image=input_image,
            input_text=input_text,
            use_amp=use_amp,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _infer_model_type(model: Model) -> str:
        config = getattr(model, "config", None)
        candidates = [
            getattr(model, "name_or_path", ""),
            getattr(config, "_name_or_path", ""),
            getattr(config, "name_or_path", ""),
            getattr(config, "model_type", ""),
            type(model).__name__,
            type(config).__name__ if config is not None else "",
        ]
        normalized = [str(value).lower() for value in candidates if value]
        if any("siglip2" in value for value in normalized):
            return "siglip2"
        if any("siglip" in value for value in normalized):
            return "siglip"
        return "clip"

    @staticmethod
    def _extract_vision_dims(model: Model) -> tuple:
        vision = model.vision_model
        embeddings = getattr(vision, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "patch_size"):
            return (
                embeddings.image_size,
                embeddings.patch_size,
                embeddings.config.num_channels,
            )
        vc = getattr(model.config, "vision_config", None) or model.config
        image_size = int(getattr(vc, "image_size", 224))
        n_channels = int(getattr(vc, "num_channels", 3))
        return (image_size, 0, n_channels)

    @staticmethod
    def _preprocess(
        processor: ProcessorMixin,
        image: PIL.Image.Image | np.ndarray | str,
        text: str,
        model_type: str,
    ) -> dict:
        kwargs: dict[str, Any] = {"images": image, "text": text, "return_tensors": "pt"}
        if model_type in ("siglip", "siglip2"):
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = 64
        elif model_type == "clip":
            kwargs["padding"] = True
        outputs = processor(**kwargs)
        if "attention_mask" not in outputs:
            outputs["attention_mask"] = (outputs["input_ids"] != 1).long()
        return outputs

    @staticmethod
    def _count_text_players(inputs: dict, model_type: str) -> int:
        input_ids = inputs["input_ids"][0]
        if model_type == "siglip2":
            return input_ids.count_nonzero().item() - 1
        if model_type == "siglip":
            return (input_ids != 1).count_nonzero().item()
        if model_type == "clip":
            return input_ids.size(0) - 2
        return 0

    @staticmethod
    def _create_segmenter(config: SegmenterConfig, **extra_kwargs: Any) -> Segmenter:
        cls = get_segmenter(config.strategy)
        return cls(config=config, **extra_kwargs)

    @staticmethod
    def _create_masker(config: MaskerConfig) -> Masker:
        cls = get_masker(config.strategy)
        return cls(config=config)
