"""VisionImputer — Core orchestration engine.

Coordinates Segmenter → Masker → Model pipeline.
Handles batching, device placement, and post-processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import PhysicalMask, ProcessorOutput, SpatialLayout, safe_processor_call

if TYPE_CHECKING:
    from typing import Protocol

    import numpy as np
    import PIL.Image
    from transformers import ProcessorMixin

    from shapiq.typing import Model

    from .maskers.base import Masker
    from .segmenters.base import Segmenter

    class _ModelOutput(Protocol):
        logits_per_image: torch.Tensor


class VisionImputer:
    """Core orchestration container for vision-language model explanation.

    Lifecycle:
        1. __init__(model, processor, segmenter, masker, inputs_original, ...)
        2. forward_1d(coalitions, batch_size) → np.ndarray
        3. forward_crossmodal(coalitions_img, coalitions_txt, batch_size) → np.ndarray
    """

    def __init__(
        self,
        model: Model,
        processor: ProcessorMixin,
        segmenter: Segmenter,
        masker: Masker,
        inputs_original: ProcessorOutput,
        inputs_raw: dict | None = None,
        *,
        input_image: PIL.Image.Image | np.ndarray | None = None,
        input_text: str = "",
    ) -> None:
        """Initialize the vision-language imputer.

        Args:
            model: HuggingFace VLM model (CLIP, SigLIP, etc.).
            processor: HuggingFace processor for the model.
            segmenter: Spatial division strategy.
            masker: Feature occlusion strategy.
            inputs_original: Preprocessed original inputs.
            inputs_raw: Raw processor output dict.
            input_image: Original PIL image or ndarray.
            input_text: Original input text.
        """
        self.model = model
        self.processor = processor
        self.segmenter = segmenter
        self.masker = masker

        self.inputs_original = inputs_original
        self.inputs_raw = inputs_raw or {}
        self.input_image = input_image
        self.input_text = input_text

        self.layout: SpatialLayout = segmenter.get_layout()
        self.image_size = self.layout.image_size
        self.patch_size = self.layout.patch_size
        self.n_channels = self.layout.n_channels
        self.grid_size = self.layout.grid_size
        self.model_type = self.layout.model_type

    @property
    def n_players_image(self) -> int:
        """Number of image players (patches/superpixels)."""
        return self.layout.n_players_image

    @property
    def n_players_text(self) -> int:
        """Number of text players (tokens)."""
        return self.layout.n_players_text

    @property
    def n_players(self) -> int:
        """Total number of players (image + text)."""
        return self.n_players_image + self.n_players_text

    # Public API

    def forward_1d(
        self,
        coalitions: np.ndarray,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate the model on a set of coalitions (1D input space)."""
        if batch_size is None:
            batch_size = coalitions.shape[0]

        n_coalitions = coalitions.shape[0]
        coalitions_image = coalitions[:, : self.n_players_image]
        coalitions_text = coalitions[:, self.n_players_image :]
        device = next(self.model.parameters()).device

        all_outputs = []
        for start in range(0, n_coalitions, batch_size):
            end = min(start + batch_size, n_coalitions)
            actual_batch = end - start

            inputs_batched = self._repeat_inputs(self.inputs_original, actual_batch, device=device)
            mask_slice = self.segmenter.generate_masks(
                coalitions_image=coalitions_image[start:end],
                coalitions_text=coalitions_text[start:end],
                device=device,
            )
            masked_inputs = self.masker.apply(inputs_batched, mask_slice)
            outputs = self._model_forward(masked_inputs)
            all_outputs.append(self._extract_diagonal(outputs))

        return torch.cat(all_outputs).cpu().numpy()

    def forward_crossmodal(
        self,
        coalitions_image: np.ndarray,
        coalitions_text: np.ndarray,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate cross-modal coalitions (2D: image x text)."""
        if batch_size is None:
            batch_size = max(coalitions_image.shape[0], coalitions_text.shape[0])

        n_img = coalitions_image.shape[0]
        n_txt = coalitions_text.shape[0]
        device = next(self.model.parameters()).device

        all_rows = []
        for img_start in range(0, n_img, batch_size):
            img_end = min(img_start + batch_size, n_img)
            img_bs = img_end - img_start

            img_slice_mask = self.segmenter.generate_masks(
                coalitions_image=coalitions_image[img_start:img_end],
                device=device,
            )
            inputs_img_batched = self._repeat_inputs(self.inputs_original, img_bs, device=device)
            inputs_img_masked = self.masker.apply(inputs_img_batched, img_slice_mask)

            col_outputs = []
            for txt_start in range(0, n_txt, batch_size):
                txt_end = min(txt_start + batch_size, n_txt)
                txt_bs = txt_end - txt_start

                txt_slice_mask = self.segmenter.generate_masks(
                    coalitions_text=coalitions_text[txt_start:txt_end],
                    device=device,
                )

                if txt_bs == img_bs:
                    masked = self.masker.apply(inputs_img_masked, txt_slice_mask)
                else:
                    img_only = PhysicalMask(image_binary_mask=img_slice_mask.image_binary_mask)
                    masked_img = self.masker.apply(inputs_img_masked, img_only)

                    kwargs: dict[str, Any] = {
                        "images": [self.input_image] * txt_bs,
                        "text": [self.input_text] * txt_bs,
                        "return_tensors": "pt",
                    }
                    if self.model_type in ("siglip", "siglip2"):
                        kwargs["padding"] = "max_length"
                        kwargs["max_length"] = 64
                    else:
                        kwargs["padding"] = True
                    text_raw = safe_processor_call(self.processor, **kwargs)
                    if "attention_mask" not in text_raw:
                        text_raw["attention_mask"] = (text_raw["input_ids"] != 1).long()

                    masked_img.input_ids = text_raw["input_ids"].to(device)
                    masked_img.attention_mask = text_raw["attention_mask"].to(device)
                    masked = self.masker.apply(masked_img, txt_slice_mask)

                outputs = self._model_forward(masked)
                col_outputs.append(outputs.logits_per_image.cpu())

            all_rows.append(torch.cat(col_outputs, dim=1))

        return torch.cat(all_rows, dim=0).cpu().numpy()

    # Internal helpers

    def _repeat_inputs(
        self,
        inputs: ProcessorOutput,
        batch_size: int,
        device: torch.device | None = None,
    ) -> ProcessorOutput:
        pixel_values = inputs.pixel_values
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        if device is not None:
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        return ProcessorOutput(
            pixel_values=pixel_values.expand(batch_size, -1, -1, -1),
            input_ids=input_ids.expand(batch_size, -1).clone(),
            attention_mask=attention_mask.expand(batch_size, -1).clone(),
            model_type=inputs.model_type,
        )

    def _model_forward(self, inputs: ProcessorOutput) -> _ModelOutput:
        device = next(self.model.parameters()).device
        inputs_dict = {k: v.to(device) for k, v in inputs.to_dict().items()}
        with torch.no_grad():
            return self.model(**inputs_dict)

    def _extract_diagonal(self, outputs: _ModelOutput) -> torch.Tensor:
        return torch.diagonal(outputs.logits_per_image).cpu()
