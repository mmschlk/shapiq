"""CLIP image and text embeddings with players removed in token space (needs torch, transformers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from shapiq_games._optional import require

from ._batching import pad_batch
from ._preprocess import displayed_image, normalized_pixels
from ._token_removal import pixel_regions, token_players, token_remover

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix
    from shapiq_games.typing import MaskStrategy

__all__ = ["CLIP_MODEL_IDS", "IMAGENET_LABEL_SOURCE", "ClipTokenModel", "imagenet_class_names"]

CLIP_MODEL_IDS: dict[str, str] = {
    "clip_vit_b16": "openai/clip-vit-base-patch16",
    "clip_vit_b32": "openai/clip-vit-base-patch32",
}
"""The builtin CLIP models by name: ViT-B/16 (14 x 14 tokens) and ViT-B/32 (7 x 7 tokens)."""

IMAGENET_LABEL_SOURCE = "facebook/dinov2-base-imagenet1k-1-layer"
"""The Hugging Face config the ImageNet-1k class names are read from (no weights are loaded)."""


def imagenet_class_names() -> list[str]:
    """Return the 1,000 ImageNet class names (the first synonym of each, e.g. ``"tench"``)."""
    transformers = require("transformers", purpose="the CLIP zero-shot labels")
    id2label = transformers.AutoConfig.from_pretrained(IMAGENET_LABEL_SOURCE).id2label
    return [id2label[i].split(",")[0].strip() for i in range(len(id2label))]


class ClipTokenModel:
    """CLIP on a ``224 x 224`` center crop, with absent players' patch tokens dropped.

    The players are near-equal rectangular blocks of CLIP's patch tokens. The tokens of absent
    players are dropped from the sequence (``"remove"``; the present ones keep their position
    embeddings) or masked (``"mask"``: CLIP has no mask token, so a masked patch keeps only its
    position embedding). :meth:`filled_embeddings` embeds the crop with the absent players'
    pixels filled instead, for removal in image space. The model sees exactly the processor's
    pixels, as in the paper's script.

    Attributes:
        image: The ``224 x 224`` crop the model sees.
        regions: The player of every pixel of :attr:`image`.
    """

    def __init__(
        self,
        image: np.ndarray,
        grid: tuple[int, int],
        *,
        model: str = "clip_vit_b16",
        mask_strategy: MaskStrategy = "remove",
        device: str = "cpu",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        """Load the model and embed the image's tokens once.

        Args:
            image: The RGB image of shape ``(height, width, 3)``.
            grid: The ``(rows, columns)`` of the player grid over the token grid.
            model: ``"clip_vit_b16"`` (default) or ``"clip_vit_b32"``.
            mask_strategy: ``"remove"`` (default, as in the paper) or ``"mask"``.
            device: The torch device. Defaults to ``"cpu"``.
            batch_size: The number of coalitions per forward pass. Defaults to ``16``.
            revision: The Hugging Face revision (branch, tag, or commit) of the model. ``None``
                loads the default branch.
        """
        if model not in CLIP_MODEL_IDS:
            msg = f"Unknown CLIP model {model!r}. Choose one of {sorted(CLIP_MODEL_IDS)}."
            raise ValueError(msg)
        torch = require("torch", purpose="the CLIP image games")
        transformers = require("transformers", purpose="the CLIP image games")
        self._torch, self.batch_size = torch, batch_size
        self._device = torch.device(device)
        model_id = CLIP_MODEL_IDS[model]
        processor = transformers.CLIPProcessor.from_pretrained(model_id, revision=revision)
        self._tokenizer = processor.tokenizer
        self._model = transformers.CLIPModel.from_pretrained(model_id, revision=revision)
        self._model.eval().to(self._device)
        image_processor = processor.image_processor
        self._mean, self._std = image_processor.image_mean, image_processor.image_std
        # the model sees exactly the processor's pixels (resized and cropped in float), as in the
        # paper's script; the game's image shows them as uint8
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        self._pixels = processor(images=pil, return_tensors="pt")["pixel_values"].to(self._device)
        self.image = displayed_image(self._pixels, self._mean, self._std)
        vision = self._model.vision_model
        self._embed = lambda pixels: vision.pre_layrnorm(vision.embeddings(pixels))
        with torch.no_grad():
            # the layer norm before the encoder acts per token, so it commutes with dropping
            embeddings = self._embed(self._pixels)
            # a masked patch has no content: its position embedding alone
            positions = vision.embeddings.position_embedding(vision.embeddings.position_ids)
            masked = vision.pre_layrnorm(positions)
        side = round((embeddings.shape[1] - 1) ** 0.5)
        rows, cols = grid
        if not (1 <= rows <= side and 1 <= cols <= side):
            msg = f"grid must have 1 to {side} rows and columns for {model}, got {grid}."
            raise ValueError(msg)
        players = token_players(side, rows, cols)
        self.regions = pixel_regions(players, self._model.config.vision_config.patch_size)

        def encode(tokens: Any) -> Any:  # noqa: ANN401
            pooled = vision.post_layernorm(
                vision.encoder(inputs_embeds=tokens).last_hidden_state[:, 0]
            )
            features = self._model.visual_projection(pooled)
            return features / features.norm(dim=-1, keepdim=True)

        self._encode = encode
        self._remover = token_remover(
            mask_strategy, torch, embeddings, masked, players, encode, batch_size
        )

    def image_embeddings(self, coalitions: CoalitionMatrix) -> np.ndarray:
        """Return the unit-length image embedding of each coalition."""
        return self._remover(coalitions)

    def filled_embeddings(
        self, coalitions: CoalitionMatrix, regions: np.ndarray, fill: np.ndarray
    ) -> np.ndarray:
        """Return the unit-length embedding of the crop with each coalition's absent players filled.

        The present pixels are the processor's, so the full coalition is the plain model.

        Args:
            coalitions: The boolean coalitions, of shape ``(n_coalitions, n_players)``.
            regions: The player of every pixel of :attr:`image`.
            fill: The fill, an RGB ``uint8`` image of :attr:`image`'s shape.

        Returns:
            The embeddings, of shape ``(n_coalitions, dim)``.
        """
        torch = self._torch
        fill_pixels = normalized_pixels(torch, fill, self._mean, self._std, self._device)
        player = torch.as_tensor(np.asarray(regions), device=self._device)
        outputs = []
        for start in range(0, coalitions.shape[0], self.batch_size):
            chunk = np.array(coalitions[start : start + self.batch_size], dtype=bool)
            present = torch.as_tensor(pad_batch(chunk, self.batch_size), device=self._device)
            pixels = torch.where(present[:, player][:, None], self._pixels, fill_pixels)
            with torch.no_grad():
                features = self._encode(self._embed(pixels))
            outputs.append(features.float().cpu().numpy()[: chunk.shape[0]])
        return np.concatenate(outputs).astype(float)

    def text_embeddings(self, texts: list[str], *, batch_size: int = 256) -> np.ndarray:
        """Return the unit-length embedding of each text."""
        torch = self._torch
        embeddings = []
        for start in range(0, len(texts), batch_size):
            inputs = self._tokenizer(
                texts[start : start + batch_size], padding=True, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                pooled = self._model.text_model(**inputs).pooler_output
                features = self._model.text_projection(pooled)
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.float().cpu().numpy().astype(float))
        return np.concatenate(embeddings)
