"""CLIP and SigLIP image and text embeddings with players removed in token space.

Needs torch and transformers; the SigLIP tokenizers also need sentencepiece and protobuf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from shapiq_games._optional import require

from ._batching import pad_batch
from ._preprocess import displayed_image, normalized_pixels
from ._regions import pixel_regions, token_players
from ._token_removal import token_remover

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix
    from shapiq_games.typing import ImageTextModel, MaskStrategy

__all__ = [
    "IMAGENET_LABEL_SOURCE",
    "IMAGE_TEXT_MODEL_IDS",
    "ImageTextTokenModel",
    "imagenet_class_names",
]

IMAGE_TEXT_MODEL_IDS: dict[str, str] = {
    "clip_vit_b16": "openai/clip-vit-base-patch16",
    "clip_vit_b32": "openai/clip-vit-base-patch32",
    "siglip_vit_b16": "google/siglip-base-patch16-224",
    "siglip2_vit_b16": "google/siglip2-base-patch16-224",
}
"""The builtin image-text models by name: CLIP ViT-B/16 (14 x 14 tokens) and ViT-B/32 (7 x 7), and
SigLIP and SigLIP 2 ViT-B/16 (14 x 14 tokens), all on ``224 x 224`` images."""

IMAGENET_LABEL_SOURCE = "facebook/dinov2-base-imagenet1k-1-layer"
"""The Hugging Face config the ImageNet-1k class names are read from (no weights are loaded)."""


def imagenet_class_names() -> list[str]:
    """Return the 1,000 ImageNet class names (the first synonym of each, e.g. ``"tench"``)."""
    transformers = require("transformers", purpose="the zero-shot labels")
    id2label = transformers.AutoConfig.from_pretrained(IMAGENET_LABEL_SOURCE).id2label
    return [id2label[i].split(",")[0].strip() for i in range(len(id2label))]


class ImageTextTokenModel:
    """CLIP or SigLIP on a ``224 x 224`` image, with absent players' patch tokens removed.

    The players are near-equal rectangular blocks of the model's patch tokens. The tokens of
    absent players are dropped from the sequence (``"remove"``; the present ones keep their
    position embeddings) or masked (``"mask"``). Neither model has a mask token, so a masked
    patch is the embedding of a patch at the normalization mean: CLIP's patch embedding has no
    bias, so only the position embedding remains; SigLIP's mean is gray. CLIP keeps its class
    token when tokens are dropped; SigLIP has none and pools its tokens by attention, so with no
    token left it returns the pooling's output bias, the model's embedding of no image.
    :meth:`filled_embeddings` embeds the image with the absent players' pixels filled instead,
    for removal in image space. The model sees exactly the processor's pixels: CLIP a center
    crop, as in the paper's script, SigLIP the whole image resized to ``224 x 224``.

    Attributes:
        image: The ``224 x 224`` image the model sees.
        regions: The player of every pixel of :attr:`image`.
    """

    def __init__(
        self,
        image: np.ndarray,
        grid: tuple[int, int],
        *,
        model: ImageTextModel = "clip_vit_b16",
        mask_strategy: MaskStrategy = "remove",
        device: str = "cpu",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        """Load the model and embed the image's tokens once.

        Args:
            image: The RGB image of shape ``(height, width, 3)``.
            grid: The ``(rows, columns)`` of the player grid over the token grid.
            model: ``"clip_vit_b16"`` (default), ``"clip_vit_b32"``, ``"siglip_vit_b16"``, or
                ``"siglip2_vit_b16"``.
            mask_strategy: ``"remove"`` (default, as in the paper) or ``"mask"``.
            device: The torch device. Defaults to ``"cpu"``.
            batch_size: The number of coalitions per forward pass. Defaults to ``16``.
            revision: The Hugging Face revision (branch, tag, or commit) of the model. ``None``
                loads the default branch.
        """
        if model not in IMAGE_TEXT_MODEL_IDS:
            msg = (
                f"Unknown image-text model {model!r}. Choose one of {sorted(IMAGE_TEXT_MODEL_IDS)}."
            )
            raise ValueError(msg)
        torch = require("torch", purpose="the image-text games")
        transformers = require("transformers", purpose="the image-text games")
        self._torch, self.batch_size = torch, batch_size
        self._device = torch.device(device)
        model_id = IMAGE_TEXT_MODEL_IDS[model]
        self._siglip = model.startswith("siglip")
        if self._siglip:
            # the fast image processor, transformers 5's default, pinned: its pixels can differ
            # from the slow one's, and naming it silences the warning about the changed default
            processor = transformers.SiglipProcessor.from_pretrained(
                model_id, revision=revision, use_fast=True
            )
            self._model = transformers.SiglipModel.from_pretrained(model_id, revision=revision)
        else:
            processor = transformers.CLIPProcessor.from_pretrained(model_id, revision=revision)
            self._model = transformers.CLIPModel.from_pretrained(model_id, revision=revision)
        self._tokenizer = processor.tokenizer
        self._model.eval().to(self._device)
        image_processor = processor.image_processor
        self._mean, self._std = image_processor.image_mean, image_processor.image_std
        # the model sees exactly the processor's pixels (resized, for CLIP also cropped, in
        # float), as in the paper's script; the game's image shows them as uint8
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        self._pixels = processor(images=pil, return_tensors="pt")["pixel_values"].to(self._device)
        self.image = displayed_image(self._pixels, self._mean, self._std)
        vision = self._model.vision_model
        if self._siglip:
            self._embed, n_prefix = vision.embeddings, 0

            def pool(hidden: Any) -> Any:  # noqa: ANN401
                return vision.head(vision.post_layernorm(hidden))

        else:
            # the layer norm before the encoder acts per token, so it commutes with dropping
            self._embed, n_prefix = lambda pixels: vision.pre_layrnorm(vision.embeddings(pixels)), 1

            def pool(hidden: Any) -> Any:  # noqa: ANN401
                return self._model.visual_projection(vision.post_layernorm(hidden[:, 0]))

        with torch.no_grad():
            embeddings = self._embed(self._pixels)
            # a masked patch has no content: the pixels of the normalization mean
            masked = self._embed(torch.zeros_like(self._pixels))
        side = round((embeddings.shape[1] - n_prefix) ** 0.5)
        rows, cols = grid
        if not (1 <= rows <= side and 1 <= cols <= side):
            msg = f"grid must have 1 to {side} rows and columns for {model}, got {grid}."
            raise ValueError(msg)
        players = token_players(side, rows, cols)
        self.regions = pixel_regions(players, self._model.config.vision_config.patch_size)

        def encode(tokens: Any) -> Any:  # noqa: ANN401
            features = pool(vision.encoder(inputs_embeds=tokens).last_hidden_state)
            return features / features.norm(dim=-1, keepdim=True)

        self._encode = encode
        self._remover = token_remover(
            mask_strategy, torch, embeddings, masked, players, encode, batch_size, n_prefix=n_prefix
        )

    def image_embeddings(self, coalitions: CoalitionMatrix) -> np.ndarray:
        """Return the unit-length image embedding of each coalition."""
        return self._remover(coalitions)

    def filled_embeddings(self, coalitions: CoalitionMatrix, fill: np.ndarray) -> np.ndarray:
        """Return the unit-length embedding of the image with each coalition's absent players filled.

        The present pixels are the processor's, so the full coalition is the plain model.

        Args:
            coalitions: The boolean coalitions, of shape ``(n_coalitions, n_players)``.
            fill: The fill, an RGB ``uint8`` image of :attr:`image`'s shape.

        Returns:
            The embeddings, of shape ``(n_coalitions, dim)``.
        """
        torch = self._torch
        fill_pixels = normalized_pixels(torch, fill, self._mean, self._std, self._device)
        player = torch.as_tensor(self.regions, device=self._device)
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
        """Return the unit-length embedding of each text.

        SigLIP reads texts as the models were trained: lowercased (the SigLIP 2 tokenizer of
        transformers 5.3 does not lowercase itself) and padded or truncated to 64 tokens.
        """
        torch = self._torch
        if self._siglip:
            texts = [text.lower() for text in texts]
            length = self._model.config.text_config.max_position_embeddings
            options: dict[str, Any] = {
                "padding": "max_length",
                "max_length": length,
                "truncation": True,
            }
        else:
            options = {"padding": True}
        embeddings = []
        for start in range(0, len(texts), batch_size):
            inputs = self._tokenizer(
                texts[start : start + batch_size], return_tensors="pt", **options
            ).to(self._device)
            with torch.no_grad():
                features = self._model.text_model(**inputs).pooler_output
                if not self._siglip:  # SigLIP's text head is part of its text model
                    features = self._model.text_projection(features)
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.float().cpu().numpy().astype(float))
        return np.concatenate(embeddings)
