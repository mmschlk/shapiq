"""DINOv2 with players removed in token space as an image classifier (needs torch, transformers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from shapiq_games._optional import require

from ._batching import pad_batch
from ._preprocess import center_crop, normalized_pixels
from ._token_removal import pixel_regions, token_players, token_remover

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import MaskStrategy

__all__ = ["DINOV2_GRIDS", "DINOV2_MODEL_ID", "DinoV2TokenModel"]

DINOV2_MODEL_ID = "facebook/dinov2-base-imagenet1k-1-layer"
"""The Hugging Face model: a frozen DINOv2 ViT-B/14 with a linear ImageNet-1k head."""

DINOV2_GRIDS: dict[int, tuple[int, int]] = {16: (4, 4), 20: (5, 4), 25: (5, 5)}
"""Number of players -> (rows, columns) of the player grid over the 16 x 16 token grid."""


class DinoV2TokenModel:
    """DINOv2 on a ``224 x 224`` center crop, with absent players removed in token space.

    The players are near-equal rectangular blocks of the 16 x 16 patch tokens. The tokens of absent
    players are dropped from the sequence (``"remove"``; the present ones keep their position
    embeddings) or masked (``"mask"``: the model's mask token, which is zeros in this checkpoint,
    plus the position embedding). The classifier reads the class token and the mean of the patch
    tokens in the sequence (zeros if there are none), as the ImageNet head of the model does.
    :meth:`classify` runs the model on whole crops instead, for removal in image space. The output
    is the softmax probability of the explained class.

    Attributes:
        image: The ``224 x 224`` crop the model sees.
        regions: The player of every pixel of :attr:`image`.
        class_index: The explained class.
        class_name: The name of the explained class.
    """

    def __init__(
        self,
        image: np.ndarray,
        n_players: int,
        *,
        mask_strategy: MaskStrategy = "remove",
        class_index: int | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        """Load the model and embed the image's tokens once.

        Args:
            image: The RGB image of shape ``(height, width, 3)``.
            n_players: The number of players: 16, 20, or 25.
            mask_strategy: ``"remove"`` (default, as in the paper) or ``"mask"``.
            class_index: The explained class, or ``None`` for the class predicted on the image.
            device: The torch device. Defaults to ``"cpu"``.
            batch_size: The number of coalitions per forward pass. Defaults to ``16``.
            revision: The Hugging Face revision (branch, tag, or commit) of the model. ``None``
                loads the default branch.
        """
        if n_players not in DINOV2_GRIDS:
            msg = f"n_players must be one of {sorted(DINOV2_GRIDS)}, got {n_players}."
            raise ValueError(msg)
        torch = require("torch", purpose="the DINOv2 image games")
        transformers = require("transformers", purpose="the DINOv2 image games")
        self._torch, self.batch_size = torch, batch_size
        self._device = device_ = torch.device(device)
        processor = transformers.AutoImageProcessor.from_pretrained(
            DINOV2_MODEL_ID, revision=revision, use_fast=True
        )
        model = transformers.AutoModelForImageClassification.from_pretrained(
            DINOV2_MODEL_ID, revision=revision
        )
        model.eval().to(device_)
        self.image = center_crop(
            image,
            processor.size["shortest_edge"],
            processor.crop_size["height"],
            Image.Resampling(int(processor.resample)),
        )
        self._mean, self._std = processor.image_mean, processor.image_std
        pixels = self._pixels(self.image[None])
        with torch.no_grad():
            embeddings = model.dinov2.embeddings(pixels)  # class token + patches, with positions
            all_masked = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=device_)[:, 1:]
            masked = model.dinov2.embeddings(pixels, bool_masked_pos=all_masked)
        self._embed = model.dinov2.embeddings
        side = round((embeddings.shape[1] - 1) ** 0.5)
        players = token_players(side, *DINOV2_GRIDS[n_players])
        self.regions = pixel_regions(players, model.config.patch_size)

        def encode(tokens: Any) -> Any:  # noqa: ANN401
            sequence = model.dinov2.layernorm(model.dinov2.encoder(tokens).last_hidden_state)
            cls = sequence[:, 0]
            patches = sequence[:, 1:]
            mean = patches.mean(dim=1) if patches.shape[1] > 0 else torch.zeros_like(cls)
            return torch.softmax(model.classifier(torch.cat((cls, mean), dim=1)), dim=-1)

        self._encode = encode
        self._remover = token_remover(
            mask_strategy, torch, embeddings, masked, players, encode, batch_size
        )
        probabilities = self._remover(np.ones((1, n_players), dtype=bool))[0]
        self.class_index = int(np.argmax(probabilities)) if class_index is None else class_index
        self.class_name = str(model.config.id2label[self.class_index])

    def _pixels(self, images: np.ndarray) -> Any:  # noqa: ANN401
        torch = self._torch
        return torch.cat(
            [
                normalized_pixels(torch, image, self._mean, self._std, self._device)
                for image in images
            ]
        )

    def classify(self, images: np.ndarray) -> np.ndarray:
        """Return the class probabilities of whole crops, of shape ``(n_images, n_classes)``."""
        torch = self._torch
        outputs = []
        for start in range(0, images.shape[0], self.batch_size):
            chunk = np.asarray(images[start : start + self.batch_size])
            with torch.no_grad():
                pixels = self._pixels(pad_batch(chunk, self.batch_size))
                probabilities = self._encode(self._embed(pixels))
            outputs.append(probabilities.float().cpu().numpy()[: chunk.shape[0]])
        return np.concatenate(outputs).astype(float)

    def __call__(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the probability of the explained class for each coalition."""
        return self._remover(coalitions)[:, self.class_index]
