"""DINOv2 with token dropping as an image classifier (requires ``torch`` and ``transformers``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from shapiq_games._optional import require

from ._preprocess import displayed_image
from ._regions import pixel_regions, token_players
from ._token_removal import TokenDropper

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix

__all__ = ["DINOV2_GRIDS", "DINOV2_MODEL_ID", "DinoV2TokenModel"]

DINOV2_MODEL_ID = "facebook/dinov2-base-imagenet1k-1-layer"
"""The Hugging Face model: a frozen DINOv2 ViT-B/14 with a linear ImageNet-1k head."""

DINOV2_GRIDS: dict[int, tuple[int, int]] = {16: (4, 4), 20: (5, 4), 25: (5, 5)}
"""Number of players -> (rows, columns) of the player grid over the 16 x 16 token grid."""


class DinoV2TokenModel:
    """DINOv2 on a ``224 x 224`` center crop, with absent players' tokens dropped.

    The players are near-equal rectangular blocks of the 16 x 16 patch tokens. The tokens of absent
    players are dropped from the sequence; the present ones keep their position embeddings. The
    classifier reads the class token and the mean of the present patch tokens (zeros if there are
    none), as the ImageNet head of the model does. The outputs are the class probabilities
    (softmax). This is the game of the benchmarking paper; DINOv2 offers no masking or
    image-space removal (its checkpoint has no trained mask token, and the paper drops tokens).

    Attributes:
        image: The ``224 x 224`` crop the model sees.
        regions: The player of every pixel of :attr:`image`.
        n_players: The number of players.
        categories: The class names, by class index.
    """

    def __init__(
        self,
        image: np.ndarray,
        n_players: int,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        """Load the model and embed the image's tokens once.

        Args:
            image: The RGB image of shape ``(height, width, 3)``.
            n_players: The number of players: 16, 20, or 25.
            device: The torch device. Defaults to ``"cpu"``.
            batch_size: The number of coalitions per forward pass. Defaults to ``16``.
            revision: The Hugging Face revision (branch, tag, or commit) of the model. ``None``
                loads the default branch.
        """
        if n_players not in DINOV2_GRIDS:
            msg = f"n_players must be one of {sorted(DINOV2_GRIDS)}, got {n_players}."
            raise ValueError(msg)
        self.n_players = n_players
        torch = require("torch", purpose="the DINOv2 image games")
        transformers = require("transformers", purpose="the DINOv2 image games")
        device_ = torch.device(device)
        processor = transformers.AutoImageProcessor.from_pretrained(
            DINOV2_MODEL_ID, revision=revision
        )
        model = transformers.AutoModelForImageClassification.from_pretrained(
            DINOV2_MODEL_ID, revision=revision
        )
        model.eval().to(device_)
        # the model sees exactly the processor's pixels (resized and cropped in float), as in the
        # paper's script; the game's image shows them as uint8
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        pixels = processor(images=pil, return_tensors="pt")["pixel_values"].to(device_)
        self.image = displayed_image(pixels, processor.image_mean, processor.image_std)
        with torch.no_grad():
            embeddings = model.dinov2.embeddings(pixels)  # class token + patches, with positions
        side = round((embeddings.shape[1] - 1) ** 0.5)
        players = token_players(side, *DINOV2_GRIDS[n_players])
        self.regions = pixel_regions(players, model.config.patch_size)

        def encode(tokens: Any) -> Any:  # noqa: ANN401
            sequence = model.dinov2.layernorm(model.dinov2.encoder(tokens).last_hidden_state)
            cls = sequence[:, 0]
            patches = sequence[:, 1:]
            mean = patches.mean(dim=1) if patches.shape[1] > 0 else torch.zeros_like(cls)
            return torch.softmax(model.classifier(torch.cat((cls, mean), dim=1)), dim=-1)

        self._dropper = TokenDropper(
            torch, embeddings[:, :1], embeddings[0, 1:], players, encode, batch_size
        )
        self.categories = [str(model.config.id2label[i]) for i in range(len(model.config.id2label))]

    def __call__(self, coalitions: CoalitionMatrix) -> np.ndarray:
        """Return the class probabilities of each coalition, of shape ``(n_coalitions, n_classes)``."""
        return self._dropper(coalitions)
