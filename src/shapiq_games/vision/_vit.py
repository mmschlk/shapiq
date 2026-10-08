"""A Hugging Face vision transformer with players removed in token space (needs torch, transformers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq_games._optional import require

from ._batching import pad_batch
from ._regions import grid_regions, token_players
from ._token_removal import token_remover

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix
    from shapiq_games.typing import MaskStrategy

__all__ = ["VIT_MODEL_ID", "VIT_PATCH_GRIDS", "ViTTokenModel"]

VIT_MODEL_ID = "google/vit-base-patch32-384"
"""The Hugging Face model: a ViT-B/32 on 384x384 images, i.e. a 12x12 grid of patches."""

VIT_PATCH_GRIDS: dict[int, int] = {9: 3, 16: 4, 36: 6, 144: 12}
"""Number of players -> side length of the grid of super-patches (the model has 12x12 patches)."""

_N_PATCHES_PER_SIDE = 12


class ViTTokenModel:
    """A vision transformer whose players are removed in token space.

    Players are square groups of the model's 12x12 patches (3x3, 4x4, 6x6, or all 12x12). The
    patch tokens of absent players are masked (``"mask"``: a zero mask token, the position
    embedding kept) or dropped from the sequence (``"remove"``). :meth:`classify` runs the model
    on whole images instead, for removal in image space. The outputs are the class probabilities
    (softmax).

    Attributes:
        image: The image, as given (the processor resizes it to 384 x 384).
        regions: The player of every pixel of :attr:`image`: the patch grid over the image.
        n_players: The number of players (super-patches).
        categories: The class names, by class index.
    """

    def __init__(
        self,
        image: np.ndarray,
        n_players: int,
        *,
        mask_strategy: MaskStrategy = "mask",
        device: str = "cpu",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        """Load the model and embed the image's patches once.

        Args:
            image: The RGB image of shape ``(height, width, 3)``.
            n_players: The number of players: 9, 16, 36, or 144.
            mask_strategy: ``"mask"`` (default) or ``"remove"``.
            device: The torch device. Defaults to ``"cpu"``.
            batch_size: The number of coalitions per forward pass. Defaults to ``16``.
            revision: The Hugging Face revision (branch, tag, or commit) of the model. ``None``
                loads the default branch.
        """
        if n_players not in VIT_PATCH_GRIDS:
            msg = f"n_players must be one of {sorted(VIT_PATCH_GRIDS)}, got {n_players}."
            raise ValueError(msg)
        torch = require("torch", purpose="the vision transformer games")
        transformers = require("transformers", purpose="the vision transformer games")
        self._torch = torch
        self.image = np.asarray(image)
        self.n_players = n_players
        self.batch_size = batch_size
        self._device = torch.device(device)

        self._processor = transformers.ViTImageProcessor.from_pretrained(
            VIT_MODEL_ID, revision=revision
        )
        model = transformers.ViTForImageClassification.from_pretrained(
            VIT_MODEL_ID, revision=revision
        )
        model.eval().to(self._device)
        # a zero mask token: masked patches lose their content but keep their position embedding
        model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, model.config.hidden_size, device=self._device)
        )
        self._model = model
        pixels = self._pixels(np.asarray(image)[None])
        all_masked = torch.ones(1, _N_PATCHES_PER_SIDE**2, dtype=torch.bool, device=self._device)
        with torch.no_grad():
            embeddings = model.vit.embeddings(pixels)
            masked = model.vit.embeddings(pixels, bool_masked_pos=all_masked)

        def encode(tokens: Any) -> Any:  # noqa: ANN401
            hidden = model.vit.layernorm(model.vit.encoder(tokens).last_hidden_state)
            return torch.softmax(model.classifier(hidden[:, 0]), dim=-1)

        grid = VIT_PATCH_GRIDS[n_players]
        height, width = self.image.shape[:2]
        self.regions = grid_regions(height, width, grid, grid)
        players = token_players(_N_PATCHES_PER_SIDE, grid, grid)
        self._remover = token_remover(
            mask_strategy, torch, embeddings, masked, players, encode, batch_size
        )
        self.categories = [str(model.config.id2label[i]) for i in range(len(model.config.id2label))]

    def _pixels(self, images: np.ndarray) -> Any:  # noqa: ANN401
        pixels = self._processor(images=list(images), return_tensors="pt")["pixel_values"]
        return pixels.to(self._device)

    def classify(self, images: np.ndarray) -> np.ndarray:
        """Return the class probabilities of whole images, of shape ``(n_images, n_classes)``."""
        torch = self._torch
        outputs = []
        for start in range(0, images.shape[0], self.batch_size):
            chunk = np.asarray(images[start : start + self.batch_size])
            with torch.no_grad():
                logits = self._model(pixel_values=self._pixels(pad_batch(chunk, self.batch_size)))
            probabilities = torch.softmax(logits.logits, dim=-1)
            outputs.append(probabilities.float().cpu().numpy()[: chunk.shape[0]])
        return np.concatenate(outputs).astype(float)

    def __call__(self, coalitions: CoalitionMatrix) -> np.ndarray:
        """Return the class probabilities of each coalition, of shape ``(n_coalitions, n_classes)``."""
        return self._remover(coalitions)
