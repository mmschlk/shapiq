"""Patch masking of a Hugging Face vision transformer (requires ``torch`` and ``transformers``)."""

from __future__ import annotations

import numpy as np

from shapiq_games._optional import require

__all__ = ["VIT_MODEL_ID", "VIT_PATCH_GRIDS", "ViTPatchModel"]

VIT_MODEL_ID = "google/vit-base-patch32-384"
"""The Hugging Face model: a ViT-B/32 on 384x384 images, i.e. a 12x12 grid of patches."""

VIT_PATCH_GRIDS: dict[int, int] = {9: 3, 16: 4, 36: 6, 144: 12}
"""Number of players -> side length of the grid of super-patches (the model has 12x12 patches)."""

_N_PATCHES_PER_SIDE = 12


def _player_masks(n_players: int) -> np.ndarray:
    """Return, per player, the boolean mask of the 144 model patches it covers."""
    grid = VIT_PATCH_GRIDS[n_players]
    size = _N_PATCHES_PER_SIDE // grid
    masks = np.zeros((n_players, _N_PATCHES_PER_SIDE, _N_PATCHES_PER_SIDE), dtype=bool)
    for player in range(n_players):
        row, col = divmod(player, grid)
        masks[player, row * size : (row + 1) * size, col * size : (col + 1) * size] = True
    return masks.reshape(n_players, -1)


class ViTPatchModel:
    """A vision transformer whose patches are removed by replacing their embeddings with zeros.

    Players are square groups of the model's 12x12 patches (3x3, 4x4, 6x6, or all 12x12).
    Absent patches are masked through the transformer's ``bool_masked_pos`` with a zero mask
    token. The output is the class probability (softmax) of the explained class.

    Attributes:
        n_players: The number of players (super-patches).
        class_index: The explained class.
    """

    def __init__(
        self,
        image: np.ndarray,
        n_players: int,
        *,
        class_index: int | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        """Load the model and preprocess the image.

        Args:
            image: The RGB image of shape ``(height, width, 3)``.
            n_players: The number of players: 9, 16, 36, or 144.
            class_index: The explained class, or ``None`` for the class predicted on the image.
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
        self.n_players = n_players
        self.batch_size = batch_size
        self._device = torch.device(device)

        processor = transformers.ViTImageProcessor.from_pretrained(VIT_MODEL_ID, revision=revision)
        model = transformers.ViTForImageClassification.from_pretrained(
            VIT_MODEL_ID, revision=revision
        )
        model.eval().to(self._device)
        hidden_size = model.config.hidden_size
        # a zero mask token: masked patches lose their content but keep their position embedding
        model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, hidden_size, device=self._device)
        )
        self._model = model
        self._pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(self._device)
        self._player_masks = torch.as_tensor(_player_masks(n_players), device=self._device)

        probabilities = self._probabilities(np.ones((1, n_players), dtype=bool))[0]
        self.class_index = int(np.argmax(probabilities)) if class_index is None else class_index
        self.class_name = str(model.config.id2label[self.class_index])

    def _probabilities(self, coalitions: np.ndarray) -> np.ndarray:
        torch = self._torch
        outputs = []
        for start in range(0, coalitions.shape[0], self.batch_size):
            # a copy: torch rejects negative strides (a reversed view of the coalitions), and
            # np.ascontiguousarray keeps them for a single row, which NumPy deems contiguous
            batch = torch.as_tensor(
                np.array(coalitions[start : start + self.batch_size], dtype=bool),
                device=self._device,
            )
            # a model patch is masked unless a present player covers it
            covered = (batch.float() @ self._player_masks.float()) > 0
            with torch.no_grad():
                hidden = self._model.vit(
                    self._pixels.expand(batch.shape[0], -1, -1, -1), bool_masked_pos=~covered
                ).last_hidden_state
                logits = self._model.classifier(hidden[:, 0, :])
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the probability of the explained class for each coalition."""
        return self._probabilities(coalitions)[:, self.class_index]
