"""Shared fixtures and helpers for the shapiq.vision test suite."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shapiq.vision.architecture import TransformerArchitecture
from shapiq.vision.masking import BoolMaskedPosStrategy
from shapiq.vision.players import CNNPlayerStrategy
from shapiq.vision.utils import to_tensor_chw


class FixedMasksStrategy(CNNPlayerStrategy):
    """A deterministic player strategy that returns user-provided spatial masks.

    Useful for correctness tests where SLIC's non-determinism would interfere.
    """

    def __init__(self, masks: np.ndarray) -> None:
        self._masks = masks.astype(bool)

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        return self._masks

    @property
    def n_players(self) -> int:
        return int(self._masks.shape[0])


class ChannelSumModel(torch.nn.Module):
    """A deterministic two-class CNN-like model.

    The class-0 logit equals the sum of all pixel intensities of the (masked)
    image; class-1 is its negation.  This makes the model output an exact linear
    function of the pixels that survive masking, which is what the correctness
    tests rely on.

    The model takes a ``(B, C, H, W)`` float tensor (as produced by
    :class:`~shapiq.vision.architecture.CNNArchitecture`) and returns a
    ``(B, 2)`` tensor.  The sum is accumulated in float64 so that comparisons
    against numpy float64 references are exact for integer-valued images.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.double().sum(dim=(1, 2, 3))
        return torch.stack([total, -total], dim=1)


@pytest.fixture
def tiny_image() -> np.ndarray:
    """A small RGB image (4x4x3) of integer-valued pixels."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(4, 4, 3)).astype(np.float64)


@pytest.fixture
def two_player_masks() -> np.ndarray:
    """Two non-overlapping spatial masks partitioning a 4x4 image into halves."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :, :2] = True  # left half
    masks[1, :, 2:] = True  # right half
    return masks


@pytest.fixture
def three_player_masks() -> np.ndarray:
    """Three non-overlapping spatial masks partitioning a 6x6 image into vertical thirds."""
    masks = np.zeros((3, 6, 6), dtype=bool)
    masks[0, :, 0:2] = True
    masks[1, :, 2:4] = True
    masks[2, :, 4:6] = True
    return masks


class MockViT:
    """HF-style ViT mock for :class:`TransformerArchitecture` tests.

    ``config.image_size=24`` and ``config.patch_size=8`` yield ``grid_size=3``,
    compatible with the default 9-player patch grid.  With ``bool_masked_pos``
    the class-0 logit equals the number of visible tokens.
    """

    class _Config:
        image_size = 24
        patch_size = 8
        hidden_size = 4

    config = _Config()

    def __init__(self) -> None:
        self.vit = SimpleNamespace(
            embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.zeros(1, 1, 4)))
        )

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        batch_size = pixel_values.shape[0]
        if bool_masked_pos is None:
            return SimpleNamespace(logits=torch.tensor([[2.0, 0.5]]).expand(batch_size, -1).clone())
        visible = (~bool_masked_pos).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))


class MockViTProcessor:
    """Mimics a HF image processor turning an HWC image into ``(1, C, H, W)``."""

    def __call__(self, images=None, return_tensors="pt"):
        arr = np.asarray(images, dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)
        return {"pixel_values": tensor}


@pytest.fixture
def image_24x24() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(24, 24, 3)).astype(np.float64)


@pytest.fixture
def transformer_architecture() -> TransformerArchitecture:
    return TransformerArchitecture(
        model=MockViT(),
        vit_processor=MockViTProcessor(),
        masking_strategy=BoolMaskedPosStrategy(),
    )


def expected_full_coalition_value(image: np.ndarray, image_input) -> float:
    """Expected channel-sum model output for a grand coalition after format conversion."""
    tensor = to_tensor_chw(image_input(image))
    return float(tensor.double().sum().item())
