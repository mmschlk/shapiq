"""Tests for masking strategies in ``shapiq.vision.masking``.

The pixel-space maskers (:class:`MeanColorMasking`, :class:`ZeroMasking`)
operate on ``(C, H, W)`` float tensors and a ``(n_coalitions, n_players)``
boolean coalition tensor, returning a ``(n_coalitions, C, H, W)`` batch.

The token-space maskers (:class:`BoolMaskedPosStrategy`,
:class:`MaskTokenStrategy`) turn a coalition tensor into a flat token-level
boolean mask where ``True`` marks an absent (masked) token.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from shapiq.vision.masking import (
    BoolMaskedPosStrategy,
    CNNMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    TransformerMaskingStrategy,
    ZeroMasking,
)


@pytest.fixture
def image() -> torch.Tensor:
    """A (C, H, W) float image with deterministic content."""
    rng = torch.Generator().manual_seed(0)
    return torch.randint(0, 255, size=(3, 4, 4), generator=rng).float()


@pytest.fixture
def half_masks() -> torch.Tensor:
    """Two non-overlapping (n_players, H, W) masks splitting a 4x4 image into halves."""
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, :, :2] = True  # left half
    masks[1, :, 2:] = True  # right half
    return masks


class TestMeanColorMasking:
    def test_is_cnn_masking_strategy(self) -> None:
        assert isinstance(MeanColorMasking(), CNNMaskingStrategy)

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = torch.tensor([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        assert out.shape == (1, 3, 4, 4)
        torch.testing.assert_close(out[0], image)

    def test_empty_coalition_uses_mean_color_everywhere(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = torch.tensor([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        mean_color = image.mean(dim=(1, 2))  # (C,)
        expected = mean_color[:, None, None].expand(3, 4, 4)
        torch.testing.assert_close(out[0], expected)

    def test_partial_coalition_masks_only_absent_player(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = torch.tensor([[True, False]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (player 0 present) preserved.
        torch.testing.assert_close(out[0, :, :, :2], image[:, :, :2])
        # Right half (player 1 absent) replaced with the per-channel mean color.
        mean_color = image.mean(dim=(1, 2))
        expected_right = mean_color[:, None, None].expand(3, 4, 2)
        torch.testing.assert_close(out[0, :, :, 2:], expected_right)

    def test_multiple_coalitions_handled_independently(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalitions = torch.tensor(
            [
                [True, True],
                [False, False],
                [True, False],
                [False, True],
            ]
        )
        out = strategy.apply(image, half_masks, coalitions)
        assert out.shape == (4, 3, 4, 4)
        torch.testing.assert_close(out[0], image)
        torch.testing.assert_close(out[2, :, :, :2], image[:, :, :2])
        torch.testing.assert_close(out[3, :, :, 2:], image[:, :, 2:])


class TestZeroMasking:
    def test_is_cnn_masking_strategy(self) -> None:
        assert isinstance(ZeroMasking(), CNNMaskingStrategy)

    def test_default_value_is_zero(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = torch.tensor([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 0).all()

    def test_custom_value(self, image, half_masks) -> None:
        strategy = ZeroMasking(value=7.0)
        coalition = torch.tensor([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 7.0).all()

    def test_partial_coalition_zeros_only_absent(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = torch.tensor([[False, True]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (absent) zeroed; right half (present) preserved.
        assert (out[0, :, :, :2] == 0).all()
        torch.testing.assert_close(out[0, :, :, 2:], image[:, :, 2:])

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = torch.tensor([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        torch.testing.assert_close(out[0], image)


def test_cnn_masking_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        CNNMaskingStrategy()  # type: ignore[abstract]


def test_transformer_masking_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        TransformerMaskingStrategy()  # type: ignore[abstract]


@pytest.fixture
def token_masks() -> torch.Tensor:
    """Four players owning one token each (flat token indices 0..3)."""
    return torch.tensor([[0], [1], [2], [3]])


class TestBoolMaskedPosStrategy:
    def test_is_transformer_masking_strategy(self) -> None:
        assert isinstance(BoolMaskedPosStrategy(), TransformerMaskingStrategy)

    def test_all_present_coalition_masks_nothing(self, token_masks) -> None:
        strategy = BoolMaskedPosStrategy()
        coalitions = torch.tensor([[True, True, True, True]])
        out = strategy.apply(coalitions, token_masks)
        assert out.shape == (1, 4)
        assert out.dtype == torch.bool
        # True == masked; all present means nothing masked.
        assert not out.any()

    def test_empty_coalition_masks_everything(self, token_masks) -> None:
        strategy = BoolMaskedPosStrategy()
        coalitions = torch.tensor([[False, False, False, False]])
        out = strategy.apply(coalitions, token_masks)
        assert out.all()

    def test_single_player_present_unmasks_only_its_token(self, token_masks) -> None:
        strategy = BoolMaskedPosStrategy()
        coalitions = torch.tensor([[True, False, False, False]])
        out = strategy.apply(coalitions, token_masks)
        # Token 0 visible (False), the rest masked (True).
        assert not out[0, 0]
        assert out[0, 1:].all()


class _MockViTWithMaskToken:
    """Minimal ViT mock satisfying :class:`MaskTokenStrategy`'s requirements.

    ``MaskTokenStrategy`` reads ``model.config.hidden_size`` to create the zero
    tensor and overwrites ``model.vit.embeddings.mask_token``.
    """

    class _Config:
        hidden_size = 4

    config = _Config()

    def __init__(self) -> None:
        self.vit = SimpleNamespace(
            embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.ones(1, 1, 4)))
        )


class TestMaskTokenStrategy:
    def test_is_transformer_masking_strategy(self) -> None:
        assert isinstance(MaskTokenStrategy(_MockViTWithMaskToken()), TransformerMaskingStrategy)

    def test_apply_returns_token_mask(self, token_masks) -> None:
        strategy = MaskTokenStrategy(_MockViTWithMaskToken())
        coalitions = torch.tensor([[True, False, False, False]])
        out = strategy.apply(coalitions, token_masks)
        assert out.shape == (1, 4)
        assert not out[0, 0]
        assert out[0, 1:].all()

    def test_apply_zeros_mask_token(self, token_masks) -> None:
        model = _MockViTWithMaskToken()
        strategy = MaskTokenStrategy(model)
        # Starts non-zero.
        assert not torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))
        coalitions = torch.tensor([[True, True, True, True]])
        strategy.apply(coalitions, token_masks)
        assert torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))
