"""Tests for masking strategies in ``shapiq.vision.masking``.

The pixel-space maskers (:class:`MeanColorMasking`, :class:`ZeroMasking`,
:class:`BlurMasking`, :class:`DatasetMeanMasking`, :class:`MarginalSampling`,
:class:`InpaintingMasking`) operate on ``(C, H, W)`` float tensors and a
``(n_coalitions, n_players)`` boolean coalition tensor, returning a
``(n_coalitions, C, H, W)`` batch.

The token-space maskers (:class:`BoolMaskedPosStrategy`,
:class:`MaskTokenStrategy`) turn a coalition tensor into a flat token-level
boolean mask where ``True`` marks an absent (masked) token.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from shapiq.vision.custom_types import CoalitionDomain
from shapiq.vision.masking import (
    BlurMasking,
    BoolMaskedPosStrategy,
    DatasetMeanMasking,
    InpaintingMasking,
    LatentBasedMaskingStrategy,
    MarginalSampling,
    MaskTokenStrategy,
    MeanColorMasking,
    PixelBasedMaskingStrategy,
    ZeroMasking,
)

from .conftest import MockViT, make_vit_config


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


def _every_pixel_strategy() -> list[PixelBasedMaskingStrategy]:
    """One configured instance of every pixel-space strategy.

    Add new pixel strategies here so they inherit the shared invariants below.
    """
    return [
        MeanColorMasking(),
        ZeroMasking(),
        BlurMasking(sigma=1.0),
        DatasetMeanMasking(mean_color=0.5),
        MarginalSampling(reference_images=[torch.zeros(3, 4, 4)]),
        InpaintingMasking(inpainter=lambda image, mask: image),
    ]


@pytest.mark.parametrize("strategy", _every_pixel_strategy(), ids=lambda s: type(s).__name__)
class TestEveryPixelStrategy:
    """Invariants that hold for every pixel-space masking strategy."""

    def test_is_a_pixel_masking_strategy(self, strategy) -> None:
        assert isinstance(strategy, PixelBasedMaskingStrategy)

    def test_full_coalition_preserves_image(self, strategy, image, half_masks) -> None:
        """Hiding nothing must return the image untouched, whatever the fill is."""
        out = strategy.apply(image, half_masks, torch.tensor([[True, True]]))
        assert out.shape == (1, 3, 4, 4)
        torch.testing.assert_close(out[0], image)

    def test_present_players_are_never_altered(self, strategy, image, half_masks) -> None:
        """Only absent players' pixels may change."""
        out = strategy.apply(image, half_masks, torch.tensor([[True, False]]))
        torch.testing.assert_close(out[0, :, :, :2], image[:, :, :2])

    def test_batches_coalitions_independently(self, strategy, image, half_masks) -> None:
        """Each row of the batch must equal that coalition evaluated on its own."""
        coalitions = torch.tensor([[True, True], [True, False], [False, False]])
        batched = strategy.apply(image, half_masks, coalitions)
        assert batched.shape == (3, 3, 4, 4)
        for index, coalition in enumerate(coalitions):
            alone = strategy.apply(image, half_masks, coalition.unsqueeze(0))
            torch.testing.assert_close(batched[index], alone[0])


class TestMeanColorMasking:
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


class TestBlurMasking:
    def test_rejects_non_positive_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            BlurMasking(sigma=0.0)

    def test_partial_coalition_blurs_only_absent(self, image, half_masks) -> None:
        strategy = BlurMasking(sigma=1.0)
        out = strategy.apply(image, half_masks, torch.tensor([[True, False]]))
        assert not torch.allclose(out[0, :, :, 2:], image[:, :, 2:])

    def test_blur_reduces_variance_and_preserves_shape(self, image) -> None:
        blurred = BlurMasking(sigma=1.0)._blur(image)
        assert blurred.shape == image.shape
        assert blurred.var() < image.var()

    def test_blur_preserves_constant_image(self) -> None:
        """A separable, normalised kernel must leave a flat image untouched."""
        flat = torch.full((3, 8, 8), 0.4)
        torch.testing.assert_close(BlurMasking(sigma=2.0)._blur(flat), flat)


class TestDatasetMeanMasking:
    def test_mean_color_is_required(self) -> None:
        """No default: the right value depends on the architecture's image scale."""
        with pytest.raises(TypeError):
            DatasetMeanMasking()  # type: ignore[call-arg]

    def test_imagenet_constants_are_offered_for_both_scales(self, image, half_masks) -> None:
        strategy = DatasetMeanMasking(DatasetMeanMasking.IMAGENET_MEAN)
        out = strategy.apply(image, half_masks, torch.tensor([[False, False]]))
        expected = torch.tensor([0.485, 0.456, 0.406])[:, None, None].expand(3, 4, 4)
        torch.testing.assert_close(out[0], expected)
        assert (
            pytest.approx([value * 255 for value in DatasetMeanMasking.IMAGENET_MEAN])
            == DatasetMeanMasking.IMAGENET_MEAN_255
        )

    def test_custom_mean_color(self, image, half_masks) -> None:
        strategy = DatasetMeanMasking(mean_color=[0.5, 0.25, 0.75])
        out = strategy.apply(image, half_masks, torch.tensor([[True, False]]))
        expected = torch.tensor([0.5, 0.25, 0.75])[:, None, None].expand(3, 4, 2)
        torch.testing.assert_close(out[0, :, :, 2:], expected)

    def test_scalar_mean_color_broadcasts(self, image, half_masks) -> None:
        strategy = DatasetMeanMasking(mean_color=0.3)
        out = strategy.apply(image, half_masks, torch.tensor([[False, False]]))
        assert torch.allclose(out[0], torch.full((3, 4, 4), 0.3))

    def test_rejects_mean_color_of_wrong_length(self, image, half_masks) -> None:
        strategy = DatasetMeanMasking(mean_color=[0.1, 0.2])
        with pytest.raises(ValueError, match="one value per channel"):
            strategy.apply(image, half_masks, torch.tensor([[False, False]]))

    def test_is_independent_of_the_image(self, half_masks) -> None:
        """Unlike MeanColorMasking, the fill must not depend on the image."""
        strategy = DatasetMeanMasking(mean_color=0.5)
        coalition = torch.tensor([[False, False]])
        first = strategy.apply(torch.zeros(3, 4, 4), half_masks, coalition)
        second = strategy.apply(torch.ones(3, 4, 4), half_masks, coalition)
        torch.testing.assert_close(first, second)


class TestMarginalSampling:
    @pytest.fixture
    def references(self) -> list[torch.Tensor]:
        return [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]

    def test_rejects_empty_reference_bank(self) -> None:
        with pytest.raises(ValueError, match="at least one image"):
            MarginalSampling(reference_images=[])

    def test_rejects_mismatched_reference_shapes(self) -> None:
        with pytest.raises(ValueError, match="share one shape"):
            MarginalSampling(reference_images=[torch.zeros(3, 4, 4), torch.zeros(3, 8, 8)])

    def test_rejects_references_not_matching_image(self, image, half_masks) -> None:
        strategy = MarginalSampling(reference_images=[torch.zeros(3, 8, 8)])
        with pytest.raises(ValueError, match="Resize the reference images"):
            strategy.apply(image, half_masks, torch.tensor([[True, False]]))

    def test_empty_coalition_uses_a_reference(self, image, half_masks, references) -> None:
        strategy = MarginalSampling(references, random_state=0)
        out = strategy.apply(image, half_masks, torch.tensor([[False, False]]))
        assert any(torch.allclose(out[0], reference) for reference in references)

    def test_partial_coalition_fills_only_absent(self, image, half_masks) -> None:
        strategy = MarginalSampling([torch.full((3, 4, 4), 0.5)], random_state=0)
        out = strategy.apply(image, half_masks, torch.tensor([[True, False]]))
        assert (out[0, :, :, 2:] == 0.5).all()

    def test_value_depends_only_on_the_coalition(self, image, half_masks, references) -> None:
        """The same coalition must get the same reference in any batch position.

        The explainer chunks coalitions into batches, so drawing a reference per
        batch row would make v(S) depend on batch size and ordering, which breaks
        the deterministic set function Shapley values are defined over.
        """
        strategy = MarginalSampling(references, random_state=0)
        empty = torch.zeros((1, 2), dtype=torch.bool)
        alone = strategy.apply(image, half_masks, empty)[0]

        batched = strategy.apply(image, half_masks, torch.zeros((5, 2), dtype=torch.bool))
        for row in batched:
            torch.testing.assert_close(row, alone)

    def test_different_coalitions_can_draw_different_references(
        self, image, half_masks, references
    ) -> None:
        """Selection must vary with coalition content, not collapse to one reference."""
        strategy = MarginalSampling(references, random_state=0)
        players = 6
        masks = torch.zeros((players, 4, 4), dtype=torch.bool)
        for player in range(players):
            masks[player, :, :] = True
        coalitions = torch.zeros((2**players, players), dtype=torch.bool)
        for index in range(2**players):
            for player in range(players):
                coalitions[index, player] = bool(index & (1 << player))
        drawn = strategy._draw_for(coalitions)
        assert len(set(drawn.tolist())) == len(references), "all references should be reachable"

    def test_random_state_changes_the_assignment(self, image, half_masks, references) -> None:
        coalitions = torch.zeros((16, 3), dtype=torch.bool)
        for index in range(16):
            coalitions[index, index % 3] = True
        first = MarginalSampling(references, random_state=0)._draw_for(coalitions)
        second = MarginalSampling(references, random_state=99)._draw_for(coalitions)
        assert not torch.equal(first, second), "a different seed should reassign references"

    def test_accepts_numpy_reference_stack(self, image, half_masks) -> None:
        """A 4-D (N, H, W, C) uint8 array is accepted and scaled to [0, 1]."""
        import numpy as np

        references = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
        strategy = MarginalSampling(references, random_state=0)
        out = strategy.apply(image, half_masks, torch.tensor([[False, False]]))
        assert (out[0] == 1.0).all()


class TestInpaintingMasking:
    def test_rejects_non_callable_inpainter(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            InpaintingMasking(inpainter="not-callable")  # type: ignore[arg-type]

    def test_inpainter_fills_absent_region(self, image, half_masks) -> None:
        def inpainter(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            filled = img.clone()
            filled[:, mask] = 0.5
            return filled

        out = InpaintingMasking(inpainter).apply(image, half_masks, torch.tensor([[True, False]]))
        assert (out[0, :, :, 2:] == 0.5).all()

    def test_full_coalition_skips_inpainter(self, image, half_masks) -> None:
        calls: list[int] = []

        def inpainter(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            return img

        InpaintingMasking(inpainter).apply(image, half_masks, torch.tensor([[True, True]]))
        assert calls == [], "an empty absence mask must not invoke the inpainter"

    def test_rejects_malformed_inpainter_output(self, image, half_masks) -> None:
        strategy = InpaintingMasking(lambda img, mask: torch.zeros(3, 8, 8))
        with pytest.raises(TypeError, match="shaped like the image"):
            strategy.apply(image, half_masks, torch.tensor([[True, False]]))


def test_cnn_masking_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        PixelBasedMaskingStrategy()  # type: ignore[abstract]


def test_transformer_masking_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        LatentBasedMaskingStrategy()  # type: ignore[abstract]


@pytest.fixture
def token_masks() -> torch.Tensor:
    """Four players owning one token each (flat token indices 0..3)."""
    return torch.tensor([[0], [1], [2], [3]])


class TestBoolMaskedPosStrategy:
    def test_is_transformer_masking_strategy(self) -> None:
        assert isinstance(BoolMaskedPosStrategy(), LatentBasedMaskingStrategy)

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


class TestMaskTokenStrategy:
    def test_is_transformer_masking_strategy(self) -> None:
        assert isinstance(MaskTokenStrategy(MockViT()), LatentBasedMaskingStrategy)

    def test_apply_returns_token_mask(self, token_masks) -> None:
        strategy = MaskTokenStrategy(MockViT())
        coalitions = torch.tensor([[True, False, False, False]])
        out = strategy.apply(coalitions, token_masks)
        assert out.shape == (1, 4)
        assert not out[0, 0]
        assert out[0, 1:].all()

    def test_apply_zeros_mask_token(self, token_masks) -> None:
        model = MockViT()
        strategy = MaskTokenStrategy(model)
        # Starts non-zero.
        assert not torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))
        coalitions = torch.tensor([[True, True, True, True]])
        strategy.apply(coalitions, token_masks)
        assert torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))

    def test_mask_token_sized_from_config_hidden_size(self, token_masks) -> None:
        """The replacement mask token is shaped from ``config.hidden_size``, not the old token."""
        model = MockViT(hidden_size=16)
        MaskTokenStrategy(model).apply(torch.tensor([[True, True, True, True]]), token_masks)
        assert model.vit.embeddings.mask_token.shape == (1, 1, 16)


class TestMaskingStrategyModelValidation:
    """``validate_model`` guards the model attributes each token masker depends on."""

    def test_mask_token_strategy_rejects_non_callable_model(self) -> None:
        model = SimpleNamespace(config=make_vit_config(), vit=SimpleNamespace(embeddings=None))
        with pytest.raises(TypeError, match="VisionModel"):
            MaskTokenStrategy(model)

    def test_mask_token_strategy_rejects_model_without_mask_token(self) -> None:
        model = MockViT()
        del model.vit
        with pytest.raises(TypeError, match=re.escape("vit.embeddings.mask_token")):
            MaskTokenStrategy(model)

    def test_mask_token_strategy_rejects_model_without_hidden_size(self) -> None:
        model = MockViT()
        model.config.hidden_size = None
        with pytest.raises(TypeError, match="hidden_size"):
            MaskTokenStrategy(model)

    def test_mask_token_strategy_error_points_to_bool_masked_pos_strategy(self) -> None:
        """The hidden_size error names the fallback so users know what to switch to."""
        model = MockViT()
        model.config.hidden_size = None
        with pytest.raises(TypeError, match="BoolMaskedPosStrategy"):
            MaskTokenStrategy(model)

    def test_bool_masked_pos_accepts_model_with_mask_token(self) -> None:
        BoolMaskedPosStrategy.validate_model(MockViT())  # does not raise

    def test_bool_masked_pos_rejects_model_without_mask_token(self) -> None:
        model = MockViT()
        del model.vit
        with pytest.raises(TypeError, match=re.escape("vit.embeddings.mask_token")):
            BoolMaskedPosStrategy.validate_model(model)

    def test_bool_masked_pos_rejects_unset_mask_token(self) -> None:
        """``use_mask_token=False`` models leave ``mask_token`` as None and must be rejected."""
        model = MockViT()
        model.vit.embeddings.mask_token = None
        with pytest.raises(TypeError, match="use_mask_token=True"):
            BoolMaskedPosStrategy.validate_model(model)

    def test_bool_masked_pos_unset_mask_token_error_suggests_mask_token_strategy(self) -> None:
        model = MockViT()
        model.vit.embeddings.mask_token = None
        with pytest.raises(TypeError, match="MaskTokenStrategy"):
            BoolMaskedPosStrategy.validate_model(model)

    def test_bool_masked_pos_rejects_non_callable_model(self) -> None:
        model = SimpleNamespace(
            vit=SimpleNamespace(embeddings=SimpleNamespace(mask_token=torch.zeros(1, 1, 4)))
        )
        with pytest.raises(TypeError, match="VisionModel"):
            BoolMaskedPosStrategy.validate_model(model)


class TestCoalitionDomains:
    """Each masker declares the coalition domain it accepts, which the architecture cross-checks."""

    @pytest.mark.parametrize("strategy", [MeanColorMasking(), ZeroMasking()])
    def test_pixel_maskers_accept_pixel_domain(self, strategy) -> None:
        assert strategy.accepted_coalition_domain is CoalitionDomain.PIXEL

    def test_token_maskers_accept_token_domain(self) -> None:
        assert BoolMaskedPosStrategy().accepted_coalition_domain is CoalitionDomain.TOKEN
        assert MaskTokenStrategy(MockViT()).accepted_coalition_domain is CoalitionDomain.TOKEN
