"""Tests for player strategies in ``shapiq.vision.players``."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.vision.custom_types import CoalitionDomain
from shapiq.vision.players import (
    CustomPlayerStrategy,
    GridStrategy,
    LatentBasedPlayerStrategy,
    PatchStrategy,
    PixelBasedPlayerStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
    labels_to_masks,
)


class TestLabelsToMasks:
    """Converts a segmentation label map into the per-player boolean masks used everywhere else."""

    def test_one_mask_per_unique_label(self) -> None:
        labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
        masks = labels_to_masks(labels)
        assert masks.shape == (3, 4, 4)
        assert masks.dtype == bool

    def test_masks_partition_the_image(self) -> None:
        labels = np.array([[0, 1], [2, 3]])
        assert (labels_to_masks(labels).sum(axis=0) == 1).all()

    def test_masks_follow_sorted_label_order(self) -> None:
        """Row i belongs to the i-th smallest label, not to label value i."""
        labels = np.array([[7, 7], [3, 3]])
        masks = labels_to_masks(labels)
        np.testing.assert_array_equal(masks[0], labels == 3)
        np.testing.assert_array_equal(masks[1], labels == 7)

    def test_non_contiguous_labels_supported(self) -> None:
        """SLIC starts at label 1, and custom label maps may skip values entirely."""
        labels = np.array([[1, 1], [5, 9]])
        masks = labels_to_masks(labels)
        assert masks.shape == (3, 2, 2)
        assert (masks.sum(axis=0) == 1).all()

    def test_single_label_yields_one_full_mask(self) -> None:
        masks = labels_to_masks(np.zeros((3, 3), dtype=int))
        assert masks.shape == (1, 3, 3)
        assert masks.all()


class TestCoalitionDomains:
    """The architecture rejects a player/masker pair whose domains disagree."""

    def test_pixel_strategies_declare_pixel_domain(self) -> None:
        assert SuperpixelStrategy(n_segments=4).coalition_domain is CoalitionDomain.PIXEL
        assert GridStrategy(grid_shape=2).coalition_domain is CoalitionDomain.PIXEL
        assert PixelBasedPlayerStrategy.coalition_domain is CoalitionDomain.PIXEL

    def test_token_strategies_declare_token_domain(self) -> None:
        assert PatchStrategy(grid_size=4, n_players=4).coalition_domain is CoalitionDomain.TOKEN
        assert LatentBasedPlayerStrategy.coalition_domain is CoalitionDomain.TOKEN


class TestPatchStrategy:
    def test_is_transformer_player_strategy(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        assert isinstance(strategy, LatentBasedPlayerStrategy)
        assert isinstance(strategy, PlayerStrategy)

    def test_n_players_property(self) -> None:
        strategy = PatchStrategy(grid_size=6, n_players=9)
        assert strategy.n_players == 9

    def test_init_computes_side_and_patch_size(self) -> None:
        strategy = PatchStrategy(grid_size=8, n_players=4)
        assert strategy.side == 2
        assert strategy.patch_size == 4

    def test_init_rejects_non_perfect_square(self) -> None:
        with pytest.raises(ValueError, match="perfect square"):
            PatchStrategy(grid_size=8, n_players=5)

    def test_init_rejects_non_divisible_grid_size(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            PatchStrategy(grid_size=14, n_players=9)

    @pytest.mark.parametrize(
        ("grid_size", "expected"),
        [(3, 9), (6, 9), (12, 9), (14, 4), (16, 4), (7, 49)],
    )
    def test_default_n_players_is_constructible(self, grid_size, expected) -> None:
        n_players = PatchStrategy.default_n_players(grid_size)
        assert n_players == expected
        strategy = PatchStrategy(grid_size=grid_size, n_players=n_players)
        assert grid_size % strategy.side == 0

    def test_get_token_masks_shape(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        token_masks = strategy.get_token_masks()
        # 4 players, each owns a 2x2 patch -> 4 tokens per player.
        assert token_masks.shape == (4, 4)
        assert np.issubdtype(token_masks.dtype, np.integer)

    def test_get_token_masks_partition_covers_all_tokens(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        token_masks = strategy.get_token_masks()
        # The union of all token indices must cover the full flattened grid exactly once.
        all_tokens = np.sort(token_masks.reshape(-1))
        np.testing.assert_array_equal(all_tokens, np.arange(16))

    def test_get_token_masks_top_left_player_indices(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        token_masks = strategy.get_token_masks()
        # Player 0 owns the top-left 2x2 patch: flat indices 0, 1, 4, 5.
        np.testing.assert_array_equal(np.sort(token_masks[0]), np.array([0, 1, 4, 5]))

    def test_get_pixel_masks_shape_and_partition(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        image = np.zeros((8, 8, 3))
        masks = strategy.get_pixel_masks(image)
        assert masks.shape == (4, 8, 8)
        assert masks.dtype == bool
        # Each pixel belongs to exactly one player.
        assert (masks.sum(axis=0) == 1).all()


class TestSuperpixelStrategy:
    def test_is_cnn_player_strategy(self) -> None:
        strategy = SuperpixelStrategy(n_segments=5)
        assert isinstance(strategy, PixelBasedPlayerStrategy)
        assert isinstance(strategy, PlayerStrategy)

    def test_n_players_matches_n_segments(self) -> None:
        strategy = SuperpixelStrategy(n_segments=7)
        assert strategy.n_players == 7

    @pytest.mark.parametrize("n_segments", [0, -1])
    def test_rejects_non_positive_n_segments(self, n_segments) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            SuperpixelStrategy(n_segments=n_segments)

    def test_slico_algorithm_selectable(self) -> None:
        """SLIC-zero enforces equal-size segments regardless of texture."""
        pytest.importorskip("skimage")
        image = np.random.default_rng(3).integers(0, 255, size=(24, 24, 3)).astype(np.float64)
        masks = SuperpixelStrategy(n_segments=4, algorithm="slico").get_masks(image)
        assert (masks.sum(axis=0) == 1).all()

    def test_get_masks_shape_and_dtype(self) -> None:
        pytest.importorskip("skimage")
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(32, 32, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=4)
        masks = strategy.get_masks(image)
        assert masks.shape[0] == strategy.n_players
        assert masks.shape[1:] == (32, 32)
        assert masks.dtype == bool

    def test_get_masks_partition_property(self) -> None:
        """Each pixel should belong to exactly one superpixel."""
        pytest.importorskip("skimage")
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, size=(24, 24, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=6)
        masks = strategy.get_masks(image)
        coverage = masks.sum(axis=0)
        assert (coverage == 1).all()

    def test_slic_updates_n_players_after_segmentation(self) -> None:
        """n_players reflects the actual SLIC output, not just the request."""
        pytest.importorskip("skimage")
        image = np.random.default_rng(2).integers(0, 255, size=(16, 16, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=20)
        masks = strategy.get_masks(image)
        assert strategy.n_players == masks.shape[0]
        assert strategy.n_players >= 1
        assert (masks.sum(axis=0) == 1).all()


class TestCustomMaskStrategy:
    def test_label_map_converted_correctly(self) -> None:
        """2-D label map → one bool mask per unique label, correct shape & coverage."""
        labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        strategy = CustomPlayerStrategy(masks=labels)
        masks = strategy.get_masks(np.zeros((4, 4, 3)))
        assert masks.shape == (4, 4, 4)
        assert masks.dtype == bool
        assert strategy.n_players == 4
        assert (masks.sum(axis=0) == 1).all()  # non-overlapping labels

    def test_bool_mask_accepted_and_non_bool_cast(self) -> None:
        """3-D bool mask sets n_players; non-bool dtypes are cast to bool."""
        masks_uint = np.zeros((2, 4, 4), dtype=np.uint8)
        masks_uint[0, :, :2] = 255
        masks_uint[1, :, 2:] = 1
        strategy = CustomPlayerStrategy(masks=masks_uint)
        result = strategy.get_masks(np.zeros((4, 4, 3)))
        assert strategy.n_players == 2
        assert result.dtype == bool

    def test_overlapping_masks_allowed(self) -> None:
        """Overlapping masks must not raise — pixels owned by multiple players are valid."""
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :, :3] = True
        masks[1, :, 1:] = True  # columns 1-2 overlap
        strategy = CustomPlayerStrategy(masks=masks)  # no error
        assert strategy.n_players == 2

    def test_uncovered_pixels_raise_user_warning(self) -> None:
        """Pixels not covered by any mask trigger a UserWarning (not ValueError)."""
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :, :1] = True
        masks[1, :, 1:2] = True  # columns 2-3 uncovered
        with pytest.warns(UserWarning):
            CustomPlayerStrategy(masks=masks)

    def test_rejects_empty_player_mask(self) -> None:
        """An all-False player mask raises ValueError."""
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :, :] = True
        # masks[1] stays all-False
        with pytest.raises(ValueError):
            CustomPlayerStrategy(masks=masks)

    def test_rejects_invalid_shape(self) -> None:
        """Non-3D arrays (and non-2D label maps) raise ValueError."""
        with pytest.raises(ValueError):
            CustomPlayerStrategy(masks=np.zeros((2, 4, 4, 1), dtype=bool))  # 4-D
        with pytest.raises(ValueError):
            CustomPlayerStrategy(masks=np.array([1, 2, 3]))  # 1-D

    def test_label_map_requires_at_least_two_labels(self) -> None:
        """A label map with only one unique value raises ValueError."""
        with pytest.raises(ValueError):
            CustomPlayerStrategy(masks=np.ones((4, 4), dtype=int))

    def test_get_masks_raises_on_spatial_mismatch(self) -> None:
        """Mask spatial dims not matching the image raises ValueError."""
        labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        strategy = CustomPlayerStrategy(masks=labels)
        with pytest.raises(ValueError, match="do not match"):
            strategy.get_masks(np.zeros((8, 8, 3)))

    def test_get_masks_image_content_ignored(self) -> None:
        """Returned masks are identical regardless of image pixel values."""
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :, :2] = True
        masks[1, :, 2:] = True
        strategy = CustomPlayerStrategy(masks=masks)
        np.testing.assert_array_equal(
            strategy.get_masks(np.zeros((4, 4, 3))),
            strategy.get_masks(np.random.rand(4, 4, 3)),
        )


class TestGridStrategy:
    def test_rejects_both_params(self) -> None:
        """Providing both patch_size and grid_shape raises ValueError."""
        with pytest.raises(ValueError):
            GridStrategy(patch_size=32, grid_shape=4)

    def test_rejects_neither_param(self) -> None:
        """Providing neither patch_size nor grid_shape raises ValueError."""
        with pytest.raises(ValueError):
            GridStrategy()

    def test_grid_shape_int_produces_square_grid(self) -> None:
        """Scalar grid_shape=4: 4x4 = 16 players, masks cover every pixel."""
        strategy = GridStrategy(grid_shape=4)
        image = np.zeros((8, 8, 3))
        masks = strategy.get_masks(image)
        assert masks.shape == (16, 8, 8)
        assert masks.dtype == bool
        assert strategy.n_players == 16
        assert (masks.sum(axis=0) == 1).all()

    def test_grid_shape_tuple(self) -> None:
        """Tuple grid_shape=(2, 3): 6 players."""
        strategy = GridStrategy(grid_shape=(2, 3))
        masks = strategy.get_masks(np.zeros((6, 9, 3)))
        assert masks.shape == (6, 6, 9)
        assert strategy.n_players == 6

    def test_grid_shape_remainder_absorbed_into_last_patch(self) -> None:
        """When image dims are not divisible, edge patches are larger."""
        strategy = GridStrategy(grid_shape=3)
        image = np.zeros((10, 10, 3))
        masks = strategy.get_masks(image)
        assert masks.shape == (9, 10, 10)
        assert (masks.sum(axis=0) == 1).all()  # full coverage

    def test_patch_size_int_infers_grid(self) -> None:
        """Scalar patch_size: grid inferred from image; all pixels covered."""
        strategy = GridStrategy(patch_size=4)
        image = np.zeros((8, 8, 3))
        masks = strategy.get_masks(image)
        assert masks.shape == (4, 8, 8)  # 2x2 grid of 4x4 patches
        assert (masks.sum(axis=0) == 1).all()

    def test_patch_size_tuple(self) -> None:
        """Tuple patch_size=(2, 4):grid_y=4, grid_x=2 for an (8,8) image."""
        strategy = GridStrategy(patch_size=(2, 4))
        masks = strategy.get_masks(np.zeros((8, 8, 3)))
        assert masks.shape == (8, 8, 8)  # 4 rows x 2 cols

    def test_patch_size_non_multiple_edge_patches_smaller(self) -> None:
        """Image dims not multiples of patch_size: last patches are smaller but full coverage holds."""
        strategy = GridStrategy(patch_size=3)
        image = np.zeros((10, 10, 3))  # ceil(10/3)=4: 4x4=16 players
        masks = strategy.get_masks(image)
        assert masks.shape[1:] == (10, 10)
        assert (masks.sum(axis=0) == 1).all()

    def test_n_players_raises_before_get_masks(self) -> None:
        """Accessing n_players before get_masks raises RuntimeError."""
        strategy = GridStrategy(grid_shape=4)
        with pytest.raises(RuntimeError):
            _ = strategy.n_players

    def test_n_players_available_after_get_masks(self) -> None:
        """n_players is accessible and correct after get_masks is called."""
        strategy = GridStrategy(grid_shape=(2, 5))
        strategy.get_masks(np.zeros((4, 10, 3)))
        assert strategy.n_players == 10

    def test_get_masks_accepts_2d_image(self) -> None:
        """(H, W) image without channel dim is accepted."""
        strategy = GridStrategy(grid_shape=2)
        masks = strategy.get_masks(np.zeros((4, 4)))
        assert masks.shape == (4, 4, 4)

    def test_get_masks_is_deterministic(self) -> None:
        """Calling get_masks twice with different content returns identical masks."""
        strategy = GridStrategy(patch_size=2)
        image = np.zeros((4, 4, 3))
        np.testing.assert_array_equal(
            strategy.get_masks(image),
            strategy.get_masks(np.random.rand(4, 4, 3)),
        )


class TestGridStrategyResolutionErrors:
    """The grid is resolved against the image at fit time, so these only surface in get_masks."""

    @pytest.mark.parametrize("grid_shape", [0, -1, (0, 2), (2, -3)])
    def test_rejects_non_positive_grid_dimensions(self, grid_shape) -> None:
        strategy = GridStrategy(grid_shape=grid_shape)
        with pytest.raises(ValueError, match="positive integers"):
            strategy.get_masks(np.zeros((8, 8, 3)))

    def test_rejects_grid_finer_than_the_image(self) -> None:
        """More tiles than pixels along an axis would leave players with no pixels at all."""
        strategy = GridStrategy(grid_shape=10)
        with pytest.raises(ValueError, match="exceeds image shape"):
            strategy.get_masks(np.zeros((4, 4, 3)))

    def test_grid_error_mentions_empty_players(self) -> None:
        strategy = GridStrategy(grid_shape=(2, 99))
        with pytest.raises(ValueError, match="empty players"):
            strategy.get_masks(np.zeros((4, 4, 3)))

    @pytest.mark.parametrize("patch_size", [0, -4, (2, 0)])
    def test_rejects_non_positive_patch_dimensions(self, patch_size) -> None:
        strategy = GridStrategy(patch_size=patch_size)
        with pytest.raises(ValueError, match="positive integers"):
            strategy.get_masks(np.zeros((8, 8, 3)))

    def test_rejects_patch_larger_than_the_image(self) -> None:
        strategy = GridStrategy(patch_size=16)
        with pytest.raises(ValueError, match="exceeds image shape"):
            strategy.get_masks(np.zeros((8, 8, 3)))

    def test_grid_equal_to_image_size_is_allowed(self) -> None:
        """One tile per pixel is the finest valid grid."""
        strategy = GridStrategy(grid_shape=4)
        masks = strategy.get_masks(np.zeros((4, 4, 3)))
        assert masks.shape == (16, 4, 4)
        assert (masks.sum(axis=0) == 1).all()
