"""Tests for ``shapiq.vision.imputer.ImageImputer``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from shapiq.game_theory.exact import ExactComputer
from shapiq.imputer.base import Imputer
from shapiq.vision.architecture import CNNArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import MeanColorMasking, ZeroMasking

from .conftest import ChannelSumModel, FixedMasksStrategy, expected_full_coalition_value


def _build_imputer(image, masks, masking_strategy, *, normalize=True, batch_size=32):
    arch = CNNArchitecture(
        model=ChannelSumModel(),
        masking_strategy=masking_strategy,
        player_strategy=FixedMasksStrategy(masks),
    )
    return ImageImputer(
        model_architecture=arch,
        image=image,
        normalize=normalize,
        batch_size=batch_size,
    )


class TestImageImputerBasics:
    def test_is_imputer_subclass(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking())
        assert isinstance(imputer, Imputer)

    def test_n_players_matches_player_strategy(self, three_player_masks) -> None:
        image = np.random.default_rng(0).integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        imputer = _build_imputer(image, three_player_masks, ZeroMasking())
        assert imputer.n_players == 3
        assert imputer.n_features == 3

    def test_player_masks_property_exposes_spatial_masks(
        self, tiny_image, two_player_masks
    ) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking())
        assert imputer.player_masks is not None
        np.testing.assert_array_equal(imputer.player_masks, two_player_masks)

    def test_empty_prediction_with_zero_masking_is_zero(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        assert imputer.empty_prediction == pytest.approx(0.0)

    def test_normalize_sets_normalization_value(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, MeanColorMasking(), normalize=True)
        assert imputer.normalization_value == pytest.approx(imputer.empty_prediction)

    def test_value_function_accepts_1d_coalition(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        out = np.atleast_1d(imputer.value_function(np.array([True, True])))
        assert out.shape[0] == 1


class TestImageImputerValues:
    def test_value_function_recovers_linear_model_output(
        self, tiny_image, two_player_masks
    ) -> None:
        """With ZeroMasking + the channel-sum model, a coalition's value equals the
        sum of pixel intensities restricted to the present players."""
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        coalitions = np.array(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
            ],
        )
        values = np.atleast_1d(imputer.value_function(coalitions))

        v_empty = 0.0
        v_left = tiny_image[:, :2].sum()
        v_right = tiny_image[:, 2:].sum()
        v_full = tiny_image.sum()
        np.testing.assert_allclose(values, [v_empty, v_left, v_right, v_full])

    def test_correctness_against_exact_computer(self, tiny_image, two_player_masks) -> None:
        """Shapley values of the imputer-induced game match the analytic linear values."""
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        ec = ExactComputer(n_players=imputer.n_players, game=imputer)
        sv = ec.probabilistic_value(index="SV")

        sv_left = tiny_image[:, :2].sum()
        sv_right = tiny_image[:, 2:].sum()
        assert sv[(0,)] == pytest.approx(sv_left)
        assert sv[(1,)] == pytest.approx(sv_right)

    def test_correctness_three_players(self, three_player_masks) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        imputer = _build_imputer(image, three_player_masks, ZeroMasking(), normalize=False)
        ec = ExactComputer(n_players=imputer.n_players, game=imputer)
        sv = ec.probabilistic_value(index="SV")

        regions = [
            image[:, 0:2].sum(),
            image[:, 2:4].sum(),
            image[:, 4:6].sum(),
        ]
        for i, expected in enumerate(regions):
            assert sv[(i,)] == pytest.approx(expected)

    def test_call_subtracts_normalization_value(self, tiny_image, two_player_masks) -> None:
        """Calling the imputer as a Game subtracts the normalization value."""
        imputer = _build_imputer(tiny_image, two_player_masks, MeanColorMasking(), normalize=True)
        coalitions = np.array([[False, False], [True, True]])
        out = imputer(coalitions)
        assert out[0] == pytest.approx(0.0, abs=1e-8)
        expected_full = tiny_image.sum() - imputer.empty_prediction
        assert out[1] == pytest.approx(expected_full)


class TestImageImputerInputFormats:
    @pytest.mark.parametrize(
        "image_input",
        [
            pytest.param(lambda img: img, id="numpy"),
            pytest.param(lambda img: Image.fromarray(img.astype(np.uint8)), id="pil"),
            pytest.param(lambda img: torch.from_numpy(img).permute(2, 0, 1), id="torch_chw"),
        ],
    )
    def test_accepts_common_image_formats(self, tiny_image, two_player_masks, image_input) -> None:
        arch = CNNArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        imputer = ImageImputer(
            model_architecture=arch,
            image=image_input(tiny_image),
            normalize=False,
        )
        values = imputer.value_function(np.array([[True, True]]))
        assert values.shape == (1,)
        expected = expected_full_coalition_value(tiny_image, image_input)
        assert values[0] == pytest.approx(expected)

    def test_accepts_torch_hwc_tensor(self, three_player_masks) -> None:
        """HWC tensors are only unambiguous when H is not in {1, 3, 4}."""
        image = np.random.default_rng(0).integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        arch = CNNArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(three_player_masks),
        )
        imputer = ImageImputer(
            model_architecture=arch,
            image=torch.from_numpy(image),
            normalize=False,
        )
        values = imputer.value_function(np.array([[True, True, True]]))
        assert values[0] == pytest.approx(image.sum())


class TestImageImputerFit:
    def test_fit_returns_self(self, tiny_image, two_player_masks) -> None:
        """fit() returns the imputer instance for chaining."""
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking())
        result = imputer.fit(tiny_image)
        assert result is imputer

    def test_fit_replaces_image_and_updates_value_function(
        self, tiny_image, two_player_masks
    ) -> None:
        """After fit(), value_function reflects the new image, not the old one."""
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        new_image = np.ones_like(tiny_image) * 10.0
        imputer.fit(new_image)
        full_coalition = np.array([[True, True]])
        value = imputer.value_function(full_coalition)[0]
        assert value == pytest.approx(new_image.sum())
        assert value != pytest.approx(tiny_image.sum())

    def test_fit_resets_empty_prediction(self, tiny_image, two_player_masks) -> None:
        """fit() resets the empty prediction baseline to match the new image."""
        imputer = _build_imputer(tiny_image, two_player_masks, MeanColorMasking(), normalize=False)
        old_empty = imputer.empty_prediction
        new_image = np.ones_like(tiny_image) * 99.0
        imputer.fit(new_image)
        assert imputer.empty_prediction != pytest.approx(old_empty)


class TestImageImputerTransformer:
    def test_transformer_architecture_value_function(
        self, transformer_architecture, image_24x24
    ) -> None:
        imputer = ImageImputer(
            model_architecture=transformer_architecture,
            image=image_24x24,
            normalize=False,
        )
        assert imputer.n_players == 9
        coalitions = np.array(
            [
                [False] * 9,
                [True] + [False] * 8,
                [True] * 5 + [False] * 4,
                [True] * 9,
            ]
        )
        values = imputer.value_function(coalitions)
        assert values.shape == (4,)
        assert values[0] == pytest.approx(0.5, abs=1e-5)
        assert values[0] < values[1] < values[2] < values[3]

    def test_empty_prediction_matches_all_absent_coalition(
        self, transformer_architecture, image_24x24
    ) -> None:
        imputer = ImageImputer(
            model_architecture=transformer_architecture,
            image=image_24x24,
            normalize=False,
        )
        assert imputer.empty_prediction == pytest.approx(
            imputer.value_function(np.array([[False] * 9]))[0]
        )
