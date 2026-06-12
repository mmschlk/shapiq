"""End-to-end tests for ``shapiq.vision.explainer.ImageExplainer``."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import CNNArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import ZeroMasking

from .conftest import ChannelSumModel, FixedMasksStrategy


def _build_arch(masks):
    return CNNArchitecture(
        model=ChannelSumModel(),
        masking_strategy=ZeroMasking(),
        player_strategy=FixedMasksStrategy(masks),
    )


class TestImageExplainer:
    def test_explainer_returns_interaction_values(self, tiny_image, two_player_masks) -> None:
        explainer = ImageExplainer(
            model_architecture=_build_arch(two_player_masks),
            data=tiny_image,
            index="k-SII",
            max_order=2,
            random_state=0,
        )
        result = explainer.explain_function(tiny_image, budget=16)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 2

    def test_explainer_baseline_matches_empty_prediction(
        self, tiny_image, two_player_masks
    ) -> None:
        explainer = ImageExplainer(
            model_architecture=_build_arch(two_player_masks),
            data=tiny_image,
            random_state=0,
        )
        assert explainer.baseline_value == pytest.approx(explainer._imputer.empty_prediction)

    def test_explainer_three_players_e2e(self, three_player_masks) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        explainer = ImageExplainer(
            model_architecture=_build_arch(three_player_masks),
            data=image,
            index="k-SII",
            max_order=2,
            random_state=42,
        )
        result = explainer.explain_function(image, budget=32)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 3
        assert np.isfinite(result.values).all()

    def test_explainer_uses_provided_imputer(self, tiny_image, two_player_masks) -> None:
        arch = _build_arch(two_player_masks)
        imputer = ImageImputer(
            model_architecture=arch,
            image=tiny_image,
            normalize=False,
            batch_size=8,
        )
        explainer = ImageExplainer(
            model_architecture=arch,
            data=tiny_image,
            imputer=imputer,
            random_state=0,
        )
        assert explainer._imputer is imputer
        assert explainer._imputer._batch_size == 8

    def test_explainer_sets_empty_interaction_for_k_sii(self, tiny_image, two_player_masks) -> None:
        explainer = ImageExplainer(
            model_architecture=_build_arch(two_player_masks),
            data=tiny_image,
            index="k-SII",
            random_state=0,
        )
        result = explainer.explain_function(tiny_image, budget=16)
        assert result[()] == pytest.approx(explainer.baseline_value)

    def test_explain_function_uses_argument_x(self, two_player_masks) -> None:
        image_a = np.full((4, 4, 3), 200.0)
        image_b = np.zeros((4, 4, 3))

        explainer = ImageExplainer(
            model_architecture=_build_arch(two_player_masks),
            data=image_a,
            index="k-SII",
            max_order=2,
            random_state=0,
        )

        # Calling with image_b should give the same result as constructing on image_b
        result_with_b = explainer.explain_function(image_b, budget=32)

        explainer_on_b = ImageExplainer(
            model_architecture=_build_arch(two_player_masks),
            data=image_b,
            index="k-SII",
            max_order=2,
            random_state=0,
        )
        native_b = explainer_on_b.explain_function(image_b, budget=32)

        np.testing.assert_allclose(result_with_b.values, native_b.values)

        # Sanity check: image_a and image_b produce different results
        result_with_a = explainer.explain_function(image_a, budget=32)
        assert not np.allclose(result_with_a.values, result_with_b.values)


class TestImageExplainerTransformer:
    def test_transformer_explainer_end_to_end(self, transformer_architecture, image_24x24) -> None:
        explainer = ImageExplainer(
            model_architecture=transformer_architecture,
            data=image_24x24,
            index="k-SII",
            max_order=2,
            random_state=0,
        )
        result = explainer.explain_function(image_24x24, budget=32)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 9
        assert np.isfinite(result.values).all()
        assert result[()] == pytest.approx(explainer.baseline_value)
