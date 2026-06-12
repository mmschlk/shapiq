"""Integration tests with real PyTorch vision backbones (no mocks).

These tests are slower than the unit suite and require optional ML extras
(``torchvision``, ``transformers``). They are marked ``integration`` so CI or
local runs can exclude them via ``pytest -m 'not integration'``.

We use randomly initialised weights (no checkpoint download) and small images
to keep runtime reasonable while still exercising real forward passes,
preprocessors, and masking paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import CNNArchitecture, TransformerArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import MeanColorMasking
from shapiq.vision.players import SuperpixelStrategy

from .conftest import FixedMasksStrategy


def _quadrant_masks(height: int, width: int) -> np.ndarray:
    masks = np.zeros((4, height, width), dtype=bool)
    masks[0, : height // 2, : width // 2] = True
    masks[1, : height // 2, width // 2 :] = True
    masks[2, height // 2 :, : width // 2] = True
    masks[3, height // 2 :, width // 2 :] = True
    return masks


@pytest.mark.integration
class TestResNetIntegration:
    def test_resnet18_imputer_value_function(self) -> None:
        torchvision = pytest.importorskip("torchvision")

        model = torchvision.models.resnet18(weights=None)
        model.eval()

        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        masks = _quadrant_masks(64, 64)

        arch = CNNArchitecture(
            model=model,
            masking_strategy=MeanColorMasking(),
            player_strategy=FixedMasksStrategy(masks),
        )
        imputer = ImageImputer(
            model_architecture=arch,
            image=image,
            normalize=False,
            batch_size=2,
        )
        coalitions = np.array([[True, True, True, True], [False, False, False, False]])
        values = imputer.value_function(coalitions)
        assert values.shape == (2,)
        assert np.isfinite(values).all()

    def test_resnet18_explainer_end_to_end(self) -> None:
        torchvision = pytest.importorskip("torchvision")

        model = torchvision.models.resnet18(weights=None)
        model.eval()

        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        masks = _quadrant_masks(64, 64)

        arch = CNNArchitecture(
            model=model,
            masking_strategy=MeanColorMasking(),
            player_strategy=FixedMasksStrategy(masks),
        )
        explainer = ImageExplainer(
            model_architecture=arch,
            data=image,
            index="k-SII",
            max_order=2,
            batch_size=2,
            random_state=0,
        )
        result = explainer.explain_function(image, budget=16)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 4
        assert np.isfinite(result.values).all()

    def test_resnet18_superpixel_players_end_to_end(self) -> None:
        pytest.importorskip("skimage")
        torchvision = pytest.importorskip("torchvision")

        model = torchvision.models.resnet18(weights=None)
        model.eval()

        rng = np.random.default_rng(2)
        image = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)

        arch = CNNArchitecture(
            model=model,
            masking_strategy=MeanColorMasking(),
            player_strategy=SuperpixelStrategy(n_segments=6),
        )
        imputer = ImageImputer(
            model_architecture=arch,
            image=image,
            normalize=False,
            batch_size=2,
        )
        assert imputer.n_players >= 1
        values = imputer.value_function(
            np.array([[True] * imputer.n_players, [False] * imputer.n_players])
        )
        assert np.isfinite(values).all()


@pytest.mark.integration
class TestViTIntegration:
    @staticmethod
    def _small_vit_and_processor():
        transformers = pytest.importorskip("transformers")
        from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

        # image_size=96, patch_size=16 -> grid_size=6, compatible with n_players=9.
        config = ViTConfig(
            image_size=96,
            patch_size=16,
            num_labels=2,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
        )
        model = ViTForImageClassification(config)
        model.eval()
        processor = ViTImageProcessor(size={"height": 96, "width": 96})
        return model, processor

    def test_vit_imputer_returns_probabilities(self) -> None:
        model, processor = self._small_vit_and_processor()

        rng = np.random.default_rng(3)
        image = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)

        # Default MaskTokenStrategy initialises mask_token on classification ViTs.
        arch = TransformerArchitecture(model=model, vit_processor=processor)
        imputer = ImageImputer(
            model_architecture=arch,
            image=image,
            normalize=False,
            batch_size=2,
        )
        assert imputer.n_players == 9
        values = imputer.value_function(np.array([[True] * 9, [False] * 9]))
        assert values.shape == (2,)
        assert np.isfinite(values).all()
        assert (values >= 0.0).all() and (values <= 1.0).all()

    def test_vit_explainer_end_to_end(self) -> None:
        model, processor = self._small_vit_and_processor()

        rng = np.random.default_rng(4)
        image = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)

        arch = TransformerArchitecture(model=model, vit_processor=processor)
        explainer = ImageExplainer(
            model_architecture=arch,
            data=image,
            index="k-SII",
            max_order=2,
            batch_size=2,
            random_state=0,
        )
        result = explainer.explain_function(image, budget=16)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 9
        assert np.isfinite(result.values).all()
        assert result[()] == pytest.approx(explainer.baseline_value)
