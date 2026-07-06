"""Tests for the ``batch_size`` parameter on ImageImputer and ImageExplainer.

The batching contract of :class:`~shapiq.vision.imputer.ImageImputer`:

- ``n_coalitions <= batch_size`` results in a single architecture call.
- otherwise the coalitions are split into chunks of at most ``batch_size`` rows,
  giving ``ceil(n / batch_size)`` calls.
- output values are identical regardless of the batch size.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import CNNArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import ZeroMasking

from .conftest import ChannelSumModel, FixedMasksStrategy


class _CountingCNN(CNNArchitecture):
    """CNNArchitecture that records the per-call coalition batch size."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.call_batch_sizes: list[int] = []

    def value_function(self, coalitions):
        self.call_batch_sizes.append(int(coalitions.shape[0]))
        return super().value_function(coalitions)


def _build(image, masks, *, batch_size):
    arch = _CountingCNN(
        model=ChannelSumModel(),
        masking_strategy=ZeroMasking(),
        player_strategy=FixedMasksStrategy(masks),
    )
    imputer = ImageImputer(
        model_architecture=arch,
        image=image,
        batch_size=batch_size,
        normalize=False,
    )
    return imputer, arch


@pytest.fixture
def setup_image_and_masks():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(4, 4, 3)).astype(np.float64)
    masks = np.zeros((4, 4, 4), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, :2, 2:] = True
    masks[2, 2:, :2] = True
    masks[3, 2:, 2:] = True
    return image, masks


def _all_coalitions(n: int) -> np.ndarray:
    return np.array([[(i >> j) & 1 == 1 for j in range(n)] for i in range(1 << n)], dtype=bool)


class TestBatchingNumerics:
    def test_output_identical_across_batch_sizes(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)

        imp_32, _ = _build(image, masks, batch_size=32)
        imp_3, _ = _build(image, masks, batch_size=3)
        imp_5, _ = _build(image, masks, batch_size=5)
        imp_1, _ = _build(image, masks, batch_size=1)

        v_32 = imp_32.value_function(coalitions)
        np.testing.assert_array_equal(v_32, imp_3.value_function(coalitions))
        np.testing.assert_array_equal(v_32, imp_5.value_function(coalitions))
        np.testing.assert_array_equal(v_32, imp_1.value_function(coalitions))

    def test_batch_size_larger_than_n_is_one_call(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)  # 16 coalitions
        imp, arch = _build(image, masks, batch_size=100)
        # Reset counter (the constructor's empty-prediction call also bumps it).
        arch.call_batch_sizes.clear()
        imp.value_function(coalitions)
        assert arch.call_batch_sizes == [16]

    def test_chunking_call_pattern(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)  # 16 coalitions
        imp, arch = _build(image, masks, batch_size=5)
        arch.call_batch_sizes.clear()
        imp.value_function(coalitions)
        # 16 split by 5 -> chunks of 5, 5, 5, 1.
        assert arch.call_batch_sizes == [5, 5, 5, 1]

    def test_batch_size_one_splits_per_coalition(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)
        imp, arch = _build(image, masks, batch_size=1)
        arch.call_batch_sizes.clear()
        imp.value_function(coalitions)
        assert arch.call_batch_sizes == [1] * 16

    def test_1d_coalition_still_works_under_batching(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        imp, _ = _build(image, masks, batch_size=2)
        out = imp.value_function(np.array([True, True, True, True]))
        # Linear model, full coalition with ZeroMasking -> original image sum.
        assert out.shape == (1,)
        assert out[0] == pytest.approx(image.sum())


class TestBatchingPropagation:
    def test_explainer_passes_batch_size_to_imputer(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        arch = _CountingCNN(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(masks),
        )
        explainer = ImageExplainer(
            model=arch,
            data=image,
            batch_size=4,
            random_state=0,
        )
        assert explainer._imputer._batch_size == 4

    def test_default_batch_size_is_int(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        imp, _ = _build(image, masks, batch_size=32)
        assert isinstance(imp._batch_size, int)
        assert imp._batch_size > 0

    def test_explainer_end_to_end_with_batching(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        arch = _CountingCNN(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(masks),
        )
        explainer = ImageExplainer(
            model=arch,
            data=image,
            batch_size=3,
            max_order=2,
            random_state=0,
        )
        result = explainer.explain_function(image, budget=16)
        assert result.n_players == 4
        # The architecture saw multiple sub-batch calls during approximation.
        assert sum(b > 0 for b in arch.call_batch_sizes) > 1
