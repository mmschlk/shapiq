"""Tests for ``shapiq.vision.architecture``.

The current package exposes two concrete architecture strategies:
:class:`ClassificationArchitecture` (pixel-space masking) and
:class:`ViTClassificationArchitecture` (token-space masking).  Both cache
image-dependent state in :meth:`prepare` and evaluate coalitions in
:meth:`value_function`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shapiq.vision.architecture import (
    ClassificationArchitecture,
    ModelArchitecture,
    ViTClassificationArchitecture,
)
from shapiq.vision.custom_types import CoalitionDomain
from shapiq.vision.masking import (
    BoolMaskedPosStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    ZeroMasking,
)
from shapiq.vision.players import PatchStrategy, SuperpixelStrategy

from .conftest import (
    ChannelSumModel,
    FixedMasksStrategy,
    MockBatchProcessor,
    MockViT,
    MockViTProcessor,
    PixelValuesSumModel,
)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class TestClassificationArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = ClassificationArchitecture(model=ChannelSumModel())
        assert isinstance(arch, ModelArchitecture)

    def test_default_player_strategy(self) -> None:
        arch = ClassificationArchitecture(model=ChannelSumModel())
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, SuperpixelStrategy)
        assert strategy.n_segments == 10

    def test_default_masking_strategy(self) -> None:
        arch = ClassificationArchitecture(model=ChannelSumModel())
        assert isinstance(arch.default_masking_strategy(), MeanColorMasking)

    def test_explicit_masking_strategy_used(self) -> None:
        zero = ZeroMasking()
        arch = ClassificationArchitecture(model=ChannelSumModel(), masking_strategy=zero)
        assert arch._masking_strategy is zero

    def test_prepare_caches_player_masks(self, tiny_image, two_player_masks) -> None:
        arch = ClassificationArchitecture(
            model=ChannelSumModel(), player_strategy=FixedMasksStrategy(two_player_masks)
        )
        assert not hasattr(arch, "_player_masks")
        arch.prepare(tiny_image)
        assert arch._player_masks is not None
        np.testing.assert_array_equal(_to_numpy(arch.player_masks), two_player_masks)

    def test_prepare_sets_class_id(self, tiny_image, two_player_masks) -> None:
        arch = ClassificationArchitecture(
            model=ChannelSumModel(), player_strategy=FixedMasksStrategy(two_player_masks)
        )
        arch.prepare(tiny_image)
        # ChannelSumModel class-0 logit (positive sum) wins.
        assert arch._class_id == 0

    def test_prepare_class_index_overrides_argmax(self, tiny_image, two_player_masks) -> None:
        arch = ClassificationArchitecture(
            model=ChannelSumModel(), player_strategy=FixedMasksStrategy(two_player_masks)
        )
        arch.prepare(tiny_image, class_index=1)
        assert arch._class_id == 1

    def test_value_function_returns_value_per_coalition(self, tiny_image, two_player_masks) -> None:
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        coalitions = torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
            ]
        )
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (4,)
        # Full coalition (no masking) equals the model's sum over the whole image.
        np.testing.assert_allclose(out[3], tiny_image.sum())
        # Empty coalition under ZeroMasking is exactly zero.
        assert out[0] == pytest.approx(0.0)

    def test_value_function_linear_decomposition(self, tiny_image, two_player_masks) -> None:
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        coalitions = torch.tensor([[True, False], [False, True]])
        out = _to_numpy(arch.value_function(coalitions))
        np.testing.assert_allclose(out[0], tiny_image[:, :2].sum())
        np.testing.assert_allclose(out[1], tiny_image[:, 2:].sum())

    def test_default_superpixel_strategy_end_to_end(self) -> None:
        pytest.importorskip("skimage")
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(32, 32, 3)).astype(np.float64)
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=SuperpixelStrategy(n_segments=4),
        )
        arch.prepare(image)
        assert arch._player_masks is not None
        assert arch._player_masks.shape[0] == arch._player_strategy.n_players
        assert (arch._player_masks.cpu().numpy().sum(axis=0) == 1).all()
        out = _to_numpy(
            arch.value_function(torch.tensor([[True] * arch._player_strategy.n_players]))
        )
        assert np.isfinite(out).all()


class TestViTClassificationArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = ViTClassificationArchitecture(model=MockViT(), vit_processor=MockViTProcessor())
        assert isinstance(arch, ModelArchitecture)

    def test_default_player_strategy_uses_model_config(self) -> None:
        arch = ViTClassificationArchitecture(model=MockViT(), vit_processor=MockViTProcessor())
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, PatchStrategy)
        assert strategy.grid_size == 3  # 24 // 8
        assert strategy.n_players == 9

    def test_default_masking_strategy(self) -> None:
        arch = ViTClassificationArchitecture(model=MockViT(), vit_processor=MockViTProcessor())
        assert isinstance(arch.default_masking_strategy(), MaskTokenStrategy)

    def test_init_succeeds_for_standard_vit(self) -> None:
        """ViT-B/16 uses a 14x14 token grid; construction must not raise on the default."""
        model = MockViT(image_size=224, patch_size=16, hidden_size=768)
        arch = ViTClassificationArchitecture(model=model, vit_processor=MockViTProcessor())
        assert isinstance(arch.default_player_strategy(), PatchStrategy)

    def test_default_player_strategy_adapts_to_standard_vit_grid(self) -> None:
        """``default_player_strategy()`` yields a valid grid for ViT-B/16's 14x14 tokens."""
        model = MockViT(image_size=224, patch_size=16, hidden_size=768)
        arch = ViTClassificationArchitecture(model=model, vit_processor=MockViTProcessor())
        strategy = arch.default_player_strategy()
        assert strategy.grid_size == 14
        # 14 is only divisible by 1, 2, 7 and 14, so the 3x3 default falls back to 2x2.
        assert strategy.n_players == 4
        assert strategy.grid_size % strategy.side == 0

    def test_prepare_sets_class_id_and_caches_state(self, image_24x24) -> None:
        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        assert arch._class_id == 0  # logits [2.0, 0.5] -> argmax 0
        assert arch._pixel_values is not None
        assert arch._pixel_values.shape == (1, 3, 24, 24)
        assert arch._token_masks is not None

    def test_prepare_class_index_overrides_argmax(self, image_24x24) -> None:
        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24, class_index=1)
        assert arch._class_id == 1

    def test_value_function_shape_and_monotonicity(self, image_24x24) -> None:
        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        coalitions = torch.tensor(
            [
                [False] * 9,
                [True] + [False] * 8,
                [True] * 5 + [False] * 4,
                [True] * 9,
            ]
        )
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (4,)
        assert np.isfinite(out).all()
        # More visible tokens -> higher class-0 probability.
        assert out[0] < out[1] < out[2] < out[3]

    def test_value_function_empty_coalition_is_half(self, image_24x24) -> None:
        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        out = _to_numpy(arch.value_function(torch.tensor([[False] * 9])))
        # No visible tokens -> logits (0, 0) -> softmax 0.5 for class 0.
        assert out[0] == pytest.approx(0.5, abs=1e-5)

    def test_value_function_with_mask_token_strategy(self, image_24x24) -> None:
        model = MockViT()
        arch = ViTClassificationArchitecture(
            model=model,
            vit_processor=MockViTProcessor(),
            masking_strategy=MaskTokenStrategy(model),
        )
        arch.prepare(image_24x24)
        coalitions = torch.tensor(
            [
                [False] * 9,
                [True] + [False] * 8,
                [True] * 5 + [False] * 4,
                [True] * 9,
            ]
        )
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (4,)
        assert out[0] < out[1] < out[2] < out[3]
        assert torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))


class TestArchitectureOutputScale:
    def test_cnn_returns_logits_transformer_returns_probabilities(
        self,
        tiny_image,
        two_player_masks,
        transformer_architecture,
        image_24x24,
    ) -> None:
        cnn = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        cnn.prepare(tiny_image)
        cnn_full = float(_to_numpy(cnn.value_function(torch.tensor([[True, True]])))[0])
        cnn_empty = float(_to_numpy(cnn.value_function(torch.tensor([[False, False]])))[0])
        # Channel-sum logits are unbounded and not confined to [0, 1].
        assert cnn_full > 1.0
        assert cnn_empty == pytest.approx(0.0)

        transformer_architecture.prepare(image_24x24)
        vit_full = float(
            _to_numpy(transformer_architecture.value_function(torch.tensor([[True] * 9])))[0]
        )
        vit_empty = float(
            _to_numpy(transformer_architecture.value_function(torch.tensor([[False] * 9])))[0]
        )
        assert 0.0 <= vit_full <= 1.0
        assert 0.0 <= vit_empty <= 1.0


class TestCoalitionDomainValidation:
    """A player strategy and a masker must agree on a domain, and it must be the architecture's.

    The failure is silent if unchecked: pixel masks handed to a token masker would
    index the wrong axis instead of raising, so construction rejects the pairing.
    """

    def test_declared_domains(self) -> None:
        assert ClassificationArchitecture.coalition_domain is CoalitionDomain.PIXEL
        assert ViTClassificationArchitecture.coalition_domain is CoalitionDomain.TOKEN

    def test_pixel_players_with_token_masker_rejected(self, two_player_masks) -> None:
        with pytest.raises(TypeError, match="incompatible"):
            ClassificationArchitecture(
                model=MockViT(),
                masking_strategy=BoolMaskedPosStrategy(),
                player_strategy=FixedMasksStrategy(two_player_masks),
            )

    def test_mismatch_message_names_both_strategies(self, two_player_masks) -> None:
        with pytest.raises(TypeError) as err:
            ClassificationArchitecture(
                model=MockViT(),
                masking_strategy=BoolMaskedPosStrategy(),
                player_strategy=FixedMasksStrategy(two_player_masks),
            )
        assert "FixedMasksStrategy" in str(err.value)
        assert "BoolMaskedPosStrategy" in str(err.value)

    def test_token_players_with_pixel_masker_rejected(self) -> None:
        with pytest.raises(TypeError, match="incompatible"):
            ViTClassificationArchitecture(
                model=MockViT(),
                vit_processor=MockViTProcessor(),
                masking_strategy=MeanColorMasking(),
                player_strategy=PatchStrategy(grid_size=3, n_players=9),
            )

    def test_token_strategies_rejected_by_pixel_architecture(self) -> None:
        """Consistent token strategies still fail: ClassificationArchitecture is pixel-space."""
        with pytest.raises(TypeError, match="ViTClassificationArchitecture"):
            ClassificationArchitecture(
                model=MockViT(),
                masking_strategy=BoolMaskedPosStrategy(),
                player_strategy=PatchStrategy(grid_size=3, n_players=9),
            )

    def test_pixel_strategies_rejected_by_token_architecture(self, two_player_masks) -> None:
        """The mirror case points users back at ClassificationArchitecture."""
        with pytest.raises(TypeError, match="ClassificationArchitecture"):
            ViTClassificationArchitecture(
                model=MockViT(),
                vit_processor=MockViTProcessor(),
                masking_strategy=MeanColorMasking(),
                player_strategy=FixedMasksStrategy(two_player_masks),
            )


class TestClassificationArchitectureModelValidation:
    def test_non_callable_model_rejected(self) -> None:
        with pytest.raises(TypeError, match="VisionModel"):
            ClassificationArchitecture(model=object())

    def test_non_callable_processor_rejected(self) -> None:
        with pytest.raises(TypeError, match="processor must be callable"):
            ClassificationArchitecture(model=ChannelSumModel(), processor=object())

    def test_uncallable_model_interface_surfaces_as_type_error(
        self, tiny_image, two_player_masks
    ) -> None:
        """A callable model that rejects a (B, C, H, W) batch fails with a readable error."""

        class _WrongSignatureModel:
            def __call__(self, *args, **kwargs):
                msg = "unexpected argument"
                raise ValueError(msg)

        arch = ClassificationArchitecture(
            model=_WrongSignatureModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        with pytest.raises(TypeError, match="expected classification interface"):
            arch.prepare(tiny_image)

    def test_model_without_logits_rejected(self, tiny_image, two_player_masks) -> None:
        """Encoder-only backbones return no logits and cannot be explained."""

        class _NoLogitsModel:
            def __call__(self, batch):
                return SimpleNamespace(last_hidden_state=torch.zeros(batch.shape[0], 4))

        arch = ClassificationArchitecture(
            model=_NoLogitsModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        with pytest.raises(TypeError, match="no classification logits"):
            arch.prepare(tiny_image)


class TestClassificationArchitectureErrors:
    def test_value_function_before_prepare_raises(self, two_player_masks) -> None:
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        with pytest.raises(RuntimeError, match="Call prepare"):
            arch.value_function(torch.tensor([[True, True]]))

    def test_out_of_range_class_index_raises(self, tiny_image, two_player_masks) -> None:
        """ChannelSumModel emits 2 classes, so class 5 cannot be scored."""
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image, class_index=5)
        with pytest.raises(RuntimeError, match="could not extract the score for class index 5"):
            arch.value_function(torch.tensor([[True, True]]))

    def test_negative_class_index_is_valid(self, tiny_image, two_player_masks) -> None:
        """Negative indices are ordinary Python indexing, so -1 selects the last class."""
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image, class_index=-1)
        out = _to_numpy(arch.value_function(torch.tensor([[True, True]])))
        np.testing.assert_allclose(out[0], -tiny_image.sum())


class TestClassificationArchitectureProcessor:
    """The processor path lets any HF classifier (Swin, BEiT, ...) run with pixel masking."""

    def _build(self, masks, processor):
        return ClassificationArchitecture(
            model=PixelValuesSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(masks),
            processor=processor,
        )

    def test_processor_applied_to_every_masked_image(self, tiny_image, two_player_masks) -> None:
        processor = MockBatchProcessor(scale=2.0)
        arch = self._build(two_player_masks, processor)
        arch.prepare(tiny_image)
        out = _to_numpy(arch.value_function(torch.tensor([[True, True]])))
        # The processor doubles every pixel before the forward pass.
        np.testing.assert_allclose(out[0], 2.0 * tiny_image.sum())

    def test_processor_receives_one_image_per_coalition(self, tiny_image, two_player_masks) -> None:
        processor = MockBatchProcessor()
        arch = self._build(two_player_masks, processor)
        arch.prepare(tiny_image)
        processor.seen_batch_sizes.clear()
        arch.value_function(torch.tensor([[True, True], [True, False], [False, False]]))
        assert processor.seen_batch_sizes == [3]

    def test_model_called_with_pixel_values_keyword(self, tiny_image, two_player_masks) -> None:
        """With a processor the batch goes in as ``pixel_values``, not positionally."""
        seen = {}

        class _KeywordOnlyModel:
            def __call__(self, *, pixel_values):
                seen["shape"] = tuple(pixel_values.shape)
                return SimpleNamespace(
                    logits=torch.zeros(pixel_values.shape[0], 2),
                )

        arch = ClassificationArchitecture(
            model=_KeywordOnlyModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
            processor=MockBatchProcessor(),
        )
        arch.prepare(tiny_image)
        arch.value_function(torch.tensor([[True, True]]))
        assert seen["shape"] == (1, 3, 4, 4)

    def test_unit_interval_float_image_is_rescaled_for_processor(self, two_player_masks) -> None:
        """Processors expect 0-255 uint8, so [0, 1] floats are scaled up before masking."""
        image = np.ones((4, 4, 3), dtype=np.float64)
        arch = self._build(two_player_masks, MockBatchProcessor())
        arch.prepare(image)
        out = _to_numpy(arch.value_function(torch.tensor([[True, True]])))
        np.testing.assert_allclose(out[0], 255.0 * image.size)

    def test_uint8_image_is_not_rescaled(self, two_player_masks) -> None:
        image = np.full((4, 4, 3), 10, dtype=np.uint8)
        arch = self._build(two_player_masks, MockBatchProcessor())
        arch.prepare(image)
        out = _to_numpy(arch.value_function(torch.tensor([[True, True]])))
        np.testing.assert_allclose(out[0], 10.0 * image.size)

    def test_masked_pixels_survive_the_processor_roundtrip(
        self, tiny_image, two_player_masks
    ) -> None:
        processor = MockBatchProcessor()
        arch = self._build(two_player_masks, processor)
        arch.prepare(tiny_image)
        out = _to_numpy(arch.value_function(torch.tensor([[True, False], [False, True]])))
        np.testing.assert_allclose(out[0], tiny_image[:, :2].sum())
        np.testing.assert_allclose(out[1], tiny_image[:, 2:].sum())

    def test_failing_processor_surfaces_as_type_error(self, tiny_image, two_player_masks) -> None:
        class _BrokenProcessor:
            def __call__(self, images=None, return_tensors="pt"):
                msg = "boom"
                raise RuntimeError(msg)

        arch = self._build(two_player_masks, _BrokenProcessor())
        with pytest.raises(TypeError, match="could not preprocess images"):
            arch.prepare(tiny_image)

    def test_processor_without_pixel_values_key_rejected(
        self, tiny_image, two_player_masks
    ) -> None:
        class _WrongKeyProcessor:
            def __call__(self, images=None, return_tensors="pt"):
                return {"inputs": torch.zeros(len(images), 3, 4, 4)}

        arch = self._build(two_player_masks, _WrongKeyProcessor())
        with pytest.raises(TypeError, match="could not preprocess images"):
            arch.prepare(tiny_image)


class TestViTArchitectureValidation:
    def test_non_callable_model_rejected(self) -> None:
        with pytest.raises(TypeError, match="VisionModel"):
            ViTClassificationArchitecture(model=object(), vit_processor=MockViTProcessor())

    def test_non_callable_processor_rejected(self) -> None:
        with pytest.raises(TypeError, match="vit_processor must be callable"):
            ViTClassificationArchitecture(model=MockViT(), vit_processor=object())

    def test_default_player_strategy_requires_config(self) -> None:
        class _NoConfigViT:
            def __call__(self, **kwargs):
                return torch.zeros(1, 2)

        with pytest.raises(TypeError, match="requires a model exposing a ``config``"):
            ViTClassificationArchitecture(model=_NoConfigViT(), vit_processor=MockViTProcessor())

    def test_default_player_strategy_requires_patch_size(self) -> None:
        model = MockViT()
        model.config.patch_size = None
        with pytest.raises(TypeError, match="``patch_size``"):
            ViTClassificationArchitecture(model=model, vit_processor=MockViTProcessor())

    def test_default_player_strategy_rejects_zero_patch_size(self) -> None:
        """A zero patch size would divide by zero when deriving the token grid."""
        model = MockViT(patch_size=0)
        with pytest.raises(TypeError, match="non-zero"):
            ViTClassificationArchitecture(model=model, vit_processor=MockViTProcessor())

    def test_missing_config_error_suggests_explicit_player_strategy(self) -> None:
        model = MockViT()
        model.config.image_size = None
        with pytest.raises(TypeError, match="explicit"):
            ViTClassificationArchitecture(model=model, vit_processor=MockViTProcessor())

    def test_explicit_player_strategy_skips_config_requirements(self) -> None:
        """An explicit strategy means the patch grid never has to be derived from config."""
        model = MockViT()
        model.config.patch_size = None
        arch = ViTClassificationArchitecture(
            model=model,
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
            player_strategy=PatchStrategy(grid_size=3, n_players=9),
        )
        assert arch.n_players == 9


class TestViTArchitectureErrors:
    def test_value_function_before_prepare_raises(self) -> None:
        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        with pytest.raises(RuntimeError, match="Call prepare"):
            arch.value_function(torch.tensor([[True] * 9]))

    def test_out_of_range_class_index_raises(self, image_24x24) -> None:
        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24, class_index=7)
        with pytest.raises(RuntimeError, match="could not extract the score for class index 7"):
            arch.value_function(torch.tensor([[True] * 9]))

    def test_failing_processor_surfaces_as_type_error(self, image_24x24) -> None:
        class _BrokenProcessor:
            def __call__(self, images=None, return_tensors="pt"):
                msg = "boom"
                raise RuntimeError(msg)

        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=_BrokenProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        with pytest.raises(TypeError, match="could not preprocess the input image"):
            arch.prepare(image_24x24)

    def test_model_rejecting_bool_masked_pos_surfaces_as_type_error(self, image_24x24) -> None:
        """Models without ``bool_masked_pos`` support must be told to use the pixel path."""

        class _NoMaskSupportViT(MockViT):
            def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
                if bool_masked_pos is not None:
                    msg = "unexpected keyword argument 'bool_masked_pos'"
                    raise TypeError(msg)
                return SimpleNamespace(logits=torch.zeros(pixel_values.shape[0], 2))

        arch = ViTClassificationArchitecture(
            model=_NoMaskSupportViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        with pytest.raises(TypeError, match="bool_masked_pos"):
            arch.value_function(torch.tensor([[True] * 9]))


class TestArchitectureProperties:
    def test_classification_model_property_returns_underlying_model(self, two_player_masks) -> None:
        model = ChannelSumModel()
        arch = ClassificationArchitecture(
            model=model,
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        assert arch.model is model

    def test_vit_model_property_returns_underlying_model(self) -> None:
        model = MockViT()
        arch = ViTClassificationArchitecture(
            model=model,
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        assert arch.model is model

    def test_n_players_delegates_to_player_strategy(self, three_player_masks) -> None:
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(three_player_masks),
        )
        assert arch.n_players == 3

    def test_vit_player_masks_are_pixel_space(self, image_24x24) -> None:
        """Players live in token space, but the masks exposed for plotting are pixels."""
        arch = ViTClassificationArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        assert arch.player_masks.shape == (9, 24, 24)
        assert arch.player_masks.dtype == torch.bool

    def test_prepare_twice_keeps_first_class_id(self, tiny_image, two_player_masks) -> None:
        """Re-preparing on a new image keeps the tracked class so values stay comparable."""
        arch = ClassificationArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image, class_index=1)
        arch.prepare(tiny_image)
        assert arch._class_id == 1
