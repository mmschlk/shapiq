"""Tests for the vision imputer module (pluggable segmenters and maskers)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from shapiq.imputer.vision import (
    CrossModalBlurMasker,
    CrossModalBlurParams,
    CrossModalMeanMasker,
    CrossModalMeanParams,
    CustomSegmenter,
    MaskerConfig,
    PatchParams,
    PatchSegmenter,
    PhysicalMask,
    ProcessorOutput,
    SegmenterConfig,
    SlicParams,
    SpatialLayout,
    TextAttentionMasker,
    VisionBlurMasker,
    VisionBlurParams,
    VisionImputer,
    VisionImputerFactory,
    VisionLanguageGame,
    VisionMeanMasker,
    VisionMeanParams,
)
from shapiq.imputer.vision.maskers import get_masker
from shapiq.imputer.vision.segmenters import get_segmenter

# ═══════════════════════════════════════════════════════════════════════
# Data type tests
# ═══════════════════════════════════════════════════════════════════════


class TestSegmenterConfig:
    def test_default_strategy(self):
        cfg = SegmenterConfig()
        assert cfg.strategy == "patch"
        assert isinstance(cfg.patch, PatchParams)

    def test_slic_config(self):
        cfg = SegmenterConfig(strategy="slic", slic=SlicParams(n_segments=60))
        assert cfg.strategy == "slic"
        assert cfg.slic.n_segments == 60

    def test_active_params(self):
        cfg = SegmenterConfig(strategy="slic", slic=SlicParams(n_segments=60))
        assert cfg.active_params is cfg.slic

    def test_factory_populated_metadata(self):
        cfg = SegmenterConfig()
        cfg.model_type = "clip"
        cfg.image_size = 224
        cfg.patch_size = 32
        cfg.grid_size = 7
        cfg.n_players_image = 49
        cfg.n_players_text = 8
        assert cfg.model_type == "clip"
        assert cfg.grid_size * cfg.grid_size == cfg.n_players_image


class TestMaskerConfig:
    def test_default_strategy(self):
        cfg = MaskerConfig()
        assert cfg.strategy == "crossmodal_mean"
        assert isinstance(cfg.crossmodal_mean, CrossModalMeanParams)

    def test_blur_strategy(self):
        cfg = MaskerConfig(strategy="crossmodal_blur")
        assert cfg.strategy == "crossmodal_blur"
        assert isinstance(cfg.crossmodal_blur, CrossModalBlurParams)

    def test_vision_mean_params(self):
        cfg = MaskerConfig(strategy="vision_mean")
        assert isinstance(cfg.vision_mean, VisionMeanParams)

    def test_vision_blur_params(self):
        cfg = MaskerConfig(strategy="vision_blur")
        assert isinstance(cfg.vision_blur, VisionBlurParams)


class TestSpatialLayout:
    def test_creation(self):
        layout = SpatialLayout(
            n_players_image=49,
            n_players_text=8,
            image_size=224,
            patch_size=32,
            grid_size=7,
            n_channels=3,
            model_type="clip",
            text_total_length=10,
        )
        assert layout.n_players_image == 49
        assert layout.n_players_text == 8
        assert not layout.is_stateful

    def test_stateful_flag(self):
        layout = SpatialLayout(
            n_players_image=100,
            n_players_text=10,
            image_size=224,
            patch_size=32,
            grid_size=7,
            n_channels=3,
            model_type="clip",
            text_total_length=64,
            is_stateful=True,
        )
        assert layout.is_stateful


class TestPhysicalMask:
    def test_batch_properties(self):
        pm = PhysicalMask(
            image_binary_mask=torch.randn(4, 3, 224, 224),
            text_attention_mask=torch.ones(8, 10),
        )
        assert pm.batch_size_img == 4
        assert pm.batch_size_txt == 8

    def test_empty_mask(self):
        pm = PhysicalMask()
        assert pm.batch_size_img == 0
        assert pm.batch_size_txt == 0


class TestProcessorOutput:
    def test_from_hf_processor(self):
        hf_dict = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 10)),
        }
        po = ProcessorOutput.from_hf_processor(hf_dict, "clip")
        assert po.model_type == "clip"
        assert po.attention_mask is not None  # derived

    def test_to_dict(self):
        po = ProcessorOutput(
            pixel_values=torch.randn(1, 3, 224, 224),
            input_ids=torch.randint(0, 100, (1, 10)),
            attention_mask=torch.ones(1, 10, dtype=torch.int),
            model_type="clip",
        )
        d = po.to_dict()
        assert "pixel_values" in d
        assert "input_ids" in d
        assert "attention_mask" in d

    def test_device_property(self):
        po = ProcessorOutput(
            pixel_values=torch.randn(1, 3, 32, 32),
            input_ids=torch.randint(0, 100, (1, 5)),
            attention_mask=torch.ones(1, 5, dtype=torch.int),
            model_type="clip",
        )
        assert isinstance(po.device, torch.device)

    def test_to_device(self):
        po = ProcessorOutput(
            pixel_values=torch.randn(1, 3, 32, 32),
            input_ids=torch.randint(0, 100, (1, 5)),
            attention_mask=torch.ones(1, 5, dtype=torch.int),
            model_type="clip",
        )
        po.to("cpu")
        assert po.pixel_values.device.type == "cpu"


# ═══════════════════════════════════════════════════════════════════════
# Segmenter tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def patch_config() -> SegmenterConfig:
    return SegmenterConfig(
        strategy="patch",
        model_type="clip",
        image_size=224,
        patch_size=32,
        n_channels=3,
        grid_size=7,
        n_players_image=49,
        n_players_text=8,
        text_total_length=10,
    )


class TestPatchSegmenter:
    def test_get_layout(self, patch_config):
        seg = PatchSegmenter(patch_config)
        layout = seg.get_layout()
        assert layout.n_players_image == 49
        assert layout.n_players_text == 8
        assert layout.grid_size == 7
        assert not layout.is_stateful

    def test_generate_image_mask_shape(self, patch_config):
        seg = PatchSegmenter(patch_config)
        mask = seg.generate_masks(coalitions_image=np.ones((2, 49), dtype=bool))
        assert mask.image_binary_mask is not None
        assert mask.image_binary_mask.shape == (2, 3, 224, 224)

    def test_generate_text_mask_shape(self, patch_config):
        seg = PatchSegmenter(patch_config)
        mask = seg.generate_masks(coalitions_text=np.ones((3, 8), dtype=bool))
        assert mask.text_attention_mask is not None
        assert mask.text_attention_mask.shape == (3, 10)

    def test_generate_both_masks(self, patch_config):
        seg = PatchSegmenter(patch_config)
        mask = seg.generate_masks(
            coalitions_image=np.ones((2, 49), dtype=bool),
            coalitions_text=np.ones((3, 8), dtype=bool),
        )
        assert mask.image_binary_mask.shape == (2, 3, 224, 224)
        assert mask.text_attention_mask.shape == (3, 10)

    def test_clip_text_mask_format(self, patch_config):
        seg = PatchSegmenter(patch_config)
        mask = seg.generate_masks(coalitions_text=np.zeros((1, 8), dtype=bool))
        attn = mask.text_attention_mask
        assert attn[0, 0] == 1  # BOS
        assert attn[0, -1] == 1  # EOS
        assert attn[0, 1:-1].sum() == 0.0  # all tokens occluded

    def test_image_mask_all_present(self, patch_config):
        seg = PatchSegmenter(patch_config)
        mask = seg.generate_masks(coalitions_image=np.ones((1, 49), dtype=bool))
        total_pixels = 1 * 3 * 224 * 224
        assert mask.image_binary_mask.sum() == pytest.approx(float(total_pixels))

    def test_image_mask_all_occluded(self, patch_config):
        seg = PatchSegmenter(patch_config)
        mask = seg.generate_masks(coalitions_image=np.zeros((1, 49), dtype=bool))
        assert mask.image_binary_mask.sum() == 0.0


class TestSegmenterRegistry:
    def test_get_patch_segmenter(self):
        cls = get_segmenter("patch")
        assert cls is PatchSegmenter

    def test_get_slic_segmenter(self):
        cls = get_segmenter("slic")
        assert cls.__name__ == "SLICSegmenter"

    def test_get_custom_segmenter(self):
        cls = get_segmenter("custom_segmenter")
        assert cls.__name__ == "CustomSegmenter"

    def test_unknown_segmenter_raises(self):
        with pytest.raises(KeyError):
            get_segmenter("nonexistent")


# ═══════════════════════════════════════════════════════════════════════
# CustomSegmenter tests
# ═══════════════════════════════════════════════════════════════════════


class TestCustomSegmenter:
    """CustomSegmenter: user-provided binary masks as players."""

    def test_requires_masks(self):
        with pytest.raises(ValueError, match="requires.*masks"):
            CustomSegmenter(SegmenterConfig(strategy="custom_segmenter"))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="3D"):
            CustomSegmenter(
                SegmenterConfig(strategy="custom_segmenter"),
                masks=np.ones((2, 3, 4, 5)),
            )

    def test_get_layout(self):
        masks = np.zeros((4, 16, 16), dtype=bool)
        masks[0, :4, :4] = True
        seg = CustomSegmenter(
            SegmenterConfig(strategy="custom_segmenter", n_channels=3),
            masks=masks,
        )
        layout = seg.get_layout()
        assert layout.n_players_image == 4
        assert layout.image_size == 16

    def test_generate_mask_shape(self):
        masks = np.zeros((3, 8, 8), dtype=bool)
        masks[0, :4, :4] = True
        seg = CustomSegmenter(
            SegmenterConfig(strategy="custom_segmenter", n_channels=1),
            masks=masks,
        )
        pm = seg.generate_masks(coalitions_image=np.ones((2, 3), dtype=bool))
        assert pm.image_binary_mask.shape == (2, 1, 8, 8)

    def test_generate_mask_union(self):
        """Only selected players' pixels are kept."""
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :2, :] = True  # player 0: top half
        masks[1, 2:, :] = True  # player 1: bottom half
        seg = CustomSegmenter(
            SegmenterConfig(strategy="custom_segmenter", n_channels=1),
            masks=masks,
        )
        # Only player 0 active
        pm = seg.generate_masks(coalitions_image=np.array([[True, False]]))
        kept = pm.image_binary_mask[0, 0]
        assert kept[:2, :].sum() == 2 * 4  # top half kept
        assert kept[2:, :].sum() == 0.0  # bottom half occluded

        # Both players active → full image kept
        pm = seg.generate_masks(coalitions_image=np.array([[True, True]]))
        assert pm.image_binary_mask[0, 0].sum() == 4 * 4

    def test_generate_mask_no_text(self):
        """Without coalitions_text, text_attention_mask stays None."""
        masks = np.ones((2, 8, 8), dtype=bool)
        seg = CustomSegmenter(
            SegmenterConfig(strategy="custom_segmenter", n_channels=3),
            masks=masks,
        )
        pm = seg.generate_masks(coalitions_image=np.ones((1, 2), dtype=bool))
        assert pm.text_attention_mask is None


# ═══════════════════════════════════════════════════════════════════════
# Masker tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def dummy_inputs() -> ProcessorOutput:
    return ProcessorOutput(
        pixel_values=torch.randn(2, 3, 32, 32),
        input_ids=torch.randint(0, 100, (2, 5)),
        attention_mask=torch.ones(2, 5, dtype=torch.int),
        model_type="clip",
    )


class TestVisionMeanMasker:
    def test_applies_mask(self, dummy_inputs):
        masker = VisionMeanMasker()
        pm = PhysicalMask(image_binary_mask=torch.ones(2, 3, 32, 32))
        result = masker.apply(dummy_inputs, pm)
        assert result is not dummy_inputs  # no mutation
        assert result.pixel_values.shape == (2, 3, 32, 32)

    def test_zeroes_out_occluded_pixels(self, dummy_inputs):
        masker = VisionMeanMasker()
        pm = PhysicalMask(image_binary_mask=torch.zeros(2, 3, 32, 32))
        result = masker.apply(dummy_inputs, pm)
        assert result.pixel_values.sum() == 0.0

    def test_no_mask_pass_through(self, dummy_inputs):
        masker = VisionMeanMasker()
        result = masker.apply(dummy_inputs, PhysicalMask())
        assert torch.equal(result.pixel_values, dummy_inputs.pixel_values)
        assert torch.equal(result.input_ids, dummy_inputs.input_ids)

    def test_preserves_text(self, dummy_inputs):
        masker = VisionMeanMasker()
        pm = PhysicalMask(image_binary_mask=torch.zeros(2, 3, 32, 32))
        result = masker.apply(dummy_inputs, pm)
        assert torch.equal(result.input_ids, dummy_inputs.input_ids)
        assert torch.equal(result.attention_mask, dummy_inputs.attention_mask)


class TestTextAttentionMasker:
    def test_swaps_attention_mask(self, dummy_inputs):
        masker = TextAttentionMasker()
        new_mask = torch.zeros(2, 5, dtype=torch.int)
        pm = PhysicalMask(text_attention_mask=new_mask)
        result = masker.apply(dummy_inputs, pm)
        assert torch.equal(result.attention_mask, new_mask)

    def test_preserves_image(self, dummy_inputs):
        masker = TextAttentionMasker()
        pm = PhysicalMask(text_attention_mask=torch.zeros(2, 5, dtype=torch.int))
        result = masker.apply(dummy_inputs, pm)
        assert torch.equal(result.pixel_values, dummy_inputs.pixel_values)


class TestCrossModalMeanMasker:
    def test_composite_apply(self, dummy_inputs):
        masker = CrossModalMeanMasker()
        pm = PhysicalMask(
            image_binary_mask=torch.ones(2, 3, 32, 32),
            text_attention_mask=torch.ones(2, 5, dtype=torch.int),
        )
        result = masker.apply(dummy_inputs, pm)
        assert result.pixel_values.shape == (2, 3, 32, 32)
        assert result.attention_mask.shape == (2, 5)


# ═══════════════════════════════════════════════════════════════════════
# Blur masker tests
# ═══════════════════════════════════════════════════════════════════════


class TestVisionBlurMasker:
    """VisionBlurMasker: Gaussian blur occlusion. Requires B3.2."""

    def test_blur_applies_gaussian(self, dummy_inputs):
        """Masked region is blurred, not zeroed."""
        pytest.importorskip("skimage.filters")
        masker = VisionBlurMasker()
        H, W = dummy_inputs.pixel_values.shape[2:]
        # Mask: top half blurred (0), bottom half kept (1)
        mask = torch.ones(2, 3, H, W)
        mask[:, :, : H // 2, :] = 0.0
        pm = PhysicalMask(image_binary_mask=mask)
        result = masker.apply(dummy_inputs, pm)
        # Shape preserved
        assert result.pixel_values.shape == dummy_inputs.pixel_values.shape
        # Kept half equals original (blend: original*1 + blurred*0)
        assert torch.equal(
            result.pixel_values[:, :, H // 2 :, :],
            dummy_inputs.pixel_values[:, :, H // 2 :, :],
        )
        # Blurred half differs from original
        assert not torch.equal(
            result.pixel_values[:, :, : H // 2, :],
            dummy_inputs.pixel_values[:, :, : H // 2, :],
        )
        # Blurred region is NOT zeroed out (blur, not zero-out)
        assert not torch.allclose(
            result.pixel_values[:, :, : H // 2, :],
            torch.zeros_like(result.pixel_values[:, :, : H // 2, :]),
        )

    def test_sigma_configurable(self):
        """sigma=1.0 vs sigma=5.0 produce different blur."""
        pytest.importorskip("skimage.filters")
        torch.manual_seed(42)
        pixel_values = torch.randn(1, 1, 32, 32)
        inputs = ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=torch.randint(0, 100, (1, 5)),
            attention_mask=torch.ones(1, 5, dtype=torch.int),
            model_type="clip",
        )
        # Occlude the center rectangle
        mask = torch.ones(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 0.0
        pm = PhysicalMask(image_binary_mask=mask)
        result_lo = VisionBlurMasker(sigma=1.0).apply(inputs, pm)
        result_hi = VisionBlurMasker(sigma=5.0).apply(inputs, pm)
        # Different sigmas → different blurred values in occluded region
        assert not torch.equal(
            result_lo.pixel_values[:, :, 8:24, 8:24],
            result_hi.pixel_values[:, :, 8:24, 8:24],
        )
        # Kept region (border) identical to original for both
        assert torch.equal(
            result_lo.pixel_values[:, :, :8, :],
            inputs.pixel_values[:, :, :8, :],
        )

    def test_no_mask_pass_through(self, dummy_inputs):
        """No PhysicalMask.image_binary_mask → unchanged."""
        pytest.importorskip("skimage.filters")
        masker = VisionBlurMasker()
        result = masker.apply(dummy_inputs, PhysicalMask())
        assert torch.equal(result.pixel_values, dummy_inputs.pixel_values)
        assert torch.equal(result.input_ids, dummy_inputs.input_ids)
        assert torch.equal(result.attention_mask, dummy_inputs.attention_mask)

    def test_preserves_text(self, dummy_inputs):
        """input_ids and attention_mask pass through unchanged."""
        pytest.importorskip("skimage.filters")
        masker = VisionBlurMasker()
        pm = PhysicalMask(image_binary_mask=torch.zeros(2, 3, 32, 32))
        result = masker.apply(dummy_inputs, pm)
        assert torch.equal(result.input_ids, dummy_inputs.input_ids)
        assert torch.equal(result.attention_mask, dummy_inputs.attention_mask)


class TestCrossModalBlurMasker:
    """CrossModalBlurMasker: VisionBlurMasker + TextAttentionMasker composite."""

    def test_composite_blur_and_text(self, dummy_inputs):
        """Both pixel_values (blur) and attention_mask (swap) modified."""
        pytest.importorskip("skimage.filters")
        masker = CrossModalBlurMasker()
        H, W = dummy_inputs.pixel_values.shape[2:]
        # Image: top half occluded
        img_mask = torch.ones(2, 3, H, W)
        img_mask[:, :, : H // 2, :] = 0.0
        new_attn = torch.zeros(2, 5, dtype=torch.int)
        pm = PhysicalMask(
            image_binary_mask=img_mask,
            text_attention_mask=new_attn,
        )
        result = masker.apply(dummy_inputs, pm)
        # pixel_values blurred (not zeroed) in occluded region
        assert result.pixel_values.shape == dummy_inputs.pixel_values.shape
        assert not torch.equal(
            result.pixel_values[:, :, : H // 2, :],
            dummy_inputs.pixel_values[:, :, : H // 2, :],
        )
        # Kept region unchanged
        assert torch.equal(
            result.pixel_values[:, :, H // 2 :, :],
            dummy_inputs.pixel_values[:, :, H // 2 :, :],
        )
        # attention_mask swapped
        assert torch.equal(result.attention_mask, new_attn)
        # input_ids unchanged
        assert torch.equal(result.input_ids, dummy_inputs.input_ids)

    def test_register_key(self):
        """Registered as \"crossmodal_blur\"."""
        assert get_masker("crossmodal_blur") is CrossModalBlurMasker


class TestMaskerRegistry:
    def test_vision_mean(self):
        cls = get_masker("vision_mean")
        assert cls is VisionMeanMasker

    def test_text_attn(self):
        cls = get_masker("text_attn")
        assert cls is TextAttentionMasker

    def test_crossmodal_mean(self):
        cls = get_masker("crossmodal_mean")
        assert cls is CrossModalMeanMasker

    def test_unknown_masker_raises(self):
        with pytest.raises(KeyError):
            get_masker("nonexistent")


# ═══════════════════════════════════════════════════════════════════════
# Factory tests (mock model)
# ═══════════════════════════════════════════════════════════════════════


class TestVisionImputerFactory:
    def test_infer_model_type_clip(self):
        """Model name ``openai/clip-vit-base-patch32`` should resolve to ``clip``."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.name_or_path = "openai/clip-vit-base-patch32"
        assert VisionImputerFactory._infer_model_type(model) == "clip"

    def test_infer_model_type_siglip(self):
        """Model name ``google/siglip-base-patch16-224`` should resolve to ``siglip``."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.name_or_path = "google/siglip-base-patch16-224"
        model.config._name_or_path = "google/siglip-base-patch16-224"
        assert VisionImputerFactory._infer_model_type(model) == "siglip"

    def test_extract_vision_dims_vit(self):
        """PatchSize > 0 indicates a ViT backbone."""
        from unittest.mock import MagicMock

        model = MagicMock()
        embeddings = MagicMock()
        embeddings.patch_size = 32
        embeddings.image_size = 224
        embeddings.config.num_channels = 3
        model.vision_model.embeddings = embeddings
        sz, ps, nc = VisionImputerFactory._extract_vision_dims(model)
        assert sz == 224
        assert ps == 32
        assert nc == 3

    def test_extract_vision_dims_cnn(self):
        """PatchSize == 0 indicates a CNN backbone (no rigid grid)."""
        from unittest.mock import MagicMock

        model = MagicMock()
        # Simulate CNN: no patch_size attribute on embeddings
        embeddings = MagicMock(spec=[])
        del embeddings.patch_size
        del embeddings.image_size
        model.vision_model.embeddings = embeddings
        model.config.vision_config = None
        model.config.image_size = 224
        model.config.num_channels = 3
        sz, ps, nc = VisionImputerFactory._extract_vision_dims(model)
        assert ps == 0
        assert nc == 3

    def test_count_text_players_clip(self):
        """CLIP: n_players_text = input_length - 2 (BOS/EOS)."""
        inputs = {"input_ids": torch.randint(0, 100, (1, 10))}
        assert VisionImputerFactory._count_text_players(inputs, "clip") == 8


# ═══════════════════════════════════════════════════════════════════════
# SLICSegmenter tests (requires scikit-image + test image)
# ═══════════════════════════════════════════════════════════════════════


class TestSLICSegmenter:
    """SLICSegmenter smoke tests using the test_croc.JPEG fixture."""

    @pytest.fixture
    def image(self):
        from PIL import Image

        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "data",
            "test_croc.JPEG",
        )
        return Image.open(path).convert("RGB").resize((64, 64))

    @pytest.fixture
    def config(self):
        return SegmenterConfig(
            strategy="slic",
            slic=SlicParams(n_segments=9, compactness=10.0, sigma=0.0),
            model_type="clip",
            image_size=64,
            n_channels=3,
            n_players_text=8,
            text_total_length=10,
        )

    def test_requires_image_array(self):
        from shapiq.imputer.vision.segmenters.slic import SLICSegmenter

        with pytest.raises(ValueError, match="requires image_array"):
            SLICSegmenter(SegmenterConfig(strategy="slic"))

    def test_slic_get_layout(self, config, image):
        try:
            from shapiq.imputer.vision.segmenters.slic import SLICSegmenter
        except ImportError:
            pytest.skip("scikit-image not available")
        seg = SLICSegmenter(config, image_array=image)
        layout = seg.get_layout()
        assert layout.n_players_image > 0
        assert layout.n_players_text == 8
        assert not layout.is_stateful

    def test_slic_generate_mask(self, config, image):
        try:
            from shapiq.imputer.vision.segmenters.slic import SLICSegmenter
        except ImportError:
            pytest.skip("scikit-image not available")
        seg = SLICSegmenter(config, image_array=image)
        mask = seg.generate_masks(
            coalitions_image=np.ones((1, seg.n_players_image), dtype=bool),
            coalitions_text=np.ones((1, 8), dtype=bool),
        )
        assert mask.image_binary_mask is not None
        assert mask.image_binary_mask.shape[2:] == (64, 64)
        assert mask.text_attention_mask is not None

    def test_slic_coerce_ndarray_input(self, config):
        """ndarray path of _coerce_rgb_uint8: >3 channels, float dtype, resize."""
        try:
            from shapiq.imputer.vision.segmenters.slic import SLICSegmenter
        except ImportError:
            pytest.skip("scikit-image not available")
        # (32, 32, 4) float RGBA in [0, 1], smaller than config.image_size (64)
        rng = np.random.default_rng(0)
        img = rng.random((32, 32, 4))
        seg = SLICSegmenter(config, image_array=img)
        assert seg.get_layout().n_players_image > 0

    def test_slic_label_map_device_cache(self, config, image):
        """_label_map_for caches per device (cache-miss branch, lines 144-145)."""
        try:
            from shapiq.imputer.vision.segmenters.slic import SLICSegmenter
        except ImportError:
            pytest.skip("scikit-image not available")
        seg = SLICSegmenter(config, image_array=image)
        meta = torch.device("meta")          # a device not yet cached
        lm = seg._label_map_for(meta)
        assert lm.device.type == "meta"
        assert meta in seg._label_map_by_device


# ═══════════════════════════════════════════════════════════════════════
# VisionImputer orchestration tests (mock model)
# ═══════════════════════════════════════════════════════════════════════


class TestVisionImputerForward:
    """Test VisionImputer.forward_1d and forward_crossmodal with mocked model."""

    @pytest.fixture
    def model(self):
        """Returns logits_per_image matching CLIP shape: (img_bs, txt_bs)."""
        import types

        class MockModel:
            name_or_path = "openai/clip-vit-base-patch32"

            def parameters(self):
                return iter([torch.nn.Parameter(torch.randn(1))])

            def __call__(self, **kwargs):
                img_bs = kwargs["pixel_values"].shape[0]
                txt_bs = kwargs["input_ids"].shape[0]
                out = types.SimpleNamespace()
                out.logits_per_image = torch.randn(img_bs, txt_bs)
                return out

        return MockModel()

    @pytest.fixture
    def segmenter(self):
        return PatchSegmenter(
            SegmenterConfig(
                strategy="patch",
                model_type="clip",
                image_size=224,
                patch_size=32,
                n_channels=3,
                grid_size=7,
                n_players_image=49,
                n_players_text=8,
                text_total_length=10,
            )
        )

    @pytest.fixture
    def masker(self):
        return CrossModalMeanMasker()

    @pytest.fixture
    def inputs_original(self):
        return ProcessorOutput(
            pixel_values=torch.randn(1, 3, 224, 224),
            input_ids=torch.randint(0, 100, (1, 10)),
            attention_mask=torch.ones(1, 10, dtype=torch.int),
            model_type="clip",
        )

    def test_forward_1d_shape(self, model, segmenter, masker, inputs_original):
        """forward_1d with 4 coalitions, batch_size=2 → shape (4,)."""
        from unittest.mock import MagicMock

        imputer = VisionImputer(
            model=model,
            processor=MagicMock(),
            segmenter=segmenter,
            masker=masker,
            inputs_original=inputs_original,
        )
        result = imputer.forward_1d(np.ones((4, 57), dtype=bool), batch_size=2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_forward_1d_single_batch(self, model, segmenter, masker, inputs_original):
        """forward_1d with 3 coalitions where batch_size > n_coalitions."""
        imputer = VisionImputer(
            model=model,
            processor=MagicMock(),
            segmenter=segmenter,
            masker=masker,
            inputs_original=inputs_original,
        )
        result = imputer.forward_1d(np.ones((3, 57), dtype=bool), batch_size=8)
        assert result.shape == (3,)

    def test_forward_crossmodal_shape(self, model, segmenter, masker, inputs_original):
        """forward_crossmodal with 4 img x 3 txt coalitions (equal batches)."""
        proc = MagicMock()
        proc.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.int),
        }
        imputer = VisionImputer(
            model=model,
            processor=proc,
            segmenter=segmenter,
            masker=masker,
            inputs_original=inputs_original,
        )
        result = imputer.forward_crossmodal(
            np.ones((4, 49), dtype=bool),
            np.ones((3, 8), dtype=bool),
            batch_size=2,
        )
        assert result.shape == (4, 3)

    def test_forward_crossmodal_equal_batches(self, model, segmenter, masker, inputs_original):
        """forward_crossmodal with 3 img x 3 txt coalitions, equal batches."""
        proc = MagicMock()
        proc.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.int),
        }
        imputer = VisionImputer(
            model=model,
            processor=proc,
            segmenter=segmenter,
            masker=masker,
            inputs_original=inputs_original,
        )
        result = imputer.forward_crossmodal(
            np.ones((3, 49), dtype=bool),
            np.ones((3, 8), dtype=bool),
            batch_size=3,
        )
        assert result.shape == (3, 3)

    def test_n_players_properties(self, model, segmenter, masker, inputs_original):
        imputer = VisionImputer(
            model=model,
            processor=MagicMock(),
            segmenter=segmenter,
            masker=masker,
            inputs_original=inputs_original,
        )
        assert imputer.n_players_image == 49
        assert imputer.n_players_text == 8
        assert imputer.n_players == 57

    def test_model_type_property(self, model, segmenter, masker, inputs_original):
        imputer = VisionImputer(
            model=model,
            processor=MagicMock(),
            segmenter=segmenter,
            masker=masker,
            inputs_original=inputs_original,
        )
        assert imputer.model_type == "clip"


# ═══════════════════════════════════════════════════════════════════════
# VisionLanguageGame tests (mock imputer)
# ═══════════════════════════════════════════════════════════════════════


class TestVisionLanguageGame:
    """Test the shapiq.Game adapter with a mocked imputer."""

    @pytest.fixture
    def mock_imputer(self):
        """Returns a MagicMock that mimics VisionImputer interface."""
        from unittest.mock import MagicMock

        imputer = MagicMock()
        imputer.n_players_image = 49
        imputer.n_players_text = 8
        imputer.n_players = 57
        imputer.inputs_raw = {"pixel_values": torch.randn(1, 3, 224, 224)}
        imputer.processor = object()
        # forward_1d returns [empty, full]
        imputer.forward_1d.return_value = np.array([-0.5, 25.0])
        return imputer

    def test_game_initialisation(self, mock_imputer):
        game = VisionLanguageGame(mock_imputer, batch_size=64)
        assert game.empty_value == -0.5
        assert game.full_value == 25.0
        assert game.n_players == 57
        assert game.n_players_image == 49
        assert game.n_players_text == 8

    def test_game_normalization_value(self, mock_imputer):
        game = VisionLanguageGame(mock_imputer, batch_size=64)
        # normalization_value should equal empty_value (game is centered)
        assert game.normalization_value == -0.5
        assert game.normalize

    def test_game_delegates_to_imputer(self, mock_imputer):
        game = VisionLanguageGame(mock_imputer, batch_size=64)
        coalitions = np.ones((3, 57), dtype=bool)
        game.value_function(coalitions)
        mock_imputer.forward_1d.assert_called()

    def test_game_inputs_property(self, mock_imputer):
        game = VisionLanguageGame(mock_imputer, batch_size=64)
        assert game.inputs is mock_imputer.inputs_raw

    def test_game_processor_property(self, mock_imputer):
        game = VisionLanguageGame(mock_imputer, batch_size=64)
        assert game.processor is mock_imputer.processor


# ═══════════════════════════════════════════════════════════════════════
# Factory build end-to-end (mock model + mock processor)
# ═══════════════════════════════════════════════════════════════════════


class TestVisionImputerFactoryBuild:
    """Test the full factory.build() assembly with mocked HF model + processor."""

    @pytest.fixture
    def mock_model(self):
        import types
        from unittest.mock import MagicMock

        class MockModel:
            name_or_path = "openai/clip-vit-base-patch32"

            def parameters(self):
                return iter([torch.nn.Parameter(torch.randn(1))])

            def __call__(self, **kwargs):
                bs = kwargs["pixel_values"].shape[0]
                out = types.SimpleNamespace()
                out.logits_per_image = torch.randn(bs, bs)
                return out

        # Mock vision_model required by _extract_vision_dims
        vc = MagicMock()
        vc.image_size = 224
        vc.num_channels = 3
        emb = MagicMock()
        emb.patch_size = 32
        emb.image_size = 224
        emb.config = vc
        vm = MagicMock()
        vm.embeddings = emb
        vm.config = vc
        MockModel.vision_model = vm
        MockModel.config = MagicMock()
        MockModel.config.vision_config = vc
        return MockModel()

    @pytest.fixture
    def mock_processor(self):
        from unittest.mock import MagicMock

        proc = MagicMock()
        proc.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.int),
        }
        return proc

    def test_build_default(self, mock_model, mock_processor):
        """Default config: PatchSegmenter + CrossModalMeanMasker."""
        factory = VisionImputerFactory()
        imputer = factory.build(
            mock_model,
            mock_processor,
            torch.randn(3, 224, 224),
            "test text",
        )
        assert imputer.n_players_image == 49
        assert imputer.n_players_text == 6  # CLIP: input_length(8) - 2
        assert imputer.n_players == 55
        assert imputer.model_type == "clip"

    def test_build_with_amp(self, mock_model, mock_processor):
        """AMP flag is passed through."""
        factory = VisionImputerFactory()
        imputer = factory.build(
            mock_model,
            mock_processor,
            torch.randn(3, 224, 224),
            "test text",
            use_amp=True,
        )
        assert imputer.use_amp is True

    def test_build_forward_1d(self, mock_model, mock_processor):
        """After build, forward_1d returns correct shape."""
        factory = VisionImputerFactory()
        imputer = factory.build(
            mock_model,
            mock_processor,
            torch.randn(3, 224, 224),
            "test text",
        )
        result = imputer.forward_1d(np.ones((2, imputer.n_players), dtype=bool), batch_size=2)
        assert result.shape == (2,)

    def test_build_forward_1d_empty_coalition(self, mock_model, mock_processor):
        """forward_1d with all-False (empty) coalition returns correct shape."""
        factory = VisionImputerFactory()
        imputer = factory.build(
            mock_model,
            mock_processor,
            torch.randn(3, 224, 224),
            "test text",
        )
        result = imputer.forward_1d(
            np.zeros((1, imputer.n_players), dtype=bool),
            batch_size=1,
        )
        assert result.shape == (1,)

    def test_build_with_amp_skip_on_cpu(self, mock_model, mock_processor):
        """AMP flag accepted; if no CUDA the autocast context is a no-op."""
        factory = VisionImputerFactory()
        imputer = factory.build(
            mock_model,
            mock_processor,
            torch.randn(3, 224, 224),
            "test text",
            use_amp=True,
        )
        assert imputer.use_amp is True
        # Call with a tiny batch to ensure the AMP codepath doesn't crash
        result = imputer.forward_1d(
            np.ones((1, imputer.n_players), dtype=bool),
            batch_size=1,
        )
        assert result.shape == (1,)


# ═══════════════════════════════════════════════════════════════════════
# SigLIP / SigLIP2 text masking tests
# ═══════════════════════════════════════════════════════════════════════


class TestTextMaskFormat:
    """Verify Segmenter._build_text_attention_mask for different model types."""

    def _make_segmenter(self, model_type: str, n_txt: int = 8, total_len: int = 64):
        return PatchSegmenter(
            SegmenterConfig(
                strategy="patch",
                model_type=model_type,
                image_size=224,
                patch_size=32,
                n_channels=3,
                grid_size=7,
                n_players_image=49,
                n_players_text=n_txt,
                text_total_length=total_len,
            )
        )

    def test_clip_bos_eos(self):
        """CLIP: pads with BOS=1 and EOS=1."""
        seg = self._make_segmenter("clip", n_txt=8, total_len=10)
        coalitions = np.zeros((1, 8), dtype=bool)
        mask = seg.generate_masks(coalitions_text=coalitions)
        attn = mask.text_attention_mask
        assert attn.shape == (1, 10)
        assert attn[0, 0].item() == 1  # BOS
        assert attn[0, -1].item() == 1  # EOS
        assert attn[0, 1:-1].sum() == 0.0  # all tokens False

    def test_siglip_right_pad(self):
        """SigLIP: right-pads with 1s after valid tokens."""
        seg = self._make_segmenter("siglip", n_txt=8, total_len=64)
        coalitions = np.ones((1, 8), dtype=bool)
        mask = seg.generate_masks(coalitions_text=coalitions)
        attn = mask.text_attention_mask
        assert attn.shape == (1, 64)
        # First 8 positions follow coalition (all 1), remaining 56 are pad (also 1)
        assert attn[0, :8].sum() == 8.0
        assert attn[0, 8:].sum() == 56.0  # all padded

    def test_siglip2_right_pad(self):
        """SigLIP2: same right-pad behaviour as SigLIP."""
        seg = self._make_segmenter("siglip2", n_txt=8, total_len=64)
        coalitions = np.zeros((1, 8), dtype=bool)
        mask = seg.generate_masks(coalitions_text=coalitions)
        attn = mask.text_attention_mask
        assert attn.shape == (1, 64)
        # All valid tokens occluded, padded positions are 1
        assert attn[0, :8].sum() == 0.0
        assert attn[0, 8:].sum() == 56.0


# ═══════════════════════════════════════════════════════════════════════
# Factory build with real image (SLIC segmenter)
# ═══════════════════════════════════════════════════════════════════════


class TestFactoryBuildWithSLIC:
    """factory.build() with image file, strategy=slic."""

    IMAGE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "data",
        "dog_and_hydrant.png",
    )
    INPUT_TEXT = "black dog next to a yellow hydrant"

    @pytest.fixture
    def model(self):
        import types
        from unittest.mock import MagicMock

        class MockModel:
            name_or_path = "openai/clip-vit-base-patch32"

            def parameters(self):
                return iter([torch.nn.Parameter(torch.randn(1))])

            def __call__(self, **kwargs):
                img_bs = kwargs["pixel_values"].shape[0]
                txt_bs = kwargs["input_ids"].shape[0]
                out = types.SimpleNamespace()
                out.logits_per_image = torch.randn(img_bs, txt_bs)
                return out

        vc = MagicMock()
        vc.image_size = 224
        vc.num_channels = 3
        emb = MagicMock()
        emb.patch_size = 32
        emb.image_size = 224
        emb.config = vc
        vm = MagicMock()
        vm.embeddings = emb
        vm.config = vc
        MockModel.vision_model = vm
        MockModel.config = MagicMock()
        MockModel.config.vision_config = vc
        return MockModel()

    @pytest.fixture
    def mock_processor(self):
        from unittest.mock import MagicMock

        proc = MagicMock()
        proc.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.int),
        }
        return proc

    def test_build_slic_default(self, model, mock_processor):
        """Build with strategy='slic' and a real image → succeeds."""
        from PIL import Image

        from shapiq.imputer.vision import SlicParams

        seg_cfg = SegmenterConfig(
            strategy="slic",
            slic=SlicParams(n_segments=15, compactness=10.0, sigma=1.0),
        )
        factory = VisionImputerFactory()
        imputer = factory.build(
            model,
            mock_processor,
            Image.open(self.IMAGE_PATH).convert("RGB"),
            self.INPUT_TEXT,
            segmenter_config=seg_cfg,
        )
        assert imputer.n_players_image > 0
        assert imputer.n_players_text > 0
        assert imputer.model_type == "clip"
        # forward_1d with a few coalitions
        result = imputer.forward_1d(
            np.ones((2, imputer.n_players), dtype=bool),
            batch_size=2,
        )
        assert result.shape == (2,)

    def test_build_slic_budget_change(self, model, mock_processor):
        """Different n_segments produces different player counts."""
        from PIL import Image

        from shapiq.imputer.vision import SlicParams

        seg_cfg_10 = SegmenterConfig(
            strategy="slic",
            slic=SlicParams(n_segments=10, compactness=10.0, sigma=1.0),
        )
        seg_cfg_50 = SegmenterConfig(
            strategy="slic",
            slic=SlicParams(n_segments=50, compactness=10.0, sigma=1.0),
        )
        factory = VisionImputerFactory()
        im1 = factory.build(
            model,
            mock_processor,
            Image.open(self.IMAGE_PATH).convert("RGB"),
            self.INPUT_TEXT,
            segmenter_config=seg_cfg_10,
        )
        im2 = factory.build(
            model,
            mock_processor,
            Image.open(self.IMAGE_PATH).convert("RGB"),
            self.INPUT_TEXT,
            segmenter_config=seg_cfg_50,
        )
        assert im2.n_players_image > im1.n_players_image
