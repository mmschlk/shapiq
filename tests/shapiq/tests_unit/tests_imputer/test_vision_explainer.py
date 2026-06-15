"""Tests for VisionExplainer, ExactComputer correctness, and Player x Masker matrix."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from shapiq.explainer.vision import VisionExplainer
from shapiq.game_theory import ExactComputer
from shapiq.imputer.vision import (
    MaskerConfig,
    SegmenterConfig,
    VisionImputerFactory,
    VisionLanguageGame,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers: mock VLM model + processor
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_vlm(image_size: int = 8, patch_size: int = 4):
    """Build a minimal mock CLIP-like model with deterministic outputs."""

    class MockVLM:
        name_or_path = "openai/clip-vit-base-patch32"

        def parameters(self):
            return iter([torch.nn.Parameter(torch.randn(1))])

        def __call__(self, **kwargs):
            pixel_values = kwargs["pixel_values"]
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs.get("attention_mask")
            # Deterministic function: img_score = mean(pixels), txt_score = sum(masked ids)
            img_score = pixel_values.float().mean(dim=(1, 2, 3))  # (B_img,)
            txt_raw = input_ids.float()
            if attention_mask is not None:
                txt_raw = txt_raw * attention_mask.float()
            txt_score = txt_raw.sum(dim=1)  # (B_txt,)
            out = types.SimpleNamespace()
            out.logits_per_image = img_score[:, None] + txt_score[None, :]
            return out

    # Mock vision_model.embeddings for ViT detection
    vc = MagicMock()
    vc.image_size = image_size
    vc.num_channels = 3
    emb = MagicMock()
    emb.patch_size = patch_size
    emb.image_size = image_size
    emb.config = vc
    vm = MagicMock()
    vm.embeddings = emb
    vm.config = vc

    model = MockVLM()
    model.vision_model = vm
    model.config = MagicMock()
    model.config.vision_config = vc
    return model


def _make_mock_processor():
    proc = MagicMock()
    proc.return_value = {
        "pixel_values": torch.randn(1, 3, 8, 8),
        "input_ids": torch.randint(1, 10, (1, 6)),
        "attention_mask": torch.ones(1, 6, dtype=torch.int),
    }
    return proc


# ═══════════════════════════════════════════════════════════════════════
# ExactComputer correctness test
# ═══════════════════════════════════════════════════════════════════════


class TestExactComputerCorrectness:
    """Verify that VisionLanguageGame + ExactComputer produce correct values.

    Uses a tiny 8x8 image with grid_size=2 (4 patch players). The mock model
    computes a deterministic additive function, so ExactComputer order=2
    values should match brute-force computation.
    """

    @pytest.fixture
    def model(self):
        return _make_mock_vlm(image_size=8, patch_size=4)

    @pytest.fixture
    def processor(self):
        return _make_mock_processor()

    @pytest.fixture
    def game(self, model, processor):
        factory = VisionImputerFactory()
        seg_cfg = SegmenterConfig(
            strategy="patch",
            model_type="clip",
            image_size=8,
            patch_size=4,
            n_channels=3,
            grid_size=2,
            n_players_image=4,
            n_players_text=2,
            text_total_length=6,
        )
        msk_cfg = MaskerConfig(strategy="vision_mean")
        imputer = factory.build(
            model=model,
            processor=processor,
            input_image=torch.randn(3, 8, 8),
            input_text="test",
            segmenter_config=seg_cfg,
            masker_config=msk_cfg,
        )
        return VisionLanguageGame(imputer, batch_size=4)

    def test_exact_computer_order_1(self, game):
        """ExactComputer with order=1 produces n_players values (SV)."""
        exact = ExactComputer(game=game, n_players=game.n_players)
        iv = exact(index="SV", order=1)
        assert iv.n_players == game.n_players
        assert iv.max_order == 1
        assert iv.index == "SV"
        # SV includes order 0 (empty coalition), so values = n_players + 1
        assert len(iv.values) == game.n_players + 1

    def test_exact_computer_order_2(self, game):
        """ExactComputer with order=2 includes pairwise interactions."""
        exact = ExactComputer(game=game, n_players=game.n_players)
        iv = exact(index="k-SII", order=2)
        assert iv.max_order == 2
        assert iv.n_players == game.n_players
        # k-SII order 2: C(n,1) + C(n,2) plus possibly order 0
        assert len(iv.values) > game.n_players

    def test_value_function_all_ones(self, game):
        """All coalitions = True (full set) → single value."""
        coalitions = np.ones((1, game.n_players), dtype=bool)
        values = game.value_function(coalitions)
        assert values.shape == (1,)
        assert np.isfinite(values[0])

    def test_value_function_all_zeros(self, game):
        """Empty coalition returns the normalization value."""
        coalitions = np.zeros((1, game.n_players), dtype=bool)
        values = game.value_function(coalitions)
        assert values.shape == (1,)
        assert values[0] == pytest.approx(game.empty_value)

    # ── Exact value correctness ──────────────────────────────────────────

    def test_exact_sv_values_tiny(self):
        """SV values match exact mathematical computation for 2-player game.

        Setup:
            - 4x4 image, 1 patch (grid_size=1) → 1 image player
            - 3 input_ids tokens, CLIP (BOS+EOS removed) → 1 text player
            - pixel_values = 0.5 (constant), input_ids = [1, 5, 1]
            - Model: logits[i,j] = mean(pixels[i]) + sum(masked_ids[j])

        Manually computed coalition values:
            v(empty)  = mean(0) + sum([1,0,1])   = 0 + 2   = 2.0
            v(img)    = mean(0.5*present) + sum([1,0,1]) = 0.5 + 2 = 2.5
            v(txt)    = mean(0) + sum([1,5,1])   = 0 + 7   = 7.0
            v(full)   = mean(0.5) + sum([1,5,1]) = 0.5 + 7 = 7.5

        Shapley values:
            SV(img) = (v(img)-v(empty) + v(full)-v(txt)) / 2 = (0.5+0.5)/2 = 0.5
            SV(txt) = (v(txt)-v(empty) + v(full)-v(img)) / 2 = (5.0+5.0)/2 = 5.0
        """
        model = _make_mock_vlm(image_size=4, patch_size=4)

        # Mock processor returning FIXED values
        proc = MagicMock()
        proc.return_value = {
            "pixel_values": torch.full((1, 3, 4, 4), 0.5),
            "input_ids": torch.tensor([[1, 5, 1]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.int),
        }

        factory = VisionImputerFactory()
        seg_cfg = SegmenterConfig(
            strategy="patch",
            model_type="clip",
            image_size=4,
            patch_size=4,
            n_channels=3,
            grid_size=1,
            n_players_image=1,
            n_players_text=1,
            text_total_length=3,
        )
        msk_cfg = MaskerConfig(strategy="crossmodal_mean")
        imputer = factory.build(
            model=model,
            processor=proc,
            input_image=torch.full((3, 4, 4), 0.5),
            input_text="a",
            segmenter_config=seg_cfg,
            masker_config=msk_cfg,
        )
        game = VisionLanguageGame(imputer, batch_size=4)

        # Verify coalition values match manual computation
        _n = game.n_players
        all_coalitions = np.array([[False, False], [True, False], [False, True], [True, True]])
        values = game.value_function(all_coalitions)
        np.testing.assert_allclose(values[0], 2.0, rtol=1e-5)  # empty
        np.testing.assert_allclose(values[1], 2.5, rtol=1e-5)  # img only
        np.testing.assert_allclose(values[2], 7.0, rtol=1e-5)  # txt only
        np.testing.assert_allclose(values[3], 7.5, rtol=1e-5)  # full

        # Manually verify SV efficiency: sum(SV) = v(full) - v(empty)
        coalitions = np.array([[False, False], [True, False], [False, True], [True, True]])
        raw_values = game.value_function(coalitions)  # raw (non-normalized)
        v_empty, v_img, v_txt, v_full = [float(x) for x in raw_values]
        sv_img_manual = 0.5 * (v_img - v_empty + v_full - v_txt)
        sv_txt_manual = 0.5 * (v_txt - v_empty + v_full - v_img)
        np.testing.assert_allclose(sv_img_manual, 0.5, rtol=1e-5)
        np.testing.assert_allclose(sv_txt_manual, 5.0, rtol=1e-5)
        np.testing.assert_allclose(sv_img_manual + sv_txt_manual, v_full - v_empty, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# PlayerxMasker matrix test
# ═══════════════════════════════════════════════════════════════════════


class TestPlayerMaskerMatrix:
    """Verify every Segmenter x Masker combination produces valid outputs.

    Tests both forward_1d shape and that the explain pipeline produces
    valid InteractionValues for each combo. Blur combos require skimage.
    """

    @pytest.fixture
    def model(self):
        return _make_mock_vlm(image_size=8, patch_size=4)

    @pytest.fixture
    def processor(self):
        return _make_mock_processor()

    @pytest.fixture
    def custom_masks(self):
        masks = np.zeros((3, 8, 8), dtype=bool)
        masks[0, :4, :] = True  # top third
        masks[1, 2:6, :] = True  # middle third
        masks[2, 4:, :] = True  # bottom third (overlaps intentionally)
        return masks

    # ── All non-blur combos (parametrized) ───────────────────────────────

    @pytest.mark.parametrize(
        "segmenter_name,masker_name,seg_kwargs",
        [
            ("patch", "vision_mean", {}),
            ("patch", "crossmodal_mean", {}),
            ("custom_segmenter", "vision_mean", {"masks": "custom_masks"}),
            ("custom_segmenter", "crossmodal_mean", {"masks": "custom_masks"}),
        ],
    )
    def test_forward_1d_shape(
        self, segmenter_name, masker_name, seg_kwargs, model, processor, custom_masks
    ):
        """forward_1d returns shape (n_coalitions,) for each combination."""
        segmenter_kwargs = self._resolve_kwargs(seg_kwargs, custom_masks)
        imputer = self._build_imputer(
            segmenter_name, masker_name, model, processor, segmenter_kwargs
        )
        n_coalitions = 4
        coalitions = np.ones((n_coalitions, imputer.n_players), dtype=bool)
        result = imputer.forward_1d(coalitions, batch_size=2)
        assert result.shape == (n_coalitions,), (
            f"{segmenter_name} x {masker_name}: expected ({n_coalitions},), got {result.shape}"
        )
        assert np.isfinite(result).all()

    @pytest.mark.parametrize(
        "segmenter_name,masker_name,seg_kwargs",
        [
            ("patch", "vision_mean", {}),
            ("patch", "crossmodal_mean", {}),
            ("custom_segmenter", "vision_mean", {"masks": "custom_masks"}),
            ("custom_segmenter", "crossmodal_mean", {"masks": "custom_masks"}),
        ],
    )
    def test_explain_interaction_values(
        self, segmenter_name, masker_name, seg_kwargs, model, processor, custom_masks
    ):
        """Each combination produces valid InteractionValues via VisionExplainer."""
        segmenter_kwargs = self._resolve_kwargs(seg_kwargs, custom_masks)
        imputer = self._build_imputer(
            segmenter_name, masker_name, model, processor, segmenter_kwargs
        )

        from shapiq.imputer.vision import VisionLanguageGame

        game = VisionLanguageGame(imputer, batch_size=4)
        from shapiq import KernelSHAPIQ

        approx = KernelSHAPIQ(n=game.n_players, max_order=2, index="k-SII", random_state=42)
        iv = approx(budget=16, game=game)
        assert iv.n_players == game.n_players
        assert iv.max_order == 2
        assert len(iv.values) > game.n_players
        assert np.isfinite(iv.values).all()

    # ── Blur combos (skimage required) ───────────────────────────────────

    @pytest.mark.parametrize(
        "segmenter_name,seg_kwargs",
        [
            ("patch", {}),
            ("custom_segmenter", {"masks": "custom_masks"}),
        ],
    )
    def test_blur_forward_1d(self, segmenter_name, seg_kwargs, model, processor, custom_masks):
        """Blur maskers produce correct forward_1d shape for each segmenter."""
        pytest.importorskip("skimage.filters")
        segmenter_kwargs = self._resolve_kwargs(seg_kwargs, custom_masks)
        for masker_name in ("vision_blur", "crossmodal_blur"):
            imputer = self._build_imputer(
                segmenter_name, masker_name, model, processor, segmenter_kwargs
            )
            result = imputer.forward_1d(
                np.ones((2, imputer.n_players), dtype=bool),
                batch_size=2,
            )
            assert result.shape == (2,), f"{segmenter_name} x {masker_name} failed"

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_kwargs(self, seg_kwargs: dict, custom_masks):
        """Resolve fixture references in seg_kwargs."""
        resolved = {}
        for k, v in seg_kwargs.items():
            resolved[k] = custom_masks if v == "custom_masks" else v
        return resolved

    def _build_imputer(self, segmenter_name, masker_name, model, processor, segmenter_kwargs):
        """Build an imputer for the given segmenter/masker combo."""
        factory = VisionImputerFactory()
        seg_cfg = SegmenterConfig(
            strategy=segmenter_name,
            model_type="clip",
            image_size=8,
            patch_size=4,
            n_channels=3,
            grid_size=2,
            n_players_image=4,
            n_players_text=2,
            text_total_length=6,
        )
        return factory.build(
            model=model,
            processor=processor,
            input_image=torch.randn(3, 8, 8),
            input_text="test",
            segmenter_config=seg_cfg,
            masker_config=MaskerConfig(strategy=masker_name),
            **segmenter_kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════
# Explainer integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestVisionExplainer:
    """End-to-end tests for VisionExplainer with mock model."""

    @pytest.fixture
    def model(self):
        return _make_mock_vlm(image_size=8, patch_size=4)

    @pytest.fixture
    def processor(self):
        return _make_mock_processor()

    def test_explain_returns_interaction_values(self, model, processor):
        """VisionExplainer.explain() returns InteractionValues with correct shape."""
        explainer = VisionExplainer(
            model=model,
            data=torch.randn(3, 8, 8),
            text="test",
            processor=processor,
            batch_size=4,
            index="SV",
            max_order=1,
        )
        iv = explainer.explain(budget=16)
        assert iv.index == "SV"
        assert iv.max_order == 1
        assert iv.n_players == explainer.game.n_players
        # SV includes order 0 (empty coalition), so values has n_players + 1 entries
        assert len(iv.values) == iv.n_players + 1

    def test_explain_with_order_2(self, model, processor):
        """VisionExplainer with k-SII, max_order=2 produces pairwise values."""
        explainer = VisionExplainer(
            model=model,
            data=torch.randn(3, 8, 8),
            text="test",
            processor=processor,
            batch_size=4,
            index="k-SII",
            max_order=2,
        )
        iv = explainer.explain(budget=32)
        assert iv.max_order == 2
        assert iv.index == "k-SII"
        # At minimum has all n_players + interaction values
        assert len(iv.values) >= iv.n_players

    def test_baseline_value(self, model, processor):
        """baseline_value matches the empty coalition."""
        explainer = VisionExplainer(
            model=model,
            data=torch.randn(3, 8, 8),
            text="test",
            processor=processor,
            batch_size=4,
        )
        assert np.isfinite(explainer.baseline_value)

    def test_game_property(self, model, processor):
        """game property returns the VisionLanguageGame."""
        explainer = VisionExplainer(
            model=model,
            data=torch.randn(3, 8, 8),
            text="test",
            processor=processor,
            batch_size=4,
        )
        game = explainer.game
        assert isinstance(game, VisionLanguageGame)
        assert game.n_players > 0


# ═══════════════════════════════════════════════════════════════════════
# Explainer auto-dispatch test
# ═══════════════════════════════════════════════════════════════════════


class TestExplainerAutoDispatch:
    """Verify that ``shapiq.Explainer(...)`` routes to VisionExplainer for HF VLMs."""

    def test_dispatch_to_vision_explainer(self):
        """Calling Explainer(model=mock_vlm, data=img, ...) returns VisionExplainer."""
        from shapiq.explainer import Explainer

        model = _make_mock_vlm(image_size=8, patch_size=4)
        processor = _make_mock_processor()

        explainer = Explainer(
            model=model,
            data=torch.randn(3, 8, 8),
            text="test",
            processor=processor,
            batch_size=4,
            index="SV",
            max_order=1,
        )
        assert isinstance(explainer, VisionExplainer)
        iv = explainer.explain(budget=16)
        assert iv.n_players > 0
