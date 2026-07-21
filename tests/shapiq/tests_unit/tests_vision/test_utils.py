"""Tests for ``shapiq.vision.utils``."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from shapiq.vision.utils import (
    _try_convert_torch_tensor,
    as_hwc_array,
    extract_logits,
    get_torch_device,
    tensor_to_numpy,
    to_tensor_chw,
)


class TestAsHwcArray:
    def test_numpy_hwc_passthrough(self) -> None:
        image = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        out = as_hwc_array(image)
        np.testing.assert_array_equal(out, image)

    def test_numpy_grayscale_adds_channel(self) -> None:
        image = np.arange(4, dtype=np.float64).reshape(2, 2)
        out = as_hwc_array(image)
        assert out.shape == (2, 2, 1)
        np.testing.assert_array_equal(out[..., 0], image)

    def test_pil_rgb_image(self) -> None:
        arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        pil = Image.fromarray(arr, mode="RGB")
        out = as_hwc_array(pil)
        np.testing.assert_array_equal(out, arr)

    def test_pil_grayscale_converted_to_rgb(self) -> None:
        gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
        pil = Image.fromarray(gray, mode="L")
        out = as_hwc_array(pil)
        assert out.shape == (2, 2, 3)
        np.testing.assert_array_equal(out[..., 0], gray)
        np.testing.assert_array_equal(out[..., 1], gray)
        np.testing.assert_array_equal(out[..., 2], gray)

    def test_torch_chw_tensor(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        out = as_hwc_array(tensor)
        expected = tensor.permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(out, expected)

    def test_torch_hwc_tensor(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        out = as_hwc_array(tensor)
        np.testing.assert_array_equal(out, tensor.numpy())

    def test_torch_hwc_tensor_4x4_layout_not_misread_as_chw(self) -> None:
        """A (4, 4, 3) tensor is HWC; H=4 must not be treated as C in a CHW layout."""
        tensor = torch.arange(48, dtype=torch.float32).reshape(4, 4, 3)
        out = as_hwc_array(tensor)
        np.testing.assert_array_equal(out, tensor.numpy())

    def test_batched_input_rejected(self) -> None:
        """4-D (batched) inputs are not supported and must raise ValueError."""
        with pytest.raises(ValueError, match="one image at a time"):
            as_hwc_array(np.zeros((1, 3, 2, 2)))

    def test_batched_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="one image at a time"):
            as_hwc_array(torch.zeros(1, 3, 2, 2))

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="image must be"):
            as_hwc_array({"not": "an image"})

    def test_scalar_rejected(self) -> None:
        with pytest.raises(TypeError, match="image must be"):
            as_hwc_array(5)

    def test_ragged_nested_sequence_rejected(self) -> None:
        """Anything numpy can only store as object dtype is not an image."""
        with pytest.raises(TypeError, match="image must be"):
            as_hwc_array([object(), object()])

    def test_nested_list_accepted_as_array_like(self) -> None:
        """The numpy fallback keeps plain nested lists working."""
        out = as_hwc_array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        assert out.shape == (1, 2, 3)

    def test_ambiguous_chw_tensor_prefers_torch_layout(self) -> None:
        """(3, 4, 4) is read as CHW: both axes look channel-like, torch layout wins."""
        tensor = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
        out = as_hwc_array(tensor)
        assert out.shape == (4, 4, 3)
        np.testing.assert_array_equal(out, tensor.permute(1, 2, 0).numpy())

    def test_single_channel_chw_tensor(self) -> None:
        tensor = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        out = as_hwc_array(tensor)
        assert out.shape == (2, 4, 1)

    def test_torch_2d_tensor_gets_channel_axis(self) -> None:
        out = as_hwc_array(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        assert out.shape == (2, 2, 1)

    def test_tensor_detached_from_graph(self) -> None:
        """A tensor carrying grad history must convert without a numpy conversion error."""
        tensor = torch.zeros(3, 4, 5, requires_grad=True)
        assert as_hwc_array(tensor).shape == (4, 5, 3)

    def test_ragged_list_rejected(self) -> None:
        """Ragged nesting cannot form an array and is caught before it reaches the model."""
        with pytest.raises(TypeError, match="image must be"):
            as_hwc_array([[1, 2], [3]])

    def test_one_dimensional_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            as_hwc_array([1, 2, 3])

    def test_batched_nested_list_rejected(self) -> None:
        """Sequences carry no ``ndim``, so the batch check only fires after conversion."""
        with pytest.raises(ValueError, match="one image at a time"):
            as_hwc_array([[[[1.0]]]])

    def test_all_channel_like_axes_default_to_chw(self) -> None:
        """(4, 4, 4) is fully ambiguous, so the torch CHW reading acts as the documented tie-break."""
        tensor = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
        np.testing.assert_array_equal(as_hwc_array(tensor), tensor.permute(1, 2, 0).numpy())

    def test_pil_unavailable_falls_through_to_other_converters(self) -> None:
        """Without PIL installed, arrays and tensors must still convert."""
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            assert as_hwc_array(np.zeros((2, 2, 3))).shape == (2, 2, 3)
            assert as_hwc_array(torch.zeros(3, 2, 2)).shape == (2, 2, 3)


class TestTryConvertTorchTensor:
    """Direct tests for the tensor converter's guards, which ``as_hwc_array`` shadows."""

    def test_returns_none_for_non_tensors(self) -> None:
        assert _try_convert_torch_tensor(np.zeros((2, 2, 3))) is None
        assert _try_convert_torch_tensor("not a tensor") is None

    def test_rejects_batched_tensor(self) -> None:
        with pytest.raises(ValueError, match="one image at a time"):
            _try_convert_torch_tensor(torch.zeros(1, 3, 2, 2))

    def test_rejects_zero_dimensional_tensor(self) -> None:
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            _try_convert_torch_tensor(torch.tensor(1.0))


class TestToTensorChw:
    def test_returns_chw_float32(self) -> None:
        image = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        tensor = to_tensor_chw(image)
        assert tensor.shape == (3, 2, 2)
        assert tensor.dtype == torch.float32

    def test_uint8_is_rescaled_to_unit_interval(self) -> None:
        image = np.full((2, 2, 3), 255, dtype=np.uint8)
        tensor = to_tensor_chw(image)
        assert torch.allclose(tensor, torch.ones(3, 2, 2))

    def test_float_input_is_not_rescaled(self) -> None:
        image = np.full((2, 2, 3), 5.0, dtype=np.float64)
        tensor = to_tensor_chw(image)
        assert torch.allclose(tensor, torch.full((3, 2, 2), 5.0))

    def test_torch_hwc_tensor_roundtrip(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        out = to_tensor_chw(tensor)
        expected = tensor.permute(2, 0, 1)
        torch.testing.assert_close(out, expected)


class TestExtractLogits:
    """Reads logits from both torchvision-style tensors and transformers-style outputs."""

    def test_bare_tensor_returned_as_is(self) -> None:
        logits = torch.zeros(2, 5)
        assert extract_logits(logits) is logits

    def test_output_object_with_logits(self) -> None:
        logits = torch.zeros(2, 5)
        assert extract_logits(SimpleNamespace(logits=logits)) is logits

    def test_logits_attribute_wins_over_tensor_nature(self) -> None:
        """A tensor subclass carrying ``.logits`` resolves through the attribute."""
        inner = torch.zeros(2, 5)
        output = SimpleNamespace(logits=inner, last_hidden_state=torch.zeros(2, 7))
        assert extract_logits(output) is inner

    def test_output_without_logits_rejected(self) -> None:
        """Encoder-only models (ViT-MAE, bare CLIP) expose no classification head."""
        output = SimpleNamespace(last_hidden_state=torch.zeros(2, 7))
        with pytest.raises(TypeError, match="no classification logits"):
            extract_logits(output)

    def test_error_names_the_output_type(self) -> None:
        with pytest.raises(TypeError, match="SimpleNamespace"):
            extract_logits(SimpleNamespace(pooler_output=torch.zeros(2, 7)))

    def test_non_tensor_logits_rejected(self) -> None:
        with pytest.raises(TypeError, match="no classification logits"):
            extract_logits(SimpleNamespace(logits=[[1.0, 2.0]]))

    def test_none_output_rejected(self) -> None:
        with pytest.raises(TypeError, match="no classification logits"):
            extract_logits(None)

    @pytest.mark.parametrize(
        ("shape", "label"),
        [
            ((5,), "1-D"),
            ((2, 21, 64, 64), "segmentation-style 4-D"),
            ((2, 100, 91), "detection-style 3-D"),
        ],
    )
    def test_non_2d_logits_rejected(self, shape, label) -> None:
        """Dense-prediction heads return per-pixel or per-query logits, which have no class axis."""
        with pytest.raises(TypeError, match="2-D classification logits"):
            extract_logits(torch.zeros(*shape))

    def test_2d_logits_accepted(self) -> None:
        logits = torch.zeros(4, 1000)
        assert extract_logits(logits).shape == (4, 1000)


class TestTorchDeviceHelpers:
    def test_get_torch_device_from_tensor(self) -> None:
        tensor = torch.zeros(2, 3)
        assert get_torch_device(tensor) == torch.device("cpu")

    def test_get_torch_device_from_module(self) -> None:
        module = torch.nn.Linear(2, 1)
        assert get_torch_device(module) == torch.device("cpu")

    def test_get_torch_device_falls_back_to_cpu(self) -> None:
        assert get_torch_device(object()) == torch.device("cpu")

    def test_tensor_to_numpy_from_cpu(self) -> None:
        tensor = torch.tensor([1.0, 2.0])
        np.testing.assert_array_equal(tensor_to_numpy(tensor), np.array([1.0, 2.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_torch_device_from_cuda_module(self) -> None:
        module = torch.nn.Linear(2, 1).cuda()
        assert get_torch_device(module).type == "cuda"
