"""Tests for ``shapiq.vision.utils``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from shapiq.vision.utils import (
    as_hwc_array,
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

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="image must be"):
            as_hwc_array({"not": "an image"})


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
