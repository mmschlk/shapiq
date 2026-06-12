"""Utility functions for image conversion and device handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image as PILImage

    ImageLike = np.ndarray | torch.Tensor | PILImage
else:
    ImageLike = np.ndarray


def as_hwc_array(image: ImageLike) -> np.ndarray:
    """Convert an image to a ``(H, W, C)`` numpy array.

    Accepts:

    - **numpy arrays**: ``(H, W)`` or ``(H, W, C)`` — used as-is after
      adding a channel axis for 2-D inputs.
    - **PIL images**: force-converted to RGB, then to a ``(H, W, 3)`` uint8
      array.
    - **PyTorch tensors**: ``(C, H, W)`` or ``(H, W, C)`` — see
      :func:`_try_convert_torch_tensor` for layout-detection rules.

    4-D inputs are **not** accepted. Pass a single image, not a batch.

    Args:
        image: Input image in a supported format.

    Returns:
        A numpy array with shape ``(H, W, C)``.

    Raises:
        TypeError: If the input type is not supported.
        ValueError: If the input has 4 or more dimensions, or if the
            resulting array cannot be reduced to 3 dimensions.
    """
    ndim = getattr(image, "ndim", None) or (
        len(getattr(image, "shape", ())) if hasattr(image, "shape") else None
    )
    if ndim is not None and ndim >= 4:
        msg = (
            f"Expected a single image with 2 or 3 dimensions, got {ndim}-D input. "
            "Pass one image at a time — batched arrays are not supported."
        )
        raise ValueError(msg)

    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
    else:
        pil_image = _try_convert_pil_image(image)
        if pil_image is not None:
            arr = pil_image
        else:
            tensor = _try_convert_torch_tensor(image)
            if tensor is not None:
                arr = tensor
            else:
                try:
                    arr = np.asarray(image)
                except (TypeError, ValueError) as exc:
                    msg = (
                        "image must be a numpy array, PIL Image, or PyTorch tensor; "
                        f"got {type(image)!r}"
                    )
                    raise TypeError(msg) from exc

    if arr.ndim == 0 or arr.dtype == object:
        msg = f"image must be a numpy array, PIL Image, or PyTorch tensor; got {type(image)!r}"
        raise TypeError(msg)

    if arr.ndim >= 4:
        msg = (
            f"Expected a single image with 2 or 3 dimensions, got shape {arr.shape}. "
            "Pass one image at a time — batched arrays are not supported."
        )
        raise ValueError(msg)

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    if arr.ndim != 3:
        msg = f"Expected image with 2 or 3 dimensions after conversion, got shape {arr.shape}"
        raise ValueError(msg)

    return arr


def to_tensor_chw(
    image: ImageLike,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert an image to a float32 ``(C, H, W)`` PyTorch tensor.

    First normalises the input to a ``(H, W, C)`` numpy array via
    :func:`as_hwc_array`, then converts to a ``(C, H, W)`` float32 tensor.
    uint8 arrays are scaled to ``[0, 1]``; arrays of any other numeric dtype
    are cast to float32 without rescaling.

    Args:
        image: Input image as a numpy array, PIL image, or PyTorch tensor.
        device: Target PyTorch device. If ``None`` the tensor is placed on
            the CPU.

    Returns:
        A float32 :class:`torch.Tensor` with shape ``(C, H, W)`` on the
        requested device.

    Raises:
        TypeError: If the input cannot be converted by :func:`as_hwc_array`.
        ValueError: If the resulting array does not represent a valid image.
    """
    import torch

    arr = as_hwc_array(image)  # (H, W, C) numpy
    arr = arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float32)

    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def get_torch_device(obj: object) -> torch.device:
    """Return the :class:`torch.device` for a model, module, or tensor.

    Inspects ``.parameters()`` first, then ``.buffers()``, to find the device
    of the first available tensor. Falls back to CPU when no tensors are found.

    Args:
        obj: A :class:`torch.nn.Module`, :class:`torch.Tensor`, or any object
            that exposes ``.parameters()`` or ``.buffers()``.

    Returns:
        The :class:`torch.device` on which the object (or its first parameter)
        resides.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        msg = "PyTorch is required to resolve a torch device"
        raise ImportError(msg) from exc

    if isinstance(obj, torch.Tensor):
        return obj.device

    for accessor in (getattr(obj, "parameters", None), getattr(obj, "buffers", None)):
        if callable(accessor):
            try:
                return next(accessor()).device
            except StopIteration:
                continue

    return torch.device("cpu")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array, moving from GPU to CPU if needed.

    Args:
        tensor: Any :class:`torch.Tensor`, on any device.

    Returns:
        A numpy array with the same shape and dtype as ``tensor``.
    """
    return tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _try_convert_pil_image(image: object) -> np.ndarray | None:
    """Attempt to convert a PIL image to an RGB numpy array.

    Args:
        image: Candidate object to convert.

    Returns:
        A ``(H, W, 3)`` uint8 numpy array if ``image`` is a
        :class:`PIL.Image.Image`, or ``None`` if PIL is not installed or
        ``image`` is not a PIL image.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    return None


def _try_convert_torch_tensor(image: object) -> np.ndarray | None:
    """Attempt to convert a PyTorch tensor to a ``(H, W, C)`` numpy array.

    Handles tensors with shapes ``(H, W)``, ``(C, H, W)``, and ``(H, W, C)``.

    Layout detection uses channel-like sizes ``{1, 3, 4}``. Unambiguous
    cases are resolved from a single channel-like axis. When both the first
    and last axes look like channel counts (e.g. ``(3, 4, 4)`` vs
    ``(4, 4, 3)``), PyTorch ``(C, H, W)`` is preferred unless the shape is
    the common HWC pattern ``(H, W, 3)`` with ``H`` in ``{1, 3, 4}``.

    Args:
        image: Candidate object to convert.

    Returns:
        A numpy array with shape ``(H, W, C)`` if ``image`` is a
        :class:`torch.Tensor`, or ``None`` if PyTorch is not installed or
        ``image`` is not a tensor.

    Raises:
        ValueError: If the tensor has an unsupported number of dimensions.
    """
    try:
        import torch
    except ImportError:
        return None
    if not isinstance(image, torch.Tensor):
        return None

    tensor = image.detach().cpu()

    if tensor.ndim >= 4:
        msg = (
            f"Expected a single image tensor with 2 or 3 dimensions, got shape {tuple(tensor.shape)}. "
            "Pass one image at a time — batched tensors are not supported."
        )
        raise ValueError(msg)
    if tensor.ndim == 2:
        return tensor.numpy()[..., np.newaxis]
    if tensor.ndim == 3:
        channel_like = (1, 3, 4)
        height, width, channels = tensor.shape

        if channels in channel_like and height not in channel_like:
            pass  # (H, W, C)
        elif height in channel_like and channels not in channel_like:
            tensor = tensor.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
        elif height in channel_like and channels in channel_like:
            # Ambiguous small tensors, e.g. (3, 4, 4) CHW vs (4, 4, 3) HWC.
            if height in (1, 3) and width >= height and channels >= height:
                tensor = tensor.permute(1, 2, 0)
            elif height == 4 and channels == 3:
                pass  # square-ish RGB patch in HWC layout
            else:
                tensor = tensor.permute(1, 2, 0)
        return tensor.numpy()

    msg = f"Expected PyTorch tensor with 2 or 3 dimensions, got shape {tuple(tensor.shape)}"
    raise ValueError(msg)
