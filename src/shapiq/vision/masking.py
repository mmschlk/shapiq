"""Masking strategies for vision models.

Defines how to replace absent players in masked images before forwarding
through the model. Masking is applied in pixel space for CNNs and token
space for ViTs. Requires PyTorch to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.vision.validation import ModelCompatible, validate_config_attributes

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

from .custom_types import CoalitionDomain, VisionModel
from .utils import get_torch_device, to_tensor_chw

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from shapiq.typing import Model

    from .utils import ImageLike


def _model_dtype(model: Model) -> torch.dtype:
    """Return the dtype of a model's parameters, defaulting to float32.

    Mirrors :func:`~shapiq.vision.utils.get_torch_device`: objects without
    parameters (test doubles, wrappers) fall back to a sane default rather than
    raising.

    Args:
        model: Model to inspect.

    Returns:
        The dtype of the model's first parameter, or ``torch.float32``.
    """
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        return torch.float32


class MaskingStrategy(ModelCompatible, ABC):
    """Base class for masking strategies with compatibility validation.

    Subclasses declare the coalition domain they accept via
    ``accepted_coalition_domain``. This is used to ensure the masking strategy
    matches the player strategy that produced the coalitions. Compatibility with
    a model protocol is enforced via ``compatible_model_protocol``, default is
    ``VisionModel``. Some strategies require additional model attributes, which
    are validated in ``validate_model``.
    """

    accepted_coalition_domain: CoalitionDomain
    compatible_model_protocol = VisionModel


class PixelBasedMaskingStrategy(MaskingStrategy, ABC):
    """Base class for pixel-space masking strategies used with CNN models.

    Implementations receive the original image as a ``(C, H, W)`` tensor and
    a coalition matrix, and return a batch of masked images ready for a
    single forward pass through the model. ``accepted_coalition_domain`` is
    ``CoalitionDomain.PIXEL``.
    """

    accepted_coalition_domain: CoalitionDomain = CoalitionDomain.PIXEL
    compatible_model_protocol = VisionModel

    @abstractmethod
    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply masking to produce a batch of masked images.

        Args:
            image: Original image as a float32 ``(C, H, W)`` tensor.
            player_masks: Boolean tensor of shape ``(n_players, H, W)``
                mapping each player to its pixel region.
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present (unmasked).

        Returns:
            Float32 tensor of shape ``(n_coalitions, C, H, W)`` with absent
            players replaced by the imputation value.
        """
        ...

    def _build_pixel_mask(
        self,
        player_masks: torch.Tensor,
        coalitions: torch.Tensor,
    ) -> torch.Tensor:
        """Build a combined pixel absence mask for all coalitions.

        Args:
            player_masks: Boolean tensor of shape ``(n_players, H, W)``.
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Boolean tensor of shape ``(n_coalitions, H, W)`` where ``True``
            means the pixel belongs to an absent player and should be imputed.
        """
        n_players, H, W = player_masks.shape
        masks_flat = player_masks.view(n_players, -1).float()

        absent_players = (~coalitions).to(masks_flat.device).float()  # (n_coalitions, n_players)
        pixel_mask = (absent_players @ masks_flat).bool()  # (n_coalitions, H*W)
        return pixel_mask.view(-1, H, W)  # (n_coalitions, H, W)


class MeanColorMasking(PixelBasedMaskingStrategy):
    """Imputes absent player regions with the per-channel mean color of the original image.

    The mean is computed per channel across all spatial positions of the
    original image and broadcast into the masked regions.
    """

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean color masking to absent player regions."""
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)
        mean_color = image.mean(dim=(1, 2))

        return torch.where(
            pixel_mask.unsqueeze(1),
            mean_color[None, :, None, None],
            image.unsqueeze(0),
        )


class ZeroMasking(PixelBasedMaskingStrategy):
    """Imputes absent player regions with a constant scalar value.

    Args:
        value: The fill value used for masked pixels. Defaults to ``0.0``.
    """

    def __init__(self, value: float = 0.0) -> None:
        """Initialize the zero masking strategy with a specified fill value."""
        self.value = value

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply zero (or constant) masking to absent player regions."""
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)

        return torch.where(
            pixel_mask.unsqueeze(1),
            torch.tensor(self.value, dtype=image.dtype, device=image.device),
            image.unsqueeze(0),
        )


class BlurMasking(PixelBasedMaskingStrategy):
    """Imputes absent player regions with a Gaussian-blurred copy of the image.

    Absent regions are filled from a blurred version of the original image
    instead of a flat color. This keeps local color statistics and removes the
    hard edges that :class:`ZeroMasking` introduces at region boundaries, which
    a model may react to as if they were content.

    Args:
        sigma: Standard deviation of the Gaussian kernel in pixels. Larger
            values erase more detail. Defaults to ``10.0``.

    Example:
        >>> architecture = ClassificationArchitecture(
        ...     model=model, masking_strategy=BlurMasking(sigma=8.0)
        ... )
    """

    def __init__(self, sigma: float = 10.0) -> None:
        """Initialize the blur masking strategy.

        Args:
            sigma: Standard deviation of the Gaussian kernel in pixels.

        Raises:
            ValueError: If ``sigma`` is not positive.
        """
        if sigma <= 0:
            msg = f"sigma must be positive, got {sigma}."
            raise ValueError(msg)
        self.sigma = sigma

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply Gaussian-blur masking to absent player regions."""
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)
        blurred = self._blur(image)

        return torch.where(
            pixel_mask.unsqueeze(1),
            blurred.unsqueeze(0),
            image.unsqueeze(0),
        )

    def _blur(self, image: torch.Tensor) -> torch.Tensor:
        """Blur a ``(C, H, W)`` image with a separable Gaussian kernel.

        The kernel radius is capped at the image dimensions: a kernel wider than
        the image carries no additional information and cannot be reflect-padded.

        Args:
            image: Original image as a float ``(C, H, W)`` tensor.

        Returns:
            The blurred image as a float ``(C, H, W)`` tensor.
        """
        channels, height, width = image.shape
        if height < 2 or width < 2:  # nothing to blur across
            return image.clone()

        radius = max(1, min(int(3.0 * self.sigma), height - 1, width - 1))
        offsets = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
        kernel = torch.exp(-(offsets**2) / (2.0 * self.sigma**2))
        kernel = kernel / kernel.sum()
        size = kernel.numel()

        weight_x = kernel.view(1, 1, 1, size).repeat(channels, 1, 1, 1)
        weight_y = kernel.view(1, 1, size, 1).repeat(channels, 1, 1, 1)

        blurred = image.unsqueeze(0)
        blurred = F.pad(blurred, (radius, radius, 0, 0), mode="reflect")
        blurred = F.conv2d(blurred, weight_x, groups=channels)
        blurred = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
        blurred = F.conv2d(blurred, weight_y, groups=channels)
        return blurred.squeeze(0)


class DatasetMeanMasking(PixelBasedMaskingStrategy):
    """Imputes absent player regions with a fixed dataset-wide mean color.

    Unlike :class:`MeanColorMasking`, which averages the image being explained,
    this uses a pre-computed baseline such as the mean RGB vector of a training
    set. The fill is therefore identical for every image, which makes values
    comparable across a dataset.

    ``mean_color`` is required and must be on the same scale as the image the
    explainer sees, because that scale depends on the architecture: without a
    processor images are scaled to ``[0, 1]``, while with a Hugging Face
    ``processor`` they are kept in their raw ``0-255`` range for the processor to
    normalize. :attr:`IMAGENET_MEAN` and :attr:`IMAGENET_MEAN_255` provide the
    ImageNet baseline for each case.

    Args:
        mean_color: Baseline color as a scalar or a length-``C`` sequence.

    Example:
        >>> # torchvision model, images scaled to [0, 1]
        >>> masking = DatasetMeanMasking(DatasetMeanMasking.IMAGENET_MEAN)
        >>> # Hugging Face model with a processor, images kept at 0-255
        >>> masking = DatasetMeanMasking(DatasetMeanMasking.IMAGENET_MEAN_255)
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    """ImageNet channel means on the ``[0, 1]`` scale."""

    IMAGENET_MEAN_255 = (123.675, 116.28, 103.53)
    """ImageNet channel means on the ``0-255`` scale."""

    def __init__(self, mean_color: Sequence[float] | float) -> None:
        """Initialize the dataset-mean masking strategy.

        Args:
            mean_color: Baseline color, on the same scale as the explained image.
        """
        self.mean_color = torch.as_tensor(mean_color, dtype=torch.float32)

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply dataset-mean masking to absent player regions.

        Raises:
            ValueError: If ``mean_color`` is neither a scalar nor one value per
                image channel.
        """
        channels = image.shape[0]
        fill = self.mean_color.to(device=image.device, dtype=image.dtype)
        if fill.ndim == 0:
            fill = fill.expand(channels)
        if fill.numel() != channels:
            msg = (
                f"mean_color has {fill.numel()} values but the image has {channels} channels. "
                "Pass a scalar or one value per channel."
            )
            raise ValueError(msg)

        pixel_mask = self._build_pixel_mask(player_masks, coalitions)
        return torch.where(
            pixel_mask.unsqueeze(1),
            fill.reshape(1, channels, 1, 1),
            image.unsqueeze(0),
        )


class MarginalSampling(PixelBasedMaskingStrategy):
    """Imputes absent player regions with pixels from a bank of reference images.

    For every coalition one reference image is drawn pseudo-randomly and its
    pixels are copied into the absent region. Absent pixels come from the
    data distribution rather than from a fixed baseline, so masked images stay
    closer to the natural image manifold than with :class:`MeanColorMasking` or
    :class:`ZeroMasking`.

    The reference is selected from the *content* of each coalition, not from its
    position in the batch, so ``v(S)`` depends only on ``S``. This keeps the game
    a deterministic set function -- which Shapley values require -- no matter how
    the explainer batches coalitions. Vary ``random_state`` to resample the
    assignment of references to coalitions.

    Reference images are converted like the explained image: ``uint8`` inputs are
    scaled to ``[0, 1]``, floats are used as-is. When the architecture is given a
    Hugging Face ``processor`` the explained image is kept in its raw ``0-255``
    range instead, so pass float references in that same range.

    Args:
        reference_images: Images to sample from. Accepts anything
            :func:`~shapiq.vision.utils.to_tensor_chw` accepts (numpy ``(H, W, C)``
            arrays, PIL images, or ``(C, H, W)`` tensors), including a 4-D
            ``(N, H, W, C)`` numpy array. All must share the explained image's shape.
        random_state: Seed mixed into the per-coalition draw. Defaults to ``0``.

    Example:
        >>> references = [np.array(Image.open(path)) for path in reference_paths]
        >>> masking = MarginalSampling(reference_images=references, random_state=0)

    """

    def __init__(
        self,
        reference_images: Sequence[ImageLike],
        random_state: int = 0,
    ) -> None:
        """Initialize the marginal sampling strategy.

        Args:
            reference_images: Images to sample absent regions from.
            random_state: Seed mixed into the per-coalition reference draw.

        Raises:
            ValueError: If ``reference_images`` is empty or the images do not
                all share the same shape.
        """
        references = [to_tensor_chw(reference) for reference in reference_images]
        if not references:
            msg = "reference_images must contain at least one image."
            raise ValueError(msg)

        shapes = {tuple(reference.shape) for reference in references}
        if len(shapes) > 1:
            msg = f"All reference_images must share one shape, got {sorted(shapes)}."
            raise ValueError(msg)

        self._references = torch.stack(references)
        self.random_state = random_state

    def _draw_for(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Pick one reference index per coalition, based only on the coalition.

        Each coalition is hashed into a reference index. Hashing the membership
        bits (rather than drawing per batch row) is what makes ``v(S)`` a
        function of ``S`` alone: the same coalition always selects the same
        reference, whatever batch it is evaluated in.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Long tensor of shape ``(n_coalitions,)`` with values in
            ``[0, len(references))``.
        """
        n_players = coalitions.shape[1]
        # Weight each player by a distinct odd multiplier, so coalitions that
        # differ in any single player get different sums.
        weights = torch.arange(1, n_players + 1, device=coalitions.device, dtype=torch.int64)
        weights = weights * 2654435761 + self.random_state  # Knuth's multiplicative constant
        keys = (coalitions.to(torch.int64) * weights).sum(dim=1) + self.random_state

        keys = (keys ^ (keys >> 15)) * 2246822519
        keys = (keys ^ (keys >> 13)) * 3266489917
        keys = keys ^ (keys >> 16)
        return keys.abs() % len(self._references)

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply marginal-sampling masking to absent player regions.

        Raises:
            ValueError: If the reference images do not match the image's shape.
        """
        if self._references.shape[1:] != image.shape:
            msg = (
                f"reference_images have shape {tuple(self._references.shape[1:])} but the image "
                f"to explain has shape {tuple(image.shape)}. Resize the reference images to match."
            )
            raise ValueError(msg)

        pixel_mask = self._build_pixel_mask(player_masks, coalitions)
        references = self._references.to(device=image.device, dtype=image.dtype)
        drawn = self._draw_for(coalitions).to(references.device)

        return torch.where(
            pixel_mask.unsqueeze(1),
            references[drawn],
            image.unsqueeze(0),
        )


class InpaintingMasking(PixelBasedMaskingStrategy):
    """Imputes absent player regions with a user-supplied inpainter.

    The inpainter is called once per coalition with the original image and that
    coalition's absence mask, and returns the filled image::

        def inpainter(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # image: (C, H, W) float tensor — the original image
            # mask:  (H, W) bool tensor — True where pixels must be filled
            # returns: (C, H, W) float tensor
            ...

    Absent pixels are reconstructed from their visible
    surroundings, which is the principled target for image explanations. Keeping
    the inpainter a callable leaves the heavy dependency (``diffusers``, ``cv2``,
    ``skimage``) with the caller. Cost is one inpainter call per coalition, so
    this is by far the most expensive pixel strategy.

    Args:
        inpainter: Callable mapping ``(image, mask) -> image``.

    Example:
        >>> from skimage.restoration import inpaint_biharmonic
        >>> def inpainter(image, mask):
        ...     filled = inpaint_biharmonic(
        ...         image.permute(1, 2, 0).cpu().numpy(), mask.cpu().numpy(), channel_axis=-1
        ...     )
        ...     return torch.from_numpy(filled).permute(2, 0, 1).to(image)
        >>> masking = InpaintingMasking(inpainter=inpainter)

    """

    def __init__(self, inpainter: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """Initialize the inpainting masking strategy.

        Args:
            inpainter: Callable ``f(image, mask) -> image``. ``image`` is a
                ``(C, H, W)`` float tensor, ``mask`` a ``(H, W)`` boolean tensor
                marking pixels to fill, and the return is a ``(C, H, W)`` tensor.

        Raises:
            TypeError: If ``inpainter`` is not callable.
        """
        if not callable(inpainter):
            msg = f"inpainter must be callable, got {type(inpainter).__name__}."
            raise TypeError(msg)
        self.inpainter = inpainter

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply inpainting masking to absent player regions.

        Coalitions that hide nothing skip the inpainter and keep the original image.

        Raises:
            TypeError: If the inpainter does not return a tensor shaped like the image.
        """
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)

        filled = []
        for mask in pixel_mask:
            if not bool(mask.any()):
                filled.append(image)
                continue
            candidate = self.inpainter(image, mask)
            if not isinstance(candidate, torch.Tensor) or candidate.shape != image.shape:
                shape = getattr(candidate, "shape", None)
                msg = (
                    f"{type(self).__name__} expects the inpainter to return a (C, H, W) tensor "
                    f"shaped like the image {tuple(image.shape)}, got "
                    f"{type(candidate).__name__} with shape {tuple(shape) if shape else None}."
                )
                raise TypeError(msg)
            filled.append(candidate.to(device=image.device, dtype=image.dtype))

        return torch.stack(filled)


class LatentBasedMaskingStrategy(MaskingStrategy, ABC):
    """Base class for token-space masking strategies used with ViT models.

    Implementations convert a coalition matrix into a ``bool_masked_pos``
    tensor suitable for passing directly to a ViT forward call.
    ``accepted_coalition_domain`` is ``CoalitionDomain.TOKEN``.
    """

    accepted_coalition_domain: CoalitionDomain = CoalitionDomain.TOKEN

    @abstractmethod
    def apply(
        self,
        coalitions: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Convert coalitions to a token-level boolean mask.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present.
            token_masks: Integer tensor of shape
                ``(n_players, tokens_per_player)`` mapping each player to its
                flat token indices.

        Returns:
            Boolean tensor of shape ``(n_coalitions, n_tokens)`` where
            ``True`` means the token is masked (player absent) and ``False``
            means the token is visible (player present).
        """
        ...

    def _to_token_mask(
        self,
        coalitions: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a coalition tensor to a flat token-level boolean mask.

        Tokens belonging to absent players are set to ``True`` (masked);
        tokens belonging to present players are set to ``False`` (visible).

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present.
            token_masks: Integer tensor of shape
                ``(n_players, tokens_per_player)`` containing the flat token
                indices for each player.

        Returns:
            Boolean tensor of shape ``(n_coalitions, n_tokens)`` on the same
            device as ``coalitions``.
        """
        n_players = token_masks.shape[0]
        n_tokens = int(token_masks.max()) + 1

        token_masks = token_masks.to(coalitions.device)

        player_to_token = torch.zeros(
            (n_players, n_tokens), dtype=torch.bool, device=coalitions.device
        )
        player_to_token.scatter_(dim=1, index=token_masks, value=True)

        visible = coalitions.float() @ player_to_token.float()
        return ~visible.bool()


class BoolMaskedPosStrategy(LatentBasedMaskingStrategy):
    """Masks tokens by passing ``bool_masked_pos`` directly to the model forward call.

    This strategy requires the model to support the ``bool_masked_pos``
    argument (e.g. :class:`~transformers.ViTForMaskedImageModeling`).

    Unlike :class:`MaskTokenStrategy`, it does not initialize the model's
    ``mask_token``: the model must already own one, which Hugging Face only
    provides when the model was built with ``use_mask_token=True``. If not setting the
    mask token, the model output will not be meaningful.
    """

    @classmethod
    def validate_model(cls, model: Model) -> None:
        """Validate that ``model`` satisfies the declared protocol and owns a mask token.

        Args:
            model: Object to validate against ``compatible_model_protocol`` and
                supports model.vit.embeddings.mask_token.

        Raises:
            TypeError: If ``model`` is not compatible with the declared protocol,
                does not expose ``vit.embeddings.mask_token``, or leaves it unset.
        """
        super().validate_model(model)
        try:
            mask_token = model.vit.embeddings.mask_token
        except AttributeError:
            msg = f"{cls.__name__} requires a model exposing ``vit.embeddings.mask_token``."
            raise TypeError(msg) from None
        if mask_token is None:
            msg = (
                f"{cls.__name__} requires a model built with ``use_mask_token=True`` (e.g. "
                "``ViTForMaskedImageModeling``), but ``vit.embeddings.mask_token`` is None. "
                "Use MaskTokenStrategy(model) instead, which initializes the mask token or set the mask_token yourself."
            )
            raise TypeError(msg)

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply boolean masking by converting coalitions to a ``bool_masked_pos`` tensor."""
        return self._to_token_mask(coalitions, token_masks)


class MaskTokenStrategy(LatentBasedMaskingStrategy):
    """Masks tokens by zeroing the mask_token embedding before the forward pass."""

    def __init__(self, model: Model) -> None:
        """Initialize with the ViT model whose mask token will be zeroed.

        Args:
            model: A ViT model with a ``vit.embeddings.mask_token``
                parameter.
        """
        self._model = model
        type(self).validate_model(model)

    @classmethod
    def validate_model(cls, model: Model) -> None:
        """Validate that ``model`` satisfies the declared protocol and exposes required attributes.

        Args:
            model: Object to validate against ``compatible_model_protocol`` and supports
                model.vit.embeddings.mask_token and model.config.hidden_size.

        Raises:
            TypeError: If ``model`` does not support the required attributes or is not
                compatible with the declared protocol.
        """
        super().validate_model(model)
        try:
            embeddings = model.vit.embeddings
            _ = embeddings.mask_token
        except AttributeError:
            msg = f"{cls.__name__} requires a model exposing ``vit.embeddings.mask_token``."
            raise TypeError(msg) from None
        validate_config_attributes(
            model,
            ("hidden_size",),
            cls.__name__,
            hint="If your model is not supporting this attribute, consider using BoolMaskedPosStrategy instead and set a mask_token in the model's config yourself.",
        )

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply masking by setting the model's mask_token embedding to zero.

        The mask token is created on the model's device: a CPU parameter on a
        CUDA model makes the forward pass fail inside the embedding layer.
        """
        embeddings = self._model.vit.embeddings
        mask_token = embeddings.mask_token
        device = get_torch_device(self._model)
        dtype = _model_dtype(self._model)
        if (
            mask_token is None
            or mask_token.device != device
            or mask_token.dtype != dtype
            or mask_token.any()
        ):
            embeddings.mask_token = torch.nn.Parameter(
                torch.zeros(1, 1, self._model.config.hidden_size, device=device, dtype=dtype)
            )
        return self._to_token_mask(coalitions, token_masks)
