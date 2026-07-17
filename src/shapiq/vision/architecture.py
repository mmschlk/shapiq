"""Architecture strategies for vision model inference.

Each strategy encapsulates a model type (CNN or Vision Transformer), its
default player and masking strategies and batched coalition evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .custom_types import CoalitionDomain, VisionModel
from .masking import MaskTokenStrategy, MeanColorMasking
from .players import PatchStrategy, SuperpixelStrategy
from .utils import extract_logits, get_torch_device, to_tensor_chw
from .validation import ModelCompatible, validate_config_attributes

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

if TYPE_CHECKING:
    from shapiq.typing import Model

    from .masking import (
        LatentBasedMaskingStrategy,
        MaskingStrategy,
        PixelBasedMaskingStrategy,
    )
    from .players import (
        LatentBasedPlayerStrategy,
        PixelBasedPlayerStrategy,
        PlayerStrategy,
    )


class ModelArchitecture(ModelCompatible, ABC):
    """Encapsulates model-specific inference logic.

    Subclasses bind a player strategy and a masking strategy to a concrete
    model type and implement batched coalition evaluation via
    :meth:`value_function`. Input images are converted to tensors after player masks are generated.

    Attributes:
        coalition_domain: The coalition domain this architecture natively operates in.
        compatible_model_protocol: The model protocol this architecture accepts.

        _player_strategy: The player strategy used to define players and generate masks.
        _masking_strategy: The masking strategy used to mask absent players.
        _model: The underlying model evaluated on masked images.
    """

    _model: VisionModel
    _player_strategy: PlayerStrategy
    _masking_strategy: MaskingStrategy

    coalition_domain: CoalitionDomain
    compatible_model_protocol = VisionModel

    def _validate_configuration(self) -> None:
        """Validate that model, player strategy, and masking strategy are compatible.

        Raises:
            TypeError: If the player and masking strategies live in different
                coalition domains, or if their (consistent) domain is not the
                one this architecture operates in.
        """
        type(self._masking_strategy).validate_model(self._model)

        player_domain = self._player_strategy.coalition_domain
        masking_domain = self._masking_strategy.accepted_coalition_domain

        if player_domain is not masking_domain:
            msg = (
                "Player strategy and masking strategy are incompatible: "
                f"{type(self._player_strategy).__name__} uses coalition domain "
                f"{player_domain.value!r}, but "
                f"{type(self._masking_strategy).__name__} expects "
                f"{masking_domain.value!r}."
            )
            raise TypeError(msg)

        if player_domain is not self.coalition_domain:
            hint = (
                "Token-space strategies require ViTClassificationArchitecture and a model "
                "that honors bool_masked_pos."
                if self.coalition_domain is CoalitionDomain.PIXEL
                else "Pixel-space masking is provided by ClassificationArchitecture(model=model, "
                "processor=processor, ...), which supports any classification model, "
                "including ViT and Swin."
            )
            msg = (
                f"{type(self).__name__} operates in coalition domain "
                f"{self.coalition_domain.value!r}, but "
                f"{type(self._player_strategy).__name__} and "
                f"{type(self._masking_strategy).__name__} use {player_domain.value!r}. "
                f"{hint}"
            )
            raise TypeError(msg)

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy:
        """Return the default player strategy for this architecture."""
        ...

    @abstractmethod
    def default_masking_strategy(self) -> PixelBasedMaskingStrategy | LatentBasedMaskingStrategy:
        """Return the default masking strategy for this architecture."""
        ...

    @abstractmethod
    def prepare(self, image: np.ndarray, class_index: int | None = None) -> None:
        """Cache image-dependent state. Called before value_function.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
            class_index: Index of the class to explain.
        """
        ...

    @abstractmethod
    def value_function(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Return model predictions for each coalition.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)``.
        """
        ...

    @property
    @abstractmethod
    def player_masks(self) -> torch.Tensor:
        """Boolean pixel masks of shape ``(n_players, H, W)`` for visualization."""
        ...

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Number of players defined by the player strategy."""
        ...

    @property
    @abstractmethod
    def model(self) -> Model:
        """Return the underlying model."""
        ...


class ClassificationArchitecture(ModelArchitecture):
    """Architecture strategy for classification models using pixel-space masking.

    Players are defined in pixel space. Absent players are
    replaced by the masking strategy before the image batch is forwarded
    through the model.

    This is also the fallback path for Hugging Face models that do not
    support token masking (e.g. Swin, BEiT, MobileViT, LeViT, CvT,
    SegFormer): pass the matching ``processor`` and each masked image is
    preprocessed with it before the forward pass, with logits read from the
    output object.
    """

    _masking_strategy: PixelBasedMaskingStrategy
    _player_strategy: PixelBasedPlayerStrategy

    coalition_domain = CoalitionDomain.PIXEL

    def __init__(
        self,
        model: VisionModel,
        masking_strategy: PixelBasedMaskingStrategy | None = None,
        player_strategy: PixelBasedPlayerStrategy | None = None,
        processor: Model | None = None,
    ) -> None:
        """Initialize the classification architecture strategy.

        Args:
            model: A model evaluated on image batches — a PyTorch CNN
                (e.g. :class:`torchvision.models.ResNet`) called directly on
                the masked tensor, or any Hugging Face image classification
                model when ``processor`` is given.
            masking_strategy: Pixel-space masking strategy. Defaults to
                :class:`~shapiq.vision.masking.MeanColorMasking`.
            player_strategy: Player definition strategy. Defaults to
                :class:`~shapiq.vision.players.SuperpixelStrategy` with 10
                segments.
            processor: Optional Hugging Face image processor. When given,
                masking happens on the original image and every masked image
                is preprocessed with the processor (resize, normalize) before
                being forwarded as ``pixel_values``.

        Raises:
            TypeError: If the model is not callable or the given strategies
                are incompatible with each other or with the model.
        """
        type(self).validate_model(model)
        self._model = model
        if processor is not None and not callable(processor):
            msg = "processor must be callable."
            raise TypeError(msg)
        self._processor = processor
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._validate_configuration()
        self._player_masks: torch.Tensor
        self._image_tensor: torch.Tensor
        self._class_id: int | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return a superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        """Return a mean-color masking strategy."""
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, class_index: int | None = None) -> None:
        """Cache the image tensor, player masks, and predicted class index.

        Runs one forward pass on the unmasked image to determine the class
        index that will be tracked across all coalition evaluations.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
            class_index: Index of the class to explain.
        """
        device = get_torch_device(self._model)
        if self._processor is not None:
            arr = image.astype(np.float32)
            if image.dtype != np.uint8 and arr.size > 0 and arr.max() <= 1.0:
                arr = arr * 255.0
            self._image_tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
        else:
            self._image_tensor = to_tensor_chw(image, device=device)
        self._player_masks = torch.from_numpy(self._player_strategy.get_masks(image)).to(device)

        if class_index is not None:
            self._class_id = class_index
        elif self._class_id is None:
            with torch.no_grad():
                logits = self._forward(self._image_tensor.unsqueeze(0))
            self._class_id = int(logits.argmax(dim=1).item())

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward a ``(B, C, H, W)`` image batch and return ``(B, n_classes)`` logits.

        Without a processor the batch goes straight into the model. With a
        processor, each image is preprocessed into ``pixel_values`` before the forward pass.

        Args:
            batch: A ``(B, C, H, W)`` image batch with pixel values.

        Returns:
            A ``(B, n_classes)`` tensor of logits.

        Raises:
            TypeError: If preprocessing fails, or if the model rejects the
                expected classification interface (``pixel_values`` or
                ``(B, C, H, W)``). Errors raised *inside* a correctly called
                model (e.g. device or shape mismatches) propagate unchanged.
        """
        processor = self._processor
        pixel_values = batch if processor is None else self._preprocess_batch(batch, processor)
        try:
            if processor is None:
                output = self._model(pixel_values)
            else:
                output = self._model(pixel_values=pixel_values)
        except (TypeError, ValueError) as err:
            msg = (
                f"{type(self).__name__} could not call {type(self._model).__name__} with the "
                "expected classification interface. The model raised: "
                f"{type(err).__name__}: {err}"
            )
            raise TypeError(msg) from err

        return extract_logits(output)

    def _preprocess_batch(self, batch: torch.Tensor, processor: Model) -> torch.Tensor:
        """Convert a masked image batch to model-ready ``pixel_values``.

        Each image is converted back to a uint8 ``(H, W, C)`` array before being
        handed to the processor, which is the format image processors expect.
        Only called when a processor is configured.

        Args:
            batch: A ``(B, C, H, W)`` image batch with pixel values
            processor: The image processor to apply. Taken as an argument rather
                than read from ``self`` so the caller resolves the optional
                processor once, before deciding to preprocess at all.

        Returns:
            A ``(B, C, H, W)`` tensor of preprocessed pixel values
            transferred to the model's device.

        Raises:
            TypeError: If the processor is not callable or does not return
                a dict with a ``pixel_values`` key.
        """
        try:
            arrays = (
                batch.clamp(0.0, 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            )
            inputs = processor(images=list(arrays), return_tensors="pt")
            return inputs["pixel_values"].to(get_torch_device(self._model))
        except Exception as err:
            msg = (
                f"{type(self).__name__} could not preprocess images with the provided "
                "processor. Expected a callable processor compatible with "
                "`processor(images=..., return_tensors='pt')` returning `pixel_values`."
            )
            raise TypeError(msg) from err

    def value_function(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Evaluate the model for a batch of coalitions.

        Creates masked image tensors via the masking strategy in a single
        batched model call.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the logit for the
            predicted class for each coalition.

        Raises:
            RuntimeError: If ``prepare(image, ...)`` has not been called before
                ``value_function`` or the user has given an invalid class index
                that is not present in the model's output.
        """
        if self._class_id is None:
            msg = "Call prepare(image, ...) before value_function(...)."
            raise RuntimeError(msg)
        with torch.no_grad():
            masked_batch = self._masking_strategy.apply(
                self._image_tensor,
                self._player_masks,
                coalitions.to(self._player_masks.device),
            )
            logits = self._forward(masked_batch)
        try:
            return logits[:, self._class_id]
        except IndexError as err:
            msg = (
                f"{type(self).__name__} could not extract the score for class index "
                f"{self._class_id} from model output with shape {tuple(logits.shape)}."
            )
            raise RuntimeError(msg) from err

    @property
    def player_masks(self) -> torch.Tensor:
        """Boolean pixel masks of shape ``(n_players, H, W)``."""
        return self._player_masks

    @property
    def n_players(self) -> int:
        """Number of players defined by the player strategy."""
        return self._player_strategy.n_players

    @property
    def model(self) -> Model:
        """Return the underlying model."""
        return self._model


class ViTClassificationArchitecture(ModelArchitecture):
    """Architecture for Vision Transformer models using latent-space masking.

    Players correspond to groups of patch tokens. Absent players are masked
    in token space via ``bool_masked_pos`` before the forward pass.

    Note: ``bool_masked_pos`` is provided by the majority of Hugging Face ViT models, but not all.
    If your model does not support it, use :class:`~shapiq.vision.ClassificationArchitecture` instead.
    Be aware that masking in the token space, in which ViT models operate, is not possible then.
    """

    _model: Model
    _masking_strategy: LatentBasedMaskingStrategy
    _player_strategy: LatentBasedPlayerStrategy

    coalition_domain = CoalitionDomain.TOKEN

    def __init__(
        self,
        model: VisionModel,
        vit_processor: Model,
        masking_strategy: LatentBasedMaskingStrategy | None = None,
        player_strategy: LatentBasedPlayerStrategy | None = None,
    ) -> None:
        """Initialize the vision transformer classifier architecture.

        Args:
            model: A vision transformer model.
            vit_processor: The matching processor used to preprocess
                the image into ``pixel_values``.
            masking_strategy: Token-space masking strategy. Defaults to
                :class:`~shapiq.vision.masking.MaskTokenStrategy`.
            player_strategy: Player definition strategy. Defaults to
                :class:`~shapiq.vision.players.PatchStrategy` sized to the model's
                patch grid.

        Raises:
            TypeError: If the model is not callable or the given strategies
                are incompatible with each other or with the model.
        """
        type(self).validate_model(model)
        self._model = model
        if vit_processor is not None and not callable(vit_processor):
            msg = "vit_processor must be callable."
            raise TypeError(msg)
        self._processor = vit_processor
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._validate_configuration()
        self._pixel_values: torch.Tensor
        self._player_masks: torch.Tensor
        self._token_masks: torch.Tensor
        self._class_id: int | None = None

    def default_player_strategy(self) -> PatchStrategy:
        """Return a patch player strategy sized to the model's patch grid.

        Raises:
            TypeError: If the model's config does not define ``image_size`` and
                ``patch_size``, which are required to derive the patch grid.
        """
        validate_config_attributes(
            self._model,
            ("image_size", "patch_size"),
            f"The default player strategy of {type(self).__name__}",
            hint="Pass an explicit ``player_strategy`` if your model is not supporting these attributes.",
        )

        if self._model.config.patch_size <= 0 or self._model.config.image_size <= 0:
            msg = (
                f"The default player strategy of {type(self).__name__} requires "
                "`model.config.image_size` and `model.config.patch_size` to be non-zero. "
                "Pass an explicit `player_strategy` for this model."
            )
            raise TypeError(msg)

        grid_size = self._model.config.image_size // self._model.config.patch_size
        return PatchStrategy(
            grid_size=grid_size, n_players=PatchStrategy.default_n_players(grid_size)
        )

    def default_masking_strategy(self) -> MaskTokenStrategy:
        """Return a token-masking strategy.

        Note:
            ``ViTForImageClassification`` has ``mask_token=None`` by default;
            :class:`~shapiq.vision.masking.MaskTokenStrategy` initializes it.
        """
        return MaskTokenStrategy(self._model)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert one input image to model-ready ``pixel_values``.

        Args:
            image: A single input image as a ``(H, W, C)`` numpy array.

        Returns:
            A ``(1, C, H, W)`` tensor of preprocessed pixel values
            transferred to the model's device.

        Raises:
            TypeError: If the processor is not callable or does not return
                a dict with a ``pixel_values`` key.
        """
        try:
            inputs = self._processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(get_torch_device(self._model))
        except Exception as err:
            msg = (
                f"{type(self).__name__} could not preprocess the input image with the "
                "provided processor. Expected a callable processor compatible with "
                "`processor(images=..., return_tensors='pt')` returning `pixel_values`."
            )
            raise TypeError(msg) from err

    def _forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward a ViT batch and return ``(B, n_classes)`` logits.

        Args:
            pixel_values: A ``(B, C, H, W)`` tensor of preprocessed pixel values.
            bool_masked_pos: Optional ``(B, n_tokens)`` boolean tensor indicating
                which tokens are masked. If None, no masking is applied.

        Returns:
            A ``(B, n_classes)`` tensor of logits.

        Raises:
            TypeError: If the model rejects the expected ViT classification
                interface (``pixel_values`` and optional ``bool_masked_pos``).
                Errors raised *inside* a correctly called model (e.g. device or
                shape mismatches) propagate unchanged.
        """
        try:
            if bool_masked_pos is None:
                output = self._model(pixel_values=pixel_values)
            else:
                output = self._model(
                    pixel_values=pixel_values,
                    bool_masked_pos=bool_masked_pos,
                )
        except (TypeError, ValueError) as err:
            msg = (
                f"{type(self).__name__} could not call {type(self._model).__name__} with the "
                "expected interface, which must accept `pixel_values` and `bool_masked_pos` "
                f"arguments. The model raised: {type(err).__name__}: {err}"
            )
            raise TypeError(msg) from err

        return extract_logits(output)

    def prepare(self, image: np.ndarray, class_index: int | None = None) -> None:
        """Cache pixel values, token masks, pixel masks, and predicted class index.

        Passes ``image`` directly to the ViT processor (which expects
        a numpy ``(H, W, C)`` or PIL image), places the resulting
        ``pixel_values`` tensor on the model's device, and runs one forward
        pass to determine the predicted class index.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
            class_index: Index of the class to explain.
        """
        device = get_torch_device(self._model)
        self._pixel_values = self._preprocess_image(image)

        if class_index is not None:
            self._class_id = class_index
        elif self._class_id is None:
            with torch.no_grad():
                logits = self._forward(self._pixel_values)
            self._class_id = int(logits.argmax(-1).item())

        self._player_masks = torch.from_numpy(self._player_strategy.get_pixel_masks(image)).to(
            device
        )
        self._token_masks = torch.from_numpy(self._player_strategy.get_token_masks()).to(device)

    def value_function(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Evaluate the ViT for a batch of coalitions.

        Converts coalition membership to a ``bool_masked_pos`` tensor and
        runs a single batched forward pass.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the softmax
            probability for the predicted class for each coalition.

        Raises:
            RuntimeError: If ``prepare(image, ...)`` has not been called before
                ``value_function`` or the user has given an invalid class index
                that is not present in the model's output.
        """
        if self._class_id is None:
            msg = "Call prepare(image, ...) before value_function(...)."
            raise RuntimeError(msg)
        with torch.no_grad():
            token_mask = self._masking_strategy.apply(
                coalitions.to(self._token_masks.device),
                self._token_masks,
            )
            batch = self._pixel_values.repeat(token_mask.shape[0], 1, 1, 1)
            logits = self._forward(batch, bool_masked_pos=token_mask)
            probs = torch.softmax(logits, dim=-1)
            try:
                return probs[:, self._class_id]
            except IndexError as err:
                msg = (
                    f"{type(self).__name__} could not extract the score for class index "
                    f"{self._class_id} from model output with shape {tuple(probs.shape)}."
                )
                raise RuntimeError(msg) from err

    @property
    def player_masks(self) -> torch.Tensor:
        """Boolean pixel masks of shape ``(n_players, H, W)``."""
        return self._player_masks

    @property
    def n_players(self) -> int:
        """Number of players defined by the player strategy."""
        return self._player_strategy.n_players

    @property
    def model(self) -> Model:
        """Return the underlying model."""
        return self._model
