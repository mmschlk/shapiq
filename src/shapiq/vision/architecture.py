"""Architecture strategies for vision model inference.

Each strategy encapsulates a model type (CNN or Vision Transformer), its
default player and masking strategies and batched coalition evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .masking import MaskTokenStrategy, MeanColorMasking
from .players import PatchStrategy, SuperpixelStrategy
from .utils import get_torch_device, to_tensor_chw

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

if TYPE_CHECKING:
    import numpy as np

    from shapiq.typing import Model

    from .masking import (
        CNNMaskingStrategy,
        TransformerMaskingStrategy,
    )
    from .players import (
        CNNPlayerStrategy,
        PlayerStrategy,
        TransformerPlayerStrategy,
    )


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic.

    Subclasses bind a player strategy and a masking strategy to a concrete
    model type and implement batched coalition evaluation via
    :meth:`value_function`. Input images are converted to tensors after player masks are generated.
    """

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy:
        """Return the default player strategy for this architecture."""
        ...

    @abstractmethod
    def default_masking_strategy(self) -> CNNMaskingStrategy | TransformerMaskingStrategy:
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


class CNNArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for CNN models (e.g. ResNet) using pixel-space masking.

    Players are defined in pixel space. Absent players are
    replaced by the masking strategy before the image batch is forwarded
    through the model.

    Args:
        model: A PyTorch CNN model (e.g. :class:`torchvision.models.ResNet`).
        masking_strategy: Pixel-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.MeanColorMasking`.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.SuperpixelStrategy` with 10
            segments.
    """

    def __init__(
        self,
        model: Model,
        masking_strategy: CNNMaskingStrategy | None = None,
        player_strategy: CNNPlayerStrategy | None = None,
    ) -> None:
        """Initialise the CNN architecture strategy."""
        self._model = model
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
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
        self._image_tensor = to_tensor_chw(image, device=device)
        self._player_masks = torch.from_numpy(self._player_strategy.get_masks(image)).to(device)

        if not self._class_id and not class_index:
            with torch.no_grad():
                logits = self._model(self._image_tensor.unsqueeze(0))
            self._class_id = int(logits.argmax(dim=1).item())
        elif class_index is not None:
            self._class_id = class_index

    def value_function(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Evaluate the CNN for a batch of coalitions.

        Creates masked image tensors via the masking strategy in a single
        batched model call.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the logit for the
            predicted class for each coalition.
        """
        with torch.no_grad():
            masked_batch = self._masking_strategy.apply(
                self._image_tensor, self._player_masks, coalitions
            )
            logits = self._model(masked_batch)
        return logits[:, self._class_id]

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


class TransformerArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for Vision Transformer models using latent-space masking.

    Players correspond to groups of patch tokens. Absent players are masked
    in token space via ``bool_masked_pos`` before the forward pass.

    Args:
        model: A vision transformer model.
        vit_processor: The matching processor used to preprocess
            the image into ``pixel_values``.
        masking_strategy: Token-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.MaskTokenStrategy`.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.PatchStrategy` sized to the model's
            patch grid.
    """

    def __init__(
        self,
        model: Model,
        vit_processor: Model,
        masking_strategy: TransformerMaskingStrategy | None = None,
        player_strategy: TransformerPlayerStrategy | None = None,
    ) -> None:
        """Initialise the Transformer architecture strategy."""
        self._model = model
        self.processor = vit_processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._pixel_values: torch.Tensor
        self._player_masks: torch.Tensor
        self._token_masks: torch.Tensor
        self._class_id: int | None = None

    def default_player_strategy(self) -> PatchStrategy:
        """Return a patch player strategy sized to the model's patch grid."""
        grid_size = self._model.config.image_size // self._model.config.patch_size
        return PatchStrategy(
            grid_size=grid_size, n_players=PatchStrategy.default_n_players(grid_size)
        )

    def default_masking_strategy(self) -> MaskTokenStrategy:
        """Return a token-masking strategy.

        Note:
            ``ViTForImageClassification`` has ``mask_token=None`` by default;
            :class:`~shapiq.vision.masking.MaskTokenStrategy` initialises it.
        """
        return MaskTokenStrategy(self._model)

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
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"].to(device)

        if not self._class_id and not class_index:
            with torch.no_grad():
                logits = self._model(pixel_values=self._pixel_values).logits
            self._class_id = int(logits.argmax(-1).item())
        elif class_index is not None:
            self._class_id = class_index

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
        """
        with torch.no_grad():
            token_mask = self._masking_strategy.apply(coalitions, self._token_masks)
            batch = self._pixel_values.repeat(token_mask.shape[0], 1, 1, 1)
            logits = self._model(pixel_values=batch, bool_masked_pos=token_mask).logits
            probs = torch.softmax(logits, dim=-1)

        return probs[:, self._class_id]

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
