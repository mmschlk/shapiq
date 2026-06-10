from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

from .masking import TransformerMaskingStrategy, MaskTokenStrategy, MeanColorMasking, CNNMaskingStrategy
from .players import TransformerPlayerStrategy, PatchStrategy, CNNPlayerStrategy, PlayerStrategy, SuperpixelStrategy

if TYPE_CHECKING:
    import torch


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic, decoupling it from
    :class:`~shapiq.vision.imputer.ImageImputer`.

    Subclasses bind a player strategy and a masking strategy to a concrete
    model type and implement batched coalition evaluation via
    :meth:`value_function`.
    Input images are converted to tensors after player masks are generated.
    """

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy: ...

    @abstractmethod
    def default_masking_strategy(self): ...

    @abstractmethod
    def prepare(self, image: np.ndarray) -> None:
        """Cache image-dependent state. Called once by ImageImputer before value_function."""
        ...

    @abstractmethod
    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Return model predictions for each coalition. Returns (n_coalitions,)."""
        ...
        
    @property
    def player_masks(self) -> torch.Tensor:
        """(n_players, H, W) boolean array of pixel masks for visualization."""
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

    def __init__(self, model, masking_strategy: CNNMaskingStrategy | None = None, player_strategy: CNNPlayerStrategy | None = None):
        self.model = model
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._player_masks: torch.Tensor | None = None
        self._image_tensor: torch.Tensor | None = None
        self._class_id: int | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def prepare(self, image: np.array) -> None:
        """Cache the image tensor, player masks, and predicted class index.

        Runs one forward pass on the unmasked image to determine the class
        index that will be tracked across all coalition evaluations.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
        """
        import torch
        from .utils import get_torch_device, to_tensor_chw
       
        device = get_torch_device(self.model)
        self._image_tensor = to_tensor_chw(image, device=device)
        self._player_masks = torch.from_numpy(self._player_strategy.get_masks(image)).to(device)
    
        with torch.no_grad():
            logits = self.model(self._image_tensor.unsqueeze(0))
            self._class_id = int(logits.argmax(dim=1).item())

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Evaluate the CNN for a batch of coalitions.

        Creates masked image tensors via the masking strategy in a single
        batched model call.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the logit for the
            predicted class for each coalition.
        """
        import torch
        
        with torch.no_grad():
            masked_batch = self._masking_strategy.apply(self._image_tensor, self._player_masks, coalitions)
            logits = self.model(masked_batch)
        return logits[:, self._class_id]
    
    @property
    def player_masks(self) -> torch.Tensor:
        return self._player_masks


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
            :class:`~shapiq.vision.players.PatchStrategy` with 9 players
            derived from the model's patch grid.
    """


    def __init__(self, model, vit_processor, masking_strategy: TransformerMaskingStrategy | None = None, player_strategy: TransformerPlayerStrategy | None = None):
        self.model = model
        self.processor = vit_processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._pixel_values: torch.Tensor | None = None
        self._player_masks: torch.Tensor | None = None
        self._token_masks: torch.Tensor | None = None
        self._class_id: int | None = None

    def default_player_strategy(self) -> PatchStrategy:
        grid_size = self.model.config.image_size // self.model.config.patch_size
        return PatchStrategy(grid_size=grid_size, n_players=9)

    def default_masking_strategy(self) -> MaskTokenStrategy:
        # ViTForImageClassification has mask_token=None by default; MaskTokenStrategy initialises it
        return MaskTokenStrategy(self.model)

    def prepare(self, image: np.ndarray) -> None:
        """Cache pixel values, token masks, pixel masks, and predicted class index.

        Passes ``image`` directly to the ViT processor (which expects
        a numpy ``(H, W, C)`` or PIL image), places the resulting
        ``pixel_values`` tensor on the model's device, and runs one forward
        pass to determine the predicted class index.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
        """
        import torch
        from .utils import get_torch_device
        
        device = get_torch_device(self.model)
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"].to(device)
        
        with torch.no_grad():
            logits = self.model(pixel_values=self._pixel_values).logits
        self._class_id = int(logits.argmax(-1).item())
        
        self._player_masks = torch.from_numpy(self._player_strategy.get_pixel_masks(image)).to(device)
        self._token_masks = torch.from_numpy(self._player_strategy.get_token_masks()).to(device)

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Evaluate the ViT for a batch of coalitions.

        Converts coalition membership to a ``bool_masked_pos`` tensor and
        runs a single batched forward pass.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the softmax
            probability for the predicted class for each coalition.
        """
        import torch
        
        with torch.no_grad():
            token_mask = self._masking_strategy.apply(coalitions, self._token_masks)
            batch = self._pixel_values.repeat(token_mask.shape[0], 1, 1, 1)
            logits = self.model(pixel_values=batch, bool_masked_pos=token_mask).logits
            probs = torch.softmax(logits, dim=-1)
            
        return probs[:, self._class_id]
    
    @property
    def player_masks(self) -> torch.Tensor:
        return self._player_masks
