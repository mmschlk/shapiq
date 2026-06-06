from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

from .masking import TransformerMaskingStrategy, MaskTokenStrategy, MeanColorMasking, CNNMaskingStrategy
from .players import TransformerPlayerStrategy, PatchStrategy, CNNPlayerStrategy, PlayerStrategy, SuperpixelStrategy

if TYPE_CHECKING:
    import torch


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic, decoupling it from ImageImputer."""

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy: ...

    @abstractmethod
    def default_masking_strategy(self): ...

    @abstractmethod
    def prepare(self, image: np.ndarray, player_strategy: PlayerStrategy) -> None:
        """Cache image-dependent state. Called once by ImageImputer before value_function."""
        ...

    @abstractmethod
    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return model predictions for each coalition. Returns (n_coalitions,)."""
        ...
        
    @property
    def player_masks(self) -> np.ndarray:
        """(n_players, H, W) boolean array of pixel masks for visualization."""
        ...


class CNNArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for CNN models (e.g. ResNet) using pixel-space masking."""

    def __init__(self, model, masking_strategy: CNNMaskingStrategy | None = None, player_strategy: CNNPlayerStrategy | None = None):
        self.model = model
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._player_masks: np.ndarray | None = None
        self._image_array: np.ndarray | None = None
        self._class_id: int | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()
    
    def _preprocess_image(self, image: np.ndarray) -> torch.tensor:
        from torchvision import transforms
        # Convert (H, W, C) to (C, H, W)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image)

    def prepare(self, image: np.ndarray) -> None:
        import torch
        self._player_masks = self._player_strategy.get_masks(image)
        self._image_array = image
        
        input_tensor = self._preprocess_image(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            self._class_id = int(logits.argmax(dim=1).item())

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        import torch
        masked = self._masking_strategy.apply(self._image_array, self._player_masks, coalitions)
        input_tensors = torch.stack([self._preprocess_image(img) for img in masked])
        
        with torch.no_grad():
            logits = self.model(input_tensors)
        return logits[:, self._class_id].numpy()
    
    @property
    def player_masks(self) -> np.ndarray:
        return self._player_masks


class TransformerArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for Vision Transformer models using latent-space masking."""

    def __init__(self, model, vit_processor, masking_strategy: TransformerMaskingStrategy | None = None, player_strategy: TransformerPlayerStrategy | None = None):
        self.model = model
        self.processor = vit_processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._pixel_values: torch.Tensor | None = None
        self._token_masks: np.ndarray | None = None
        self._class_id: int | None = None

    def default_player_strategy(self) -> PatchStrategy:
        grid_size = self.model.config.image_size // self.model.config.patch_size
        return PatchStrategy(grid_size=grid_size, n_players=9)

    def default_masking_strategy(self) -> MaskTokenStrategy:
        # ViTForImageClassification has mask_token=None by default; MaskTokenStrategy initialises it
        return MaskTokenStrategy(self.model)

    def prepare(self, image: np.ndarray) -> None:
        import torch
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            logits = self.model(pixel_values=self._pixel_values).logits
        self._class_id = int(logits.argmax(-1).item())
        self._player_masks = self._player_strategy.get_pixel_masks(image)
        self._token_masks = self._player_strategy.get_token_masks()

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        import torch
        player_token_indices = self._player_strategy.get_token_masks()
        
        with torch.no_grad():
            token_mask = self._masking_strategy.apply(coalitions, player_token_indices)
            batch = self._pixel_values.repeat(token_mask.shape[0], 1, 1, 1)
            logits = self.model(pixel_values=batch, bool_masked_pos=token_mask).logits
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self._class_id].cpu().numpy()
    
    @property
    def player_masks(self) -> np.ndarray:
        return self._player_masks
