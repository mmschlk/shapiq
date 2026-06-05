from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision import transforms

from .masking import LatentMaskingStrategy, MaskTokenStrategy, MeanColorMasking, PixelMaskingStrategy
from .players import LatentPlayerStrategy, PatchStrategy, PixelPlayerStrategy, PlayerStrategy, SuperpixelStrategy


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


class ResNetArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for CNN models (e.g. ResNet) using pixel-space masking."""

    def __init__(self, model, masking_strategy: PixelMaskingStrategy | None = None, player_strategy: PixelPlayerStrategy | None = None):
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
        # Convert (H, W, C) to (C, H, W)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image)

    def prepare(self, image: np.ndarray) -> None:
        self._player_masks = self._player_strategy.get_masks(image)
        self._image_array = image
        
        input_tensor = self._preprocess_image(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            self._class_id = int(logits.argmax(dim=1).item())

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        masked = self._masking_strategy.apply(self._image_array, self._player_masks, coalitions)
        input_tensors = torch.stack([self._preprocess_image(img) for img in masked])
        
        with torch.no_grad():
            logits = self.model(input_tensors)
        return logits[:, self._class_id].numpy()


class ViTArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for Vision Transformer models using latent-space masking."""

    def __init__(self, model, vit_processor, masking_strategy: LatentMaskingStrategy | None = None, player_strategy: LatentPlayerStrategy | None = None):
        self.model = model
        self.processor = vit_processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._pixel_values: torch.Tensor | None = None
        self._class_id: int | None = None

    def default_player_strategy(self) -> PatchStrategy:
        grid_size = self.model.config.image_size // self.model.config.patch_size
        return PatchStrategy(grid_size=grid_size, n_players=9)

    def default_masking_strategy(self) -> MaskTokenStrategy:
        # ViTForImageClassification has mask_token=None by default; MaskTokenStrategy initialises it
        return MaskTokenStrategy()

    def prepare(self, image: np.ndarray) -> None:
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            logits = self.model(pixel_values=self._pixel_values).logits
        self._class_id = int(logits.argmax(-1).item())
        self._player_masks = self._build_pixel_masks(image)

    def _build_pixel_masks(self, image: np.ndarray) -> np.ndarray:
        """Rectangular grid masks of shape (n_players, H, W) for pixel-space visualization."""
        n = self._player_strategy.n_players
        H, W = image.shape[:2]
        side = self._player_strategy.side
        bh, bw = H // side, W // side
        masks = np.zeros((n, H, W), dtype=bool)
        for p in range(n):
            r, c = divmod(p, side)
            masks[
                p,
                r * bh : (H if r == side - 1 else (r + 1) * bh),
                c * bw : (W if c == side - 1 else (c + 1) * bw),
            ] = True
        return masks

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        bool_masks = torch.stack(
            [self._player_strategy.get_latent_mask(c) for c in coalitions]
        )
        with torch.no_grad():
            logits = self._masking_strategy.predict_logits(self.model, self._pixel_values, bool_masks)
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self._class_id].cpu().numpy()
