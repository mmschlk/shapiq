from abc import ABC, abstractmethod
import numpy as np
import torch

class PixelMaskingStrategy(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """
        Args:
            image:        (H, W, C) original image
            player_masks: (n_players, H, W) boolean masks per player
            coalition:    (n_coalitions, n_players) boolean array
        
        Returns:
            masked_images: (n_coalitions, H, W, C)
        """
        ...


class MeanColorMasking(PixelMaskingStrategy):
    """Imputes the masked pixels with the mean color of the entire image."""
    
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape
        
        masked_images = np.stack([image] * n_coalitions, axis=0) # shape (n_coalitions, H, W, C)
        
        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = image.mean(axis=(0, 1))
        return masked_images


class ZeroMasking(PixelMaskingStrategy):
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape
        
        masked_images = np.stack([image] * n_coalitions, axis=0) # shape (n_coalitions, H, W, C)
        
        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = self.value
        return masked_images


class LatentMaskingStrategy(ABC):
    """Defines how tokens are masked in latent/embedding space."""

    @abstractmethod
    def predict_logits(
        self,
        model,
        pixel_values: torch.Tensor,  # (1, 3, H, W)
        bool_masks: torch.Tensor,    # (B, n_tokens)
    ) -> torch.Tensor:               # (B, n_classes)
        ...


class BoolMaskedPosStrategy(LatentMaskingStrategy):
    """Masks tokens via the bool_masked_pos argument in the model forward pass."""

    def predict_logits(self, model, pixel_values, bool_masks):
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits


class MaskTokenStrategy(LatentMaskingStrategy):
    """Masks tokens by zeroing the mask_token embedding before the forward pass."""

    def predict_logits(self, model, pixel_values, bool_masks):
        model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, model.config.hidden_size)
        )
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits
