import torch
import math
import numpy as np
from typing import Optional, Literal
from abc import ABC, abstractmethod

class PlayerStrategy(ABC):
    """Defines how the image is split into n_players regions."""

    @property
    @abstractmethod
    def n_players(self) -> int: ...


class PixelPlayerStrategy(PlayerStrategy, ABC):
    """Player strategy that returns spatial masks in pixel space."""

    @abstractmethod
    def get_masks(self, image: np.ndarray) -> np.ndarray:
        # returns (n_players, H, W)
        ...


class LatentPlayerStrategy(PlayerStrategy, ABC):
    """Player strategy that returns a 1D boolean mask in latent/token space."""

    @abstractmethod
    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        # returns (n_tokens,) bool
        ...


class PatchStrategy(LatentPlayerStrategy):
    """Splits the image into patches for ViT models."""

    def __init__(self, grid_size: int, n_players: int):
        side = int(math.sqrt(n_players))
        if side * side != n_players:
            raise ValueError("n_players must be a perfect square.")
        if grid_size % side != 0:
            raise ValueError("grid_size must be divisible by sqrt(n_players).")
        self.grid_size = grid_size
        self.patch_size = grid_size // side
        self.side = side
        self._n_players = n_players

    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        # True = masked, False = visible; shape (grid_size * grid_size,)
        mask_2d = torch.ones((self.grid_size, self.grid_size), dtype=torch.bool)
        for player, is_present in enumerate(coalition):
            if is_present:
                y = (player // self.side) * self.patch_size
                x = (player % self.side) * self.patch_size
                mask_2d[y : y + self.patch_size, x : x + self.patch_size] = False
        return mask_2d.flatten()
    
    @property
    def n_players(self) -> int:
        return self._n_players


class SuperpixelStrategy(PixelPlayerStrategy):
    """Splits the image into superpixels using SLIC or a custom mask."""

    def __init__(
        self,
        n_segments: int | None = None,
        algorithm: Literal["slic", "slico"] = "slico",
        mask: Optional[np.ndarray] = None,
    ):
        """Create a SuperpixelStrategy.

        Args:
            n_segments: Optional if no mask provided. Preferred number of segments 
                passed to `slic`.
            algorithm: Which SLIC algorithm to use. "slic" may produce smooth
                regular-sized superpixels in smooth regions and highly irregular
                superpixels in textured regions. "slico" generates regular shaped
                superpixels in both textured and non-textured regions alike.
            mask: Optional precomputed mask. Can be either:
                - 2D integer label array with shape (H, W), where each unique
                  integer denotes a superpixel (labels need not be contiguous
                  or start at 1), or
                - 3D boolean array with shape (n_players, H, W).
        """
        if mask is None and n_segments is None:
            raise ValueError("Either n_segments or mask must be provided.")
        
        self.n_segments = n_segments
        self._algorithm = algorithm
        self._custom_mask: Optional[np.ndarray] = None
        self._n_players: int = n_segments or 0

        if mask is not None:
            self.set_mask(mask)
        
    @staticmethod
    def _labels_to_masks(labels: np.ndarray) -> np.ndarray:
        """Converts a 2D integer label array to a 3D boolean mask array.
        
        Args:
            labels: (H, W) integer array.
        
        Returns:
            masks: (n_players, H, W) boolean array.
        """
        n_players = np.unique(labels)
        return (labels == n_players.reshape(-1, 1, 1))
    
    
    def set_mask(self, mask: np.ndarray) -> None:
        """Validate, convert, and store a custom mask.

        Accepts either a 2D integer label array (H, W) or a 3D boolean array
        (n_players, H, W). Shape compatibility with a specific image is checked
        in `get_masks` when the image is available.

        Args:
            mask: 2D integer label array (H, W) or 3D boolean array (n_players, H, W).

        Raises:
            ValueError: If the mask has an invalid dtype, shape, or contains
                overlapping regions.
        """
        mask = np.asarray(mask)

        if mask.ndim == 2:
            if not np.issubdtype(mask.dtype, np.integer):
                raise ValueError("2D mask must contain integer labels.")
            if mask.size == 0:
                raise ValueError("Provided 2D mask is empty.")
            mask = self._labels_to_masks(mask)

        if mask.ndim == 3:
            mask = mask.astype(bool)
            if (mask.sum(axis=0) > 1).any():
                raise ValueError(
                    "Masks are overlapping — each pixel must belong to exactly one player."
                )
            if not mask.any(axis=0).all():
                raise ValueError("Not all pixels are covered by at least one player.")
        else:
            raise ValueError(
                "mask must be either a 2D label array (H, W) or a "
                "3D boolean array (n_players, H, W)."
            ) 
            
        self._custom_mask = mask
        self.n_segments = mask.shape[0] 
        self._n_players = self.n_segments  
    
    
    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Run SLIC and return the superpixel mask.
        
        If a user-provided mask was supplied, this method
        validates it against the provided `image`. 
        Otherwise `slic` is run to compute superpixels. The algorithm may not 
        return exactly `n_segments` superpixels. The result will not be clipped
        afterwards, but it is ensured that at least `n_segments` superpixels are
        returned if possible within a reasonable number of iterations.

        Returns:
            A boolean mask array with shape (n_players, H, W) where
            masks[i, y, x] == True iff pixel (y,x) belongs to superpixel i.

        """
        
        if self._custom_mask is not None:
            if self._custom_mask.shape[1:] != image.shape[:2]:
                raise ValueError(
                    f"Custom mask shape {self._custom_mask.shape[1:]} does not match "
                    f"image shape {image.shape[:2]}."
                )
            return self._custom_mask
        
        from skimage.segmentation import slic
        
        slic_zero = self._algorithm == "slico"
        superpixels = slic(image, n_segments=self.n_segments, start_label=1, slic_zero=slic_zero)
        n_superpixels = len(np.unique(superpixels))

        if n_superpixels < self.n_segments:
            iteration, n_segments_iter = 0, self.n_segments
            while iteration < 20 and n_superpixels < self.n_segments:
                n_segments_iter += 1
                superpixels = slic(image, n_segments=n_segments_iter, start_label=1, slic_zero=slic_zero)
                n_superpixels = len(np.unique(superpixels))
                iteration += 1

        # Reset n_players to the actual number of superpixels found (which may be > n_segments)
        self._n_players = n_superpixels

        return self._labels_to_masks(superpixels)
    
    
    @property
    def n_players(self) -> int:
        return self._n_players