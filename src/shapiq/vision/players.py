"""Player strategies for vision models, that define how to create players from images. 
Players are defined in pixel space for CNNs and token space for ViTs. 
Each strategy returns boolean masks that map each player to its corresponding pixels or tokens.

Requires scikit-image for superpixel segmentation, otherwise numpy only."""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Literal
from abc import ABC, abstractmethod

class PlayerStrategy(ABC):
    """Abstract base class for all player strategies.
    A player strategy encapsulates the rule by which an image is divided into
    ``n_players`` disjoint regions.
    """

    @property
    @abstractmethod
    def n_players(self) -> int: ...
    """Number of players (image regions) produced by this strategy.

        Returns:
            int: The number of players.
    """


class CNNPlayerStrategy(PlayerStrategy, ABC):
    """Abstract base class for player strategies that returns spatial masks in pixel space."""

    @abstractmethod
    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Compute and return per-player pixel masks for the given image.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array

        Returns:
            Boolean numpy array of shape ``(n_players, H, W)``
        """
        ...
        

class SuperpixelStrategy(CNNPlayerStrategy):
    """Partition the image into superpixels.

    Players are computed either by running the SLIC (Simple Linear Iterative
    Clustering) algorithm from ``scikit-image``, or by accepting a
    user-provided segmentation mask.

    **SLIC behaviour**: SLIC is not guaranteed to produce exactly that many superpixels as passed by ``n_segments``. If fewer than ``n_segments``
    superpixels are produced, the strategy retries with a progressively larger
    request (up to 20 additional attempts) before accepting the result.
    :attr:`n_players` always reflects the *actual* number of superpixels
    found after the algorithm ran.

    Args:
        n_segments: Preferred number of superpixels to request from SLIC.
            Required when no ``mask`` is provided; ignored once a custom mask
            is set.
        algorithm: SLIC variant to use.

            - ``"slico"`` (default): SLIC-zero — enforces equal-size
              superpixels regardless of image texture, producing a more
              uniform grid.
            - ``"slic"``: standard SLIC — segment size follows image content,
              which can yield very irregular segments in textured regions.

        mask: Optional precomputed segmentation. Accepted formats:

            - **2-D integer array** ``(H, W)``: each unique integer value
              identifies one superpixel. Labels need not be contiguous or
              start at 0.
            - **3-D boolean array** ``(n_players, H, W)``: ``mask[i]`` is
              the binary pixel mask for player ``i``. Must be non-overlapping
              (each pixel belongs to at most one player) and cover every pixel.

    Raises:
        ValueError: If neither ``n_segments`` nor ``mask`` is provided.

    Example:
        >>> strategy = SuperpixelStrategy(n_segments=16)
        >>> masks = strategy.get_masks(image)   # (n_players, H, W) bool
        >>> strategy.n_players                  # actual superpixel count
        16
    """

    def __init__(
        self,
        n_segments: int | None = None,
        algorithm: Literal["slic", "slico"] = "slico",
        mask: Optional[np.ndarray] = None,
    ):
    
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


class TransformerPlayerStrategy(PlayerStrategy, ABC):
    """Abstract base class for token-space player strategies."""

    @abstractmethod
    def get_token_masks(self) -> np.ndarray:
        """Return the flat token indices owned by each player.

        Returns:
            Integer numpy array of shape ``(n_players, tokens_per_player)``
        """
        ...


class PatchStrategy(TransformerPlayerStrategy):
    """Splits the image into patches for ViT models.
    
    Each player corresponds to a group of tokens in the latent space.
    Token indices are precomputed in the constructor and can be used
    by masking strategies to build bool_masked_pos tensors.
    """

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
        self._token_masks = self._compute_token_masks()

    def _compute_token_masks(self) -> np.ndarray:
        """Precompute token masks for each player consisting of the token indices corresponding to that player's patch.
        
        Returns:
            (n_players, tokens_per_player) integer array where
            token_masks[i] contains the flat token indices belonging to player i.
        """
        tokens_per_player = self.patch_size * self.patch_size
        indices = np.zeros((self._n_players, tokens_per_player), dtype=int)
        
        for player in range(self._n_players):
            y_start = (player // self.side) * self.patch_size
            x_start = (player % self.side) * self.patch_size
            token_idx = 0
            for y in range(y_start, y_start + self.patch_size):
                for x in range(x_start, x_start + self.patch_size):
                    indices[player, token_idx] = y * self.grid_size + x
                    token_idx += 1
        
        return indices

    def get_token_masks(self) -> np.ndarray:
        """Returns token indices per player.
        
        Returns:
            (n_players, tokens_per_player) integer array.
        """
        return self._token_masks
    
    def get_pixel_masks(self, image: np.ndarray) -> np.ndarray:
        """Build rectangular pixel-space masks for visualization.

        Returns a boolean array of shape (n_players, H, W) where each player
        corresponds to a rectangular patch of the image.
        """
        n = self._n_players
        H, W = image.shape[:2]
        side = self.side
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

    @property
    def n_players(self) -> int:
        return self._n_players


