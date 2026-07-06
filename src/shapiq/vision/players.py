"""Player strategies for vision models.

Defines how to create players from images. Players are defined in pixel
space for CNNs and token space for ViTs. Requires scikit-image for
superpixel segmentation, otherwise numpy only.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


def labels_to_masks(labels: np.ndarray) -> np.ndarray:
    """Converts a 2D integer label array to a 3D boolean mask array.

    Args:
        labels: (H, W) integer array where each unique value corresponds to a player

    Returns:
        masks: (n_players, H, W) boolean array.
    """
    n_players = np.unique(labels)
    return labels == n_players.reshape(-1, 1, 1)


class PlayerStrategy(ABC):
    """Abstract base class for all player strategies.

    A player strategy encapsulates the rule by which an image is divided into
    ``n_players`` disjoint regions.
    """

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Number of players (image regions) produced by this strategy."""
        ...


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


class CustomPlayerStrategy(CNNPlayerStrategy):
    """Uses a set of pre-computed binary masks as players provided by the user.

    Provided masks may overlap — pixels covered by multiple
    players will be masked whenever any of those players is absent.

    Pixels not covered by any player mask are outside the game: they stay
    visible in every coalition because no player owns them and cannot be
    attributed or masked away. A :exc:`UserWarning` is raised when uncovered
    pixels are detected.

    Args:
        masks: either: Array of shape ``(n_players, H, W)``. Any dtype is accepted and
            will be cast to ``bool``. Should be evaluated to ``True`` for pixels
            belonging to the player and ``False`` otherwise.
            or: a 2-D integer segmentation label map of shape ``(H, W)`` where each
            unique integer corresponds to a player, with at least 2 distinct labels

    Raises:
        ValueError: If ``masks`` is not a 3-D array or any player mask is
            entirely empty.

    Example::

        # From a pre-computed boolean mask array
        strategy = CustomPlayerStrategy(masks)  # (n_players, H, W) bool
    """

    def __init__(self, masks: np.ndarray) -> None:
        """Initialize the strategy with pre-computed masks."""
        masks = np.asarray(masks)

        if masks.ndim == 2 and np.issubdtype(masks.dtype, np.integer):
            n_unique = len(np.unique(masks))
            if n_unique < 2:
                msg = (
                    f"Expected a segmentation label map with at least 2 distinct labels, "
                    f"but found only {n_unique} unique value(s). "
                )
                raise ValueError(msg)
            masks = labels_to_masks(masks)

        if masks.ndim != 3:
            msg = f"masks must be a 3-D array of shape (n_players, H, W), got shape {masks.shape}."
            raise ValueError(msg)
        self._masks = masks.astype(bool)
        self._verify(self._masks)

    @staticmethod
    def _verify(masks: np.ndarray) -> None:
        """Validate mask array and warn about uncovered pixels.

        Args:
            masks: Boolean array of shape ``(n_players, H, W)``.

        Raises:
            ValueError: If any player mask is entirely empty.
        """
        import warnings

        if not masks.any(axis=(1, 2)).all():
            empty = np.flatnonzero(~masks.any(axis=(1, 2))).tolist()
            msg = (
                f"Player mask(s) at index {empty} are entirely empty (all False). "
                "Each player must cover at least one pixel."
            )
            raise ValueError(msg)
        uncovered = (~masks.any(axis=0)).sum()
        if uncovered > 0:
            warnings.warn(
                f"{uncovered} pixel(s) are not covered by any player mask. "
                "These pixels will stay visible in every coalition and cannot be attributed.",
                UserWarning,
                stacklevel=3,
            )

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Return the pre-computed masks, validating against the image dimensions.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array. Used only for
                dimension validation — the image content is ignored.

        Returns:
            Boolean numpy array of shape ``(n_players, H, W)``.

        Raises:
            ValueError: If the mask spatial dimensions do not match the image.
        """
        if self._masks.shape[1:] != image.shape[:2]:
            msg = (
                f"Mask spatial dimensions {self._masks.shape[1:]} do not match "
                f"image dimensions {image.shape[:2]}."
            )
            raise ValueError(msg)
        return self._masks

    @property
    def n_players(self) -> int:
        """Number of players (image regions) produced by this strategy."""
        return self._masks.shape[0]


class GridStrategy(CNNPlayerStrategy):
    """Splits the image into a regular rectangular grid of players.

    The strategy must be initialized implicitly via :meth:`get_masks`
    before :attr:`n_players` can be accessed.

    Exactly one of ``patch_size`` or ``grid_shape`` must be provided:

    - ``grid_shape``: fixes the number of tiles; the grid dimensions are set
    directly and patch sizes are derived from the image shape at fit time.
    The image is divided using floor division — any remainder pixels are
    absorbed into the last row and/or column, making the edge patches
    potentially larger than the interior patches.
    - ``patch_size``: fixes the pixel size of each patch; the grid dimensions
    are inferred from the image shape at fit time. Some patches may be
    smaller than ``patch_size`` if the image dimensions are not exact multiples.

    Args:
        patch_size: ``(patch_height, patch_width)`` or a single int for square
            patches. The grid shape is inferred from the image.
        grid_shape: ``(grid_y, grid_x)`` or a single int for a square grid.
            The patch size is derived from the image.

    Raises:
        ValueError: If both or neither of ``patch_size`` and ``grid_shape``
            are provided.

    Example::

        # Fixed grid of 4x4 tiles — edge patches absorb remainder pixels
        strategy = GridStrategy(grid_shape=4)
        masks = strategy.get_masks(image)  # (16, H, W)

        # Fixed 32x32 patches — last row/column may be smaller if H or W
        # is not a multiple of 32
        strategy = GridStrategy(patch_size=32)
        masks = strategy.get_masks(image)  # (n_players, H, W)
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int] | None = None,
        grid_shape: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the strategy with either a patch size or grid shape."""
        if (patch_size is None) == (grid_shape is None):
            msg = "Must provide exactly one of 'patch_size' or 'grid_shape'."
            raise ValueError(msg)

        self._mode = "patch" if patch_size is not None else "grid"

        if patch_size is not None:
            self._input_patch_size: int | tuple[int, int] = patch_size
        if grid_shape is not None:
            self._input_grid_shape: int | tuple[int, int] = grid_shape

        self._is_initialized = False
        self.h: int
        self.w: int
        self.grid_y: int
        self.grid_x: int

    def _resolve_grid_shape(self, h: int, w: int) -> tuple[int, int]:
        """Resolve ``grid_shape`` or ``patch_size`` into ``(grid_y, grid_x)``.

        Args:
            h: Image height in pixels.
            w: Image width in pixels.

        Returns:
            Tuple ``(grid_y, grid_x)`` — the number of tiles along each axis.

        Raises:
            ValueError: If the resolved grid or patch dimensions are invalid
                given the image shape.
        """
        if self._mode == "grid":
            gy, gx = (
                (self._input_grid_shape, self._input_grid_shape)
                if isinstance(self._input_grid_shape, int)
                else self._input_grid_shape
            )
            if gy < 1 or gx < 1:
                msg = "Grid dimensions must be positive integers."
                raise ValueError(msg)
            if gy > h or gx > w:
                msg = (
                    f"Grid shape {(gy, gx)} exceeds image shape {(h, w)}. "
                    "This would result in empty players."
                )
                raise ValueError(msg)
            return gy, gx

        # patch mode
        ph, pw = (
            (self._input_patch_size, self._input_patch_size)
            if isinstance(self._input_patch_size, int)
            else self._input_patch_size
        )
        if ph < 1 or pw < 1:
            msg = "Patch dimensions must be positive integers."
            raise ValueError(msg)
        if ph > h or pw > w:
            msg = f"Patch size {(ph, pw)} exceeds image shape {(h, w)}."
            raise ValueError(msg)
        return math.ceil(h / ph), math.ceil(w / pw)

    @staticmethod
    def _build_player_grid(h: int, w: int, gy: int, gx: int) -> np.ndarray:
        """Build a ``(H, W)`` integer label map for a ``gy x gx`` grid.

        Uses integer floor division so every pixel is assigned exactly one
        player. Edge patches absorb any remainder pixels.

        Args:
            h: Image height in pixels.
            w: Image width in pixels.
            gy: Number of grid rows.
            gx: Number of grid columns.

        Returns:
            Integer numpy array of shape ``(H, W)`` with values in
            ``[0, gy * gx)``.
        """
        row_edges = [r * h // gy for r in range(gy + 1)]
        col_edges = [c * w // gx for c in range(gx + 1)]

        row_assign = np.repeat(np.arange(gy), np.diff(row_edges))
        col_assign = np.repeat(np.arange(gx), np.diff(col_edges))

        return row_assign[:, None] * gx + col_assign[None, :]

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Return per-patch boolean masks of shape ``(n_players, H, W)``.

        Resolves and caches the grid dimensions on the first call using the
        provided image shape.

        Args:
            image: Input image as a ``(H, W[, C])`` numpy array.

        Returns:
            Boolean numpy array of shape ``(n_players, H, W)`` where
            ``masks[i, y, x] == True`` iff pixel ``(y, x)`` belongs to
            player ``i``.
        """
        h, w = image.shape[:2]
        gy, gx = self._resolve_grid_shape(h, w)
        player_grid = self._build_player_grid(h, w, gy, gx)

        self.h, self.w = h, w
        self.grid_y, self.grid_x = gy, gx
        self._is_initialized = True

        return player_grid == np.arange(gy * gx)[:, None, None]

    @property
    def n_players(self) -> int:
        """Number of players (image regions) produced by this strategy.

        Raises:
            RuntimeError: If called before :meth:`get_masks`.
        """
        if not self._is_initialized:
            msg = "Call `get_masks(image)` first to compute the number of players."
            raise RuntimeError(msg)
        return self.grid_y * self.grid_x


class SuperpixelStrategy(CNNPlayerStrategy):
    """Splits the image into superpixels using SLIC.

    Uses the SLIC or SLICO algorithm from :mod:`skimage.segmentation` to
    partition the image into compact, perceptually uniform regions. The
    algorithm may not return exactly ``n_segments`` superpixels; the
    implementation iterates up to 20 times with an increasing segment count
    to ensure at least ``n_segments`` are returned where possible.

    To use a pre-computed segmentation, convert it first with
    :func:`labels_to_masks` and pass the result to
    :class:`CustomPlayerStrategy`.

    Args:
        n_segments: Preferred number of superpixels to request from SLIC.
        algorithm: SLIC variant to use. Either ``"slico"`` (default) for
            SLIC-zero, which enforces equal-size superpixels regardless of
            image texture, or ``"slic"`` for standard SLIC, where segment
            size follows image content and can yield irregular segments in
            textured regions.

    Raises:
        ValueError: If neither ``n_segments`` nor ``mask`` is provided.

    Example:
        >>> strategy = SuperpixelStrategy(n_segments=16)
        >>> masks = strategy.get_masks(image)   # (n_players, H, W) bool
        >>> strategy.n_players  # actual superpixel count, may differ from 16
        16
    """

    def __init__(self, n_segments: int, algorithm: Literal["slic", "slico"] = "slic") -> None:
        """Initialize the strategy with the desired number of superpixels and algorithm."""
        if n_segments < 1:
            msg = "n_segments must be a positive integer."
            raise ValueError(msg)

        self.n_segments = n_segments
        self._algorithm = algorithm
        self._n_players: int = n_segments

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Run SLIC and return the superpixel mask.

        The algorithm may not return exactly `n_segments` superpixels.
        The result will not be clipped afterwards, but it is ensured that at
        least `n_segments` superpixels are returned if possible within a
        reasonable number of iterations.

        Returns:
            A boolean mask array with shape (n_players, H, W) where
            masks[i, y, x] == True iff pixel (y,x) belongs to superpixel i.

        """
        try:
            from skimage.segmentation import slic
        except ImportError as err:
            from ._error import _vision_import_error

            raise _vision_import_error from err

        slic_zero = self._algorithm == "slico"
        superpixels = slic(image, n_segments=self.n_segments, start_label=1, slic_zero=slic_zero)
        n_superpixels = len(np.unique(superpixels))

        if n_superpixels < self.n_segments:
            iteration, n_segments_iter = 0, self.n_segments
            while iteration < 20 and n_superpixels < self.n_segments:
                n_segments_iter += 1
                superpixels = slic(
                    image, n_segments=n_segments_iter, start_label=1, slic_zero=slic_zero
                )
                n_superpixels = len(np.unique(superpixels))
                iteration += 1

        # Reset n_players to the actual number of superpixels found (which may be > n_segments)
        self._n_players = n_superpixels

        return labels_to_masks(superpixels)

    @property
    def n_players(self) -> int:
        """Number of players (image regions) produced by this strategy."""
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

    @abstractmethod
    def get_pixel_masks(self, image: np.ndarray) -> np.ndarray:
        """Return pixel-space boolean masks for visualization.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.

        Returns:
            Boolean numpy array of shape ``(n_players, H, W)``.
        """
        ...


class PatchStrategy(TransformerPlayerStrategy):
    """Splits the image into patches for ViT models.

    Each player corresponds to a group of tokens in the latent space.
    Token indices are precomputed in the constructor and can be used
    by masking strategies to build bool_masked_pos tensors.
    """

    def __init__(self, grid_size: int, n_players: int) -> None:
        """Initialize the strategy with the grid size and number of players."""
        side = int(math.sqrt(n_players))
        if side * side != n_players:
            msg = "n_players must be a perfect square."
            raise ValueError(msg)
        if grid_size % side != 0:
            msg = "grid_size must be divisible by sqrt(n_players)."
            raise ValueError(msg)
        self.grid_size = grid_size
        self.patch_size = grid_size // side
        self.side = side
        self._n_players = n_players
        self._token_masks = self._compute_token_masks()

    @staticmethod
    def default_n_players(grid_size: int) -> int:
        """Return a default player count whose square root divides ``grid_size``.

        Picks the divisor ``d`` of ``grid_size`` (preferring ``d >= 2``) whose square
        is closest to nine, keeping the 3x3 default where the grid allows it and
        adapting otherwise so :class:`PatchStrategy` never rejects the default.
        """
        divisors = [d for d in range(1, grid_size + 1) if grid_size % d == 0]
        candidates = [d for d in divisors if d >= 2] or divisors
        side = min(candidates, key=lambda d: (abs(d * d - 9), d))
        return side * side

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
        """Number of players (image regions) produced by this strategy."""
        return self._n_players
