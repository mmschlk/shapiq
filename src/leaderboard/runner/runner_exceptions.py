"""Runner Exceptions."""

from __future__ import annotations


class MissingMetricsKeyError(KeyError):
    """Raised when a successful run record is missing the 'metrics' key."""

    def __init__(self) -> None:
        """Initialize with a message indicating the missing 'metrics' key."""
        super().__init__("Successful run record is missing 'metrics'.")


class NullMetricsError(ValueError):
    """Raised when a successful run record has metrics=None."""

    def __init__(self) -> None:
        """Initialize with a message indicating that the metrics are None."""
        super().__init__("Successful run record has metrics=None.")


class NoSuccessfulRunsError(ValueError):
    """Raised when there are no successful runs to aggregate."""

    def __init__(self) -> None:
        """Initialize with a message indicating that there are no successful runs to aggregate."""
        super().__init__("No successful runs to aggregate.")


class InteractionKeyMismatchError(ValueError):
    """Raised when the interaction keys of ground truth and approximated values do not match."""

    def __init__(self, gt_keys: set, approx_keys: set) -> None:
        """Initialize with a message indicating the mismatch in interaction keys and the specific missing keys."""
        missing_in_approx = gt_keys - approx_keys
        missing_in_gt = approx_keys - gt_keys

        super().__init__(
            "Interaction keys do not match. "
            f"Missing in approx: {len(missing_in_approx)}. "
            f"Missing in ground truth: {len(missing_in_gt)}."
        )


class UnknownGameError(ValueError):
    """Raised when a configured game is unknown."""

    def __init__(self, game_name: str, available_games: list[str] | tuple[str, ...]) -> None:
        """Initialize with the name of the game and the list of available games for display to the user."""
        available = ", ".join(available_games)
        super().__init__(
            f"Unknown game: {game_name}. Available games: {available}"
        )
