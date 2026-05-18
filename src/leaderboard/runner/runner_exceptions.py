"""Runner Exceptions."""

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