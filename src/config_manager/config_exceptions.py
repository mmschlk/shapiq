"""
Custom exceptions for the benchmark configuration system.

Defines:
- InvalidBudgetError: Raised when a budget value is invalid (e.g., non-positive).
- UnsupportedApproximatorError: Raised when an approximator name is not recognized.
- ApproximatorNotFoundError: Raised when an approximator class cannot be found in shapiq.
- ApproximatorIndexIncompatibleError: Raised when an approximator does not support the chosen index.
- InvalidOrderForIndexError: Raised when max_order is invalid for the chosen index.
"""


class InvalidBudgetError(ValueError):
    """Raised when a budget value is invalid (e.g., non-positive)."""

    def __init__(self, value: int) -> None:
        """Initialize with the invalid budget value."""
        super().__init__(f"Budget must be greater than 0, found illegal value: {value}")


class UnsupportedApproximatorError(ValueError):
    """Raised when an approximator name is not recognized."""

    def __init__(self, name: str, valid: list[str]) -> None:
        """Initialize with the invalid approximator name and list of valid names."""
        super().__init__(f"Approximator '{name}' not supported or spelling error. Valid list: {valid}")


class ApproximatorNotFoundError(ValueError):
    """Raised when an approximator class cannot be found in shapiq.approximator."""

    def __init__(self, name: str) -> None:
        """Initialize with the missing approximator name."""
        super().__init__(f"Approximator '{name}' not found in shapiq.approximator!")


class ApproximatorIndexIncompatibleError(ValueError):
    """Raised when an approximator does not support the chosen index."""

    def __init__(self, name: str, index: str) -> None:
        """Initialize with the approximator name and incompatible index."""
        super().__init__(f"'{name}' does not support {index} calculation!")


class InvalidOrderForIndexError(ValueError):
    """Raised when max_order is invalid for the chosen index."""    

    def __init__(self, index: str, *, must_be_one: bool = False) -> None:
        """Initialize with the index type and whether max_order must be 1."""
        if must_be_one:
            super().__init__("When computing SV, max_order must be 1")
        else:
            super().__init__(
                f"When computing interaction index {index}, max_order must be at least 2"
            )