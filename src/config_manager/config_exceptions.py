"""
Custom exceptions for the benchmark configuration system.
"""


class InvalidBudgetError(ValueError):
    def __init__(self, value: int) -> None:
        super().__init__(f"Budget must be greater than 0, found illegal value: {value}")


class UnsupportedApproximatorError(ValueError):
    def __init__(self, name: str, valid: list[str]) -> None:
        super().__init__(f"Approximator '{name}' not supported or spelling error. Valid list: {valid}")


class ApproximatorNotFoundError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Approximator '{name}' not found in shapiq.approximator!")


class ApproximatorIndexIncompatibleError(ValueError):
    def __init__(self, name: str, index: str) -> None:
        super().__init__(f"'{name}' does not support {index} calculation!")


class InvalidOrderForIndexError(ValueError):
    def __init__(self, index: str, *, must_be_one: bool = False) -> None:
        if must_be_one:
            super().__init__("When computing SV, max_order must be 1")
        else:
            super().__init__(
                f"When computing interaction index {index}, max_order must be at least 2"
            )