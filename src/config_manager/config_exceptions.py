"""Custom exceptions for the benchmark configuration system.

Defines:
- InvalidBudgetError: Raised when a budget value is invalid (e.g., non-positive).
- UnsupportedApproximatorError: Raised when an approximator name is not recognized.
- ApproximatorNotFoundError: Raised when an approximator class cannot be found in shapiq.
- ApproximatorIndexIncompatibleError: Raised when an approximator does not support the chosen index.
- InvalidOrderForIndexError: Raised when max_order is invalid for the chosen index.
"""

from __future__ import annotations


class InvalidBudgetError(ValueError):
    """Raised when a budget value is invalid (e.g., non-positive)."""

    def __init__(self, value: int) -> None:
        """Initialize with the invalid budget value."""
        super().__init__(f"Budget must be greater than 0, found illegal value: {value}")


class UnsupportedApproximatorError(ValueError):
    """Raised when an approximator name is not recognized."""

    def __init__(self, name: str, valid: list[str]) -> None:
        """Initialize with the invalid approximator name and list of valid names."""
        super().__init__(
            f"Approximator '{name}' not supported or spelling error. Valid list: {valid}"
        )


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


class InvalidYAMLTypeError(TypeError):
    """Raised when the loaded YAML config is not a dictionary/object at the top level."""

    def __init__(self) -> None:
        """Initialize with a message about the expected YAML structure."""
        super().__init__("YAML config must contain a dictionary/object at the top level.")


class InvalidConfigMissingFieldsError(ValueError):
    """Raised when a benchmark configuration is missing one of more required fields."""

    def __init__(self) -> None:
        """Initialize with a message describing the configuration error."""
        super().__init__("Invalid benchmark configuration")


class InvalidConfigMissingApproximatorsError(InvalidConfigMissingFieldsError):
    """Raised when a benchmark configuration is missing the 'approximator(s)' field."""

    def __init__(self) -> None:
        """Initialize with a message about the missing 'approximator(s)' field."""
        super().__init__(
            "Benchmark configuration must include 'approximator' or 'approximators' field."
        )


class InvalidConfigMissingBudgetsError(InvalidConfigMissingFieldsError):
    """Raised when a benchmark configuration is missing the 'budget(s)' field."""

    def __init__(self) -> None:
        """Initialize with a message about the missing 'budget(s)' field."""
        super().__init__("Benchmark configuration must include 'budget' or 'budgets' field.")

class UnsupportedGameError(ValueError):
    """Exception raised when a game is not supported."""

    def __init__(self, game: str, supported_games: set[str] | list[str]) -> None:
        """Initialize the UnsupportedGameError with the invalid game and allowed options."""
        formatted_games = ", ".join(sorted(supported_games))
        message = f"Unsupported game '{game}'. Available games: {formatted_games}"
        super().__init__(message)

class InvalidGameFamilyError(ValueError):
    """Exception raised when a game is incompatible with its declared family."""

    def __init__(self, game: str, game_family: str) -> None:
        """Initialize the InvalidGameFamilyError with the conflicting game and family."""
        message = f"Game '{game}' is not available as a {game_family} game."
        super().__init__(message)
class BudgetRangeError(ValueError):
    """Exception raised when a budget value falls outside the allowed range."""

    def __init__(
        self, budget: int, n_players: int, min_allowed: int, max_exclusive: int
    ) -> None:
        """Initialize the BudgetRangeError with detailed constraint bounds."""
        message = (
            f"Invalid budget value {budget}. For this project, n_players={n_players} "
            f"so budgets must satisfy {min_allowed} <= budget < {max_exclusive} "
            f"(for n=14, this is 15 <= budget < 16384). Please remove this value "
            f"or change `n_players` if your game has a different size."
        )
        super().__init__(message)
class UnsupportedImputerError(ValueError):
    """Exception raised when an imputer method is not supported."""

    def __init__(self, imputer: str, supported_imputers: set[str] | list[str]) -> None:
        """Initialize the UnsupportedImputerError with the invalid imputer and allowed options."""
        formatted_imputers = ", ".join(sorted(supported_imputers))
        message = f"Unsupported imputer '{imputer}'. Available imputers: {formatted_imputers}"
        super().__init__(message)
class InvalidBudgetStrategyError(ValueError):
    """Exception raised when an invalid budget policy strategy is provided."""

    def __init__(self, strategy: str) -> None:
        """Initialize the InvalidBudgetStrategyError with the invalid strategy value."""
        message = (
            f"budget_policy.strategy must be 'range' when provided, "
            f"but got '{strategy}'."
        )
        super().__init__(message)
class InvalidBudgetStepsError(ValueError):
    """Exception raised when budget_policy.steps is invalid."""

    def __init__(self, steps_input: object,*, is_negative: bool = False) -> None:
        """Initialize the InvalidBudgetStepsError based on the type of validation failure."""
        if is_negative:
            message = f"budget_policy.steps must be greater than 0, but got {steps_input}."
        else:
            message = (
                f"budget_policy.steps must be an integer greater than 0, "
                f"but got unacceptable value '{steps_input}'."
            )
        super().__init__(message)
