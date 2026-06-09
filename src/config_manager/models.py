"""Configuration Data Models and Validators.

This module contains all Pydantic models and their associated validators for the benchmark configuration system.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

import shapiq

from .config_exceptions import (
    ApproximatorIndexIncompatibleError,
    ApproximatorNotFoundError,
    BudgetRangeError,
    InvalidBudgetError,
    InvalidBudgetStepsError,
    InvalidBudgetStrategyError,
    InvalidGameFamilyError,
    InvalidOrderForIndexError,
    UnsupportedApproximatorError,
    UnsupportedGameError,
    UnsupportedImputerError,
)
from .constants import (
    ALL_SUPPORTED_APPROXIMATORS,
    GLOBAL_GAMES,
    LOCAL_GAMES,
    SUPPORTED_GAMES,
    SUPPORTED_IMPUTERS,
    VALID_INDICES,
)


# --- Ground Truth Configuration Model ---
class GroundTruthConfig(BaseModel):
    """Configuration for ground truth computation or lookup."""

    strategy: str = Field(default="compute")  # "compute" or "lookup"
    method: str = Field(default="ExactComputer")
    storage_path: str | None = Field(default=None)


# --- Main Configuration Model (MVP) ---
class MVPRunConfig(BaseModel):
    """Main configuration model for MVP benchmark runs.

    Each configuration targets a fixed Game and Index, with sweep parameters
    for approximators, budgets, and random seeds.
    """

    # MVP requirement: Each Sweep targets a fixed Game and Index
    game: str = Field(..., min_length=1)
    # family: 'local_xai' or 'global_xai' -- controls which game parameter set applies
    game_family: Literal["local_xai", "global_xai"] = Field(default="local_xai")
    index: VALID_INDICES = Field(...)
    max_order: int = Field(default=1, ge=1)
    game_seed: int = Field(default=42)
    # Number of game players/features used for budget validation.
    # The current project convention assumes n = 14 unless configured otherwise.
    n_players: int = Field(default=14, ge=1)

    # Sweep parameters: Accepts lists for Runner iteration
    approximators: list[str] = Field(..., min_length=1)
    budgets: list[int] = Field(..., min_length=1)
    seeds: list[int] = Field(..., min_length=1)
    game_params: dict[str, Any] = Field(default_factory=dict)
    # Optional policy describing how budgets should be generated. If empty, runner will
    # use the explicit `budgets` list. Example:
    # budget_policy: strategy: 'range', start: 'n+1', end: '2^n-1', steps: 20
    budget_policy: dict[str, Any] = Field(default_factory=dict)

    # Nested GT configuration
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)

    # --- Validation Logic ---
    @model_validator(mode="after")
    def validate_game(self) -> MVPRunConfig:
        """Validate game name against the runner's supported game registry."""
        if self.game not in SUPPORTED_GAMES:
            raise UnsupportedGameError(self.game, SUPPORTED_GAMES) from None

        # verify membership according to declared family
        if (self.game_family == "local_xai" and self.game not in LOCAL_GAMES) or (self.game_family == "global_xai" and self.game not in GLOBAL_GAMES):
            raise InvalidGameFamilyError(self.game, self.game_family) from None
        return self

    @model_validator(mode="after")
    def validate_budgets(self) -> MVPRunConfig:
        """Ensure all budgets satisfy the project range rule [n+1, 2^n)."""
        min_allowed = self.n_players + 1
        max_exclusive = 2**self.n_players
        for b in self.budgets:
            if b <= 0:
                raise InvalidBudgetError(b) from None
            if b < min_allowed or b >= max_exclusive:
                raise BudgetRangeError(
                    budget=b,
                    n_players=self.n_players,
                    min_allowed=min_allowed,
                    max_exclusive=max_exclusive,
                ) from None
        return self

    @model_validator(mode="after")
    def validate_approximators(self) -> MVPRunConfig:
        """Validate that approximators are supported and compatible with the chosen index.

        Performs two checks:
        1. Whitelist check: approximator exists in supported list
        2. Compatibility check: approximator supports the chosen index
        """
        for app in self.approximators:
            # 1. Check if it's in the hardcoded whitelist
            if app not in ALL_SUPPORTED_APPROXIMATORS:
                raise UnsupportedApproximatorError(app, ALL_SUPPORTED_APPROXIMATORS) from None

            # 2. Dynamic attribute check and Index matching
            try:
                app_class = getattr(shapiq.approximator, app)
            except AttributeError:
                raise ApproximatorNotFoundError(app) from None

            # Complete index validation against __init__.py categories
            if (self.index == "SV" and app_class not in shapiq.approximator.SV_APPROXIMATORS) or (
                self.index in ["SII", "k-SII"]
                and app_class not in shapiq.approximator.SII_APPROXIMATORS
            ):
                raise ApproximatorIndexIncompatibleError(app, self.index) from None
            if (
                (self.index == "STII" and app_class not in shapiq.approximator.STII_APPROXIMATORS)
                or (
                    self.index == "FSII" and app_class not in shapiq.approximator.FSII_APPROXIMATORS
                )
                or (
                    self.index == "FBII" and app_class not in shapiq.approximator.FBII_APPROXIMATORS
                )
            ):
                raise ApproximatorIndexIncompatibleError(app, self.index) from None
            if self.index in ["BV", "BII", "CHII"]:
                # If shapiq doesn't have explicit lists for these yet, you can pass or define custom logic
                pass

        return self

    @model_validator(mode="after")
    def validate_order(self) -> MVPRunConfig:
        """Validate max_order based on the chosen index.

        Rules:
        - SV: max_order must be 1
        - Interaction indices (SII, STII, FSII): max_order must be >= 2
        """
        if self.index == "SV" and self.max_order != 1:
            raise InvalidOrderForIndexError(self.index, must_be_one=True) from None
        if self.index in ["SII", "STII", "FSII"] and self.max_order < 2:
            raise InvalidOrderForIndexError(self.index) from None
        return self

    @model_validator(mode="after")
    def validate_game_params(self) -> MVPRunConfig:
        """Validate optional game-specific parameters used by the game factory."""
        imputer = self.game_params.get("imputer")
        if imputer is not None and imputer not in SUPPORTED_IMPUTERS:
            raise UnsupportedImputerError(imputer, SUPPORTED_IMPUTERS) from None
        return self

    @model_validator(mode="after")
    def validate_budget_policy(self) -> MVPRunConfig:
        """Validate optional budget policy metadata when provided."""
        if not self.budget_policy:
            return self

        strategy = self.budget_policy.get("strategy")
        if strategy is not None and strategy != "range":
            raise InvalidBudgetStrategyError(strategy) from None

        steps = self.budget_policy.get("steps")
        if steps is not None:
            # Be permissive about YAML types (e.g. "10"), but require an integer > 0.
            try:
                steps_value = int(steps)
            except (ValueError, TypeError):
                raise InvalidBudgetStepsError(steps_input=steps, is_negative=False) from None

            if steps_value <= 0:
                raise InvalidBudgetStepsError(steps_input=steps_value, is_negative=True) from None

        return self
