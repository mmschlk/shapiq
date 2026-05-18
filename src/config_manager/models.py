"""Configuration Data Models and Validators.

This module contains all Pydantic models and their associated validators for the benchmark configuration system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .config_exceptions import (
    ApproximatorIndexIncompatibleError,
    ApproximatorNotFoundError,
    InvalidBudgetError,
    InvalidOrderForIndexError,
    UnsupportedApproximatorError,
)
from .constants import ALL_SUPPORTED_APPROXIMATORS, VALID_INDICES


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
    index: VALID_INDICES = Field(...)
    max_order: int = Field(default=1, ge=1)

    # Sweep parameters: Accepts lists for Runner iteration
    approximators: list[str] = Field(..., min_length=1)
    budgets: list[int] = Field(..., min_length=1)
    seeds: list[int] = Field(..., min_length=1)

    # Nested GT configuration
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)

    # --- Validation Logic ---
    @model_validator(mode="after")
    def validate_budgets(self) -> MVPRunConfig:
        """Ensure all budgets are positive integers."""
        for b in self.budgets:
            if b <= 0:
                raise InvalidBudgetError(b) from None
        return self

    @model_validator(mode="after")
    def validate_approximators(self) -> MVPRunConfig:
        """Validate that approximators are supported and compatible with the chosen index.

        Performs two checks:
        1. Whitelist check: approximator exists in supported list
        2. Compatibility check: approximator supports the chosen index
        """
        import shapiq

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
