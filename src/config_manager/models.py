"""
Configuration Data Models and Validators

This module contains all Pydantic models and their associated validators
for the benchmark configuration system.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional
from .constants import VALID_INDICES, ALL_SUPPORTED_APPROXIMATORS


# --- Ground Truth Configuration Model ---
class GroundTruthConfig(BaseModel):
    """Configuration for ground truth computation or lookup."""
    strategy: str = Field(default="compute")  # "compute" or "lookup"
    method: str = Field(default="ExactComputer")
    storage_path: Optional[str] = Field(default=None)


# --- Main Configuration Model (MVP) ---
class MVPRunConfig(BaseModel):
    """
    Main configuration model for MVP benchmark runs.
    
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
    def validate_budgets(self):
        """Ensure all budgets are positive integers."""
        for b in self.budgets:
            if b <= 0:
                raise ValueError(
                    f"Budget must be greater than 0, found illegal value: {b}"
                )
        return self

    @model_validator(mode="after")
    def validate_approximators(self):
        """
        Validate that approximators are supported and compatible with the chosen index.
        
        Performs two checks:
        1. Whitelist check: approximator exists in supported list
        2. Compatibility check: approximator supports the chosen index
        """
        import shapiq

        for app in self.approximators:
            # 1. Check if it's in the hardcoded whitelist
            if app not in ALL_SUPPORTED_APPROXIMATORS:
                raise ValueError(
                    f"Approximator '{app}' not supported or spelling error. Valid list: {ALL_SUPPORTED_APPROXIMATORS}"
                )

            # 2. Dynamic attribute check and Index matching
            try:
                app_class = getattr(shapiq.approximator, app)
            except AttributeError:
                raise ValueError(
                    f"Approximator '{app}' not found in shapiq.approximator!"
                )

            # Complete index validation against __init__.py categories
            if self.index == "SV" and app_class not in shapiq.approximator.SV_APPROXIMATORS:
                raise ValueError(f"'{app}' does not support SV calculation!")
            elif self.index in ["SII", "k-SII"] and app_class not in shapiq.approximator.SII_APPROXIMATORS:
                raise ValueError(f"'{app}' does not support SII / k-SII calculation!")
            elif self.index == "STII" and app_class not in shapiq.approximator.STII_APPROXIMATORS:
                raise ValueError(f"'{app}' does not support STII calculation!")
            elif self.index == "FSII" and app_class not in shapiq.approximator.FSII_APPROXIMATORS:
                raise ValueError(f"'{app}' does not support FSII calculation!")
            elif self.index == "FBII" and app_class not in shapiq.approximator.FBII_APPROXIMATORS:
                raise ValueError(f"'{app}' does not support FBII calculation!")
            elif self.index in ["BV", "BII", "CHII"]:
                # If shapiq doesn't have explicit lists for these yet, you can pass or define custom logic
                pass

        return self

    @model_validator(mode="after")
    def validate_order(self):
        """
        Validate max_order based on the chosen index.
        
        Rules:
        - SV: max_order must be 1
        - Interaction indices (SII, STII, FSII): max_order must be >= 2
        """
        if self.index == "SV" and self.max_order != 1:
            raise ValueError("When computing SV, max_order must be 1")
        if self.index in ["SII", "STII", "FSII"] and self.max_order < 2:
            raise ValueError(
                f"When computing interaction index {self.index}, max_order must be at least 2"
            )
        return self
