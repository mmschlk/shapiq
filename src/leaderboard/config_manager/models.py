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
    CLASSIFICATION_GAMES,
    GAME_PLAYER_COUNTS,
    GLOBAL_GAMES,
    LOCAL_GAMES,
    REGRESSION_GAMES,
    SUPPORTED_GAMES,
    SUPPORTED_IMPUTERS,
    SUPPORTED_VISUAL_MODELS,
    VALID_INDICES,
    VISUAL_GAMES,
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
        if self.game in GAME_PLAYER_COUNTS:
            self.n_players = GAME_PLAYER_COUNTS[self.game]
        if self.game == "SOUM":
            return self
        # verify membership according to declared family
        if (self.game_family == "local_xai" and self.game not in LOCAL_GAMES) or (
            self.game_family == "global_xai" and self.game not in GLOBAL_GAMES
        ):
            raise InvalidGameFamilyError(self.game, self.game_family) from None
        return self

    @model_validator(mode="after")
    def validate_budgets(self) -> MVPRunConfig:
        """Filter out invalid budgets falling outside the allowed range [n+1, 2^n)"""
        min_allowed = self.n_players + 1
        max_exclusive = 2**self.n_players

        cleaned_budgets = []
        for b in self.budgets:
            # Skip non-positive or out-of-range budgets silently
            if b > 0 and min_allowed <= b < max_exclusive:
                cleaned_budgets.append(b)

        # If ALL budgets were invalid, we still need at least one fallback to prevent downstream crash
        if not cleaned_budgets:
            fallback_budget = min_allowed * 2
            cleaned_budgets.append(fallback_budget)

        # Re-assign back to the field permanently to update model_dump
        self.budgets = cleaned_budgets
        return self

    @model_validator(mode="after")
    def validate_approximators(self) -> MVPRunConfig:
        """Filter out un-runnable, unsupported, or index-incompatible approximators
        instead of crashing the process.
        """
        # 1. Block indices that are completely unsupported by the runner pipeline
        NOT_RUNNABLE_INDICES = ["BV", "BII", "CHII"]
        if self.index in NOT_RUNNABLE_INDICES:
            # Index incompatibility is a structural error; we still allow a fallback index or empty apps
            self.approximators = []
            return self

        cleaned_apps = []
        for app in self.approximators:
            # Check 1: Whitelist membership
            if app not in ALL_SUPPORTED_APPROXIMATORS:
                continue

            # Check 2: Existence inside shapiq core library
            try:
                app_class = getattr(shapiq.approximator, app)
            except AttributeError:
                continue

            # Check 3: Dynamic Index Compatibility Checks
            if (
                self.index in ["SV", "kADD-SHAP"]
                and app_class not in shapiq.approximator.SV_APPROXIMATORS
            ):
                continue
            if (
                self.index in ["SII", "k-SII"]
                and app_class not in shapiq.approximator.SII_APPROXIMATORS
            ):
                continue
            if self.index == "STII" and app_class not in shapiq.approximator.STII_APPROXIMATORS:
                continue
            if self.index == "FSII" and app_class not in shapiq.approximator.FSII_APPROXIMATORS:
                continue
            if self.index == "FBII" and app_class not in shapiq.approximator.FBII_APPROXIMATORS:
                continue

            # If all checks pass, keep the approximator
            cleaned_apps.append(app)

        # Re-assign the strictly cleaned/filtered whitelist back to the Pydantic instance
        self.approximators = cleaned_apps
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
        if self.index in ["SII", "k-SII", "STII", "FSII", "FBII"] and self.max_order < 2:
            raise InvalidOrderForIndexError(self.index) from None
        return self

    @model_validator(mode="after")
    def validate_game_params(self) -> MVPRunConfig:
        """Validate and dynamically purge family-incompatible parameters at the configuration layer.

        Ensures the underlying game factory receives a strictly filtered and clean parameter dictionary.
        """

        # Create a mutable copy of game_params to ensure Pydantic registers the field mutation permanently
        cleaned_params = dict(self.game_params)

        # Intercept visual games first to bypass tabular-specific validation
        if self.game in VISUAL_GAMES:
            model_name = cleaned_params.get("model_name")

            # 💡 NEW: Logic to purge n_superpixel_resnet if not resnet_18
            if model_name != "resnet_18":
                cleaned_params.pop("n_superpixel_resnet", None)

            # Validation for model_name
            if model_name not in SUPPORTED_VISUAL_MODELS:
                raise ValueError(
                    f"Invalid visual model '{model_name}'. Supported models: {SUPPORTED_VISUAL_MODELS}"
                )

            # Validation for mandatory path parameter
            if "x_explain_path" not in cleaned_params:
                raise ValueError("Visual games require 'x_explain_path' in game_params.")

            # Purge tabular-specific parameters
            tabular_forbidden = [
                "imputer",
                "x",
                "class_to_explain",
                "loss_function",
                "n_samples_eval",
                "n_samples_empty",
            ]
            for key in tabular_forbidden:
                cleaned_params.pop(key, None)

            # Commit changes and exit early
            self.game_params = cleaned_params
            return self

        # 1. For pure synthetic mathematical game (SOUM), purge all machine learning/tabular-specific parameters
        if self.game == "SOUM":
            soum_forbidden = [
                "model_name",
                "imputer",
                "x",
                "class_to_explain",
                "normalize",
                "verbose",
                "loss_function",
                "n_samples_eval",
                "n_samples_empty",
            ]
            for key in soum_forbidden:
                cleaned_params.pop(key, None)

            # Sync back to persistence layer
            self.game_params = cleaned_params
            return self

        if self.game != "SOUM":
            soum_exclusive = ["n", "n_basis_games", "min_interaction_size", "max_interaction_size"]
            for key in soum_exclusive:
                cleaned_params.pop(key, None)

        # 2. For global explanation games (global_xai), strictly purge parameters unique to local explanations
        if self.game_family == "global_xai":
            global_forbidden = ["imputer", "x", "class_to_explain"]
            for key in global_forbidden:
                cleaned_params.pop(key, None)

            self.game_params = cleaned_params
            return self

        # 3. For local explanation games (local_xai), perform standard parameter validation and precise cleaning
        if self.game_family == "local_xai":
            local_forbidden = ["loss_function", "n_samples_eval", "n_samples_empty"]
            for key in local_forbidden:
                cleaned_params.pop(key, None)
            # If it's a regression game, class_to_explain is completely illegal. Purge it dynamically!
            if self.game in REGRESSION_GAMES:
                cleaned_params.pop("class_to_explain", None)

        # 3. For local explanation games (local_xai), perform standard parameter validation
        imputer = cleaned_params.get("imputer")
        if imputer is not None and imputer not in SUPPORTED_IMPUTERS:
            raise UnsupportedImputerError(imputer, SUPPORTED_IMPUTERS) from None

        # Sync back the cleaned state to the model object permanently
        self.game_params = cleaned_params
        return self

    @model_validator(mode="after")
    def validate_visual_game_constraints(self) -> MVPRunConfig:
        """Enforce strict constraints for visual games defined in template_visual.yaml."""
        if self.game != "ImageClassifier":
            return self

        # 1. Enforce Ground Truth constraint: ExactComputer only
        if self.ground_truth.method != "ExactComputer":
            raise ValueError(
                "CRITICAL: Visual games (ImageClassifier) are neural networks. "
                "'TreeExplainer' is not applicable. Please use 'ExactComputer'."
            )

        # 2. Enforce n_players consistency
        # Mismatches between n_players and model patch count will crash shapiq
        model_name = self.game_params.get("model_name")

        # Define expected player counts per model
        vit_patch_map = {
            "vit_9_patches": 9,
            "vit_16_patches": 16,
        }

        if model_name == "resnet_18":
            expected_n = self.game_params.get("n_superpixel_resnet", 14)
        else:
            expected_n = vit_patch_map.get(model_name)

        if expected_n is not None and self.n_players != expected_n:
            raise ValueError(
                f"Configuration Mismatch: '{model_name}' requires n_players={expected_n}, "
                f"but your config specifies n_players={self.n_players}. Please align them."
            )

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

    @model_validator(mode="after")
    def validate_gt_method_for_large_games(self) -> MVPRunConfig:
        """Ensure ExactComputer is not used for large games (n > 14) to prevent freezing.
        UNLESS it's SOUM which bypasses it.
        """
        if self.game == "SOUM":
            return self
        if self.n_players > 14 and self.ground_truth.strategy == "compute":
            if self.ground_truth.method == "ExactComputer":
                if self.game_family == "global_xai":
                    raise ValueError(
                        f"CRITICAL: Global SAGE games with n > 14 ({self.game}) cannot be evaluated exactly. "
                        f"Please switch to an approximate fallback dataset reference or reduce n_players."
                    )
                raise ValueError(
                    f"CRITICAL CONFIG ERROR: Game '{self.game}' has {self.n_players} players. "
                    f"Using 'ExactComputer' requires 2^{self.n_players} ({2**self.n_players}) "
                    f"model evaluations, which will freeze or crash your machine.\n"
                    f"SOLUTION: Please change 'ground_truth.method' to 'TreeExplainer' "
                    f"in your configuration file to utilize polynomial-time exact computation."
                )
        return self

    @model_validator(mode="after")
    def validate_gt_method_for_global_games(self) -> MVPRunConfig:
        """Ensure TreeExplainer is never used for global_xai games.

        GlobalExplanation games return dataset-wide loss values rather than raw tree model
        predictions, making polynomial-time tree conversion mathematically impossible.
        """
        if self.game_family == "global_xai" and self.ground_truth.strategy == "compute":
            if self.ground_truth.method in ["TreeExplainer"]:
                raise ValueError(
                    f"CONFIG ERROR: Game '{self.game}' belongs to 'global_xai'. "
                    f"Global XAI games wrap loss function logic (<class 'method'>) rather than "
                    f"raw tree structures. You MUST use 'ExactComputer' as the ground_truth.method."
                )
        return self
