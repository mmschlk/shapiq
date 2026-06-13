from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from config_manager import GroundTruthConfig, MVPRunConfig


@pytest.fixture
def base_mvp_config() -> MVPRunConfig:
    """Fixture providing a standard, fully valid MVPRunConfig instance."""
    return MVPRunConfig(
        game="CaliforniaHousing",
        game_family="local_xai",
        index="SV",
        max_order=1,
        n_players=14,
        game_seed=42,
        approximators=["PermutationSamplingSV"],
        budgets=[15, 100, 500, 1000, 16383],
        seeds=[0, 1, 2],
        ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
    )


def test_budget_validation_accepts_range_for_n_14(base_mvp_config: MVPRunConfig):
    """Test that the configuration properly accepts valid budgets within range."""
    assert base_mvp_config.n_players == 14
    assert base_mvp_config.budgets == [15, 100, 500, 1000, 16383]


@pytest.mark.parametrize("budget", [1, 14, 16384, 20000])
def test_budget_validation_rejects_out_of_range_values_for_n_14(budget: int):
    with pytest.raises(ValidationError, match="15 <= budget < 16384"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            max_order=1,
            n_players=14,
            game_seed=42,
            approximators=["PermutationSamplingSV"],
            budgets=[budget],
            seeds=[0, 1, 2],
            ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
        )


def test_budget_policy_validation_rejects_bad_steps():
    with pytest.raises(ValidationError, match=r"budget_policy\.steps must be greater than 0"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            max_order=1,
            n_players=14,
            game_seed=42,
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            budget_policy={"strategy": "range", "start": "n+1", "end": "2^n-1", "steps": 0},
            seeds=[0, 1, 2],
            ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
        )


def test_game_family_validation_rejects_mismatched_family():
    # CaliforniaHousing is available as both families; use a game that is local-only
    # 'Mushroom' is in LOCAL_GAMES but not in GLOBAL_GAMES according to constants.
    with pytest.raises(ValidationError, match="not available as a global_xai game"):
        MVPRunConfig(
            game="Mushroom",
            game_family="global_xai",
            index="SV",
            max_order=1,
            n_players=14,
            game_seed=42,
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            seeds=[0, 1, 2],
            ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
        )


def test_game_params_rejects_invalid_imputer():
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "Unsupported imputer 'bogus'. Available imputers: baseline, conditional, marginal"
        ),
    ):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            max_order=1,
            n_players=14,
            game_seed=42,
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            seeds=[0, 1, 2],
            game_params={"imputer": "bogus"},
            ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
        )


def test_order_validation_rejects_invalid_sv_order():
    """SV max_order must be 1"""
    with pytest.raises(ValidationError, match="When computing SV, max_order must be 1"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            max_order=2,  # Error: max_order should be 1
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            seeds=[0],
        )


@pytest.mark.parametrize("interaction_index", ["SII", "STII", "FSII"])
def test_order_validation_rejects_low_order_for_interactions(interaction_index):
    """interaction_index max_order must be >= 2"""
    with pytest.raises(ValidationError, match="max_order must be at least 2"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index=interaction_index,
            max_order=1,  # Error: max_order should be >= 2
            approximators=["SHAPIQ"],
            budgets=[100],
            seeds=[0],
        )


def test_negative_or_zero_budget_raises_invalid_budget_error():
    with pytest.raises(ValidationError, match="Budget must be greater than 0"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            approximators=["PermutationSamplingSV"],
            budgets=[0],  # Wrong budget value
            seeds=[0],
        )


def test_budget_policy_rejects_invalid_strategy():
    """Test invalid strategy string"""
    with pytest.raises(ValidationError, match=re.escape("budget_policy.strategy must be 'range'")):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            budget_policy={"strategy": "invalid_strat"},
            seeds=[0],
        )


def test_budget_policy_rejects_string_steps():
    """Test steps that cannot be converted to an integer"""
    with pytest.raises(ValidationError, match="must be an integer greater than 0"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            budget_policy={"strategy": "range", "steps": "not_a_number"},
            seeds=[0],
        )


def test_sii_interaction_index_validation_from_default():
    """Test Case 1: Validate SII interaction index combination, requiring max_order >= 2 and matching approximators."""
    config = MVPRunConfig(
        game="CaliforniaHousing",
        game_family="local_xai",
        index="SII",
        max_order=2,
        n_players=14,
        game_params={
            "model_name": "decision_tree",
            "imputer": "marginal",
            "x": 0,
        },
        ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
        approximators=["SHAPIQ", "KernelSHAPIQ"],
        budgets=[200, 1000],
        seeds=[42],
    )
    assert config.index == "SII"
    assert config.max_order == 2


def test_budget_policy_range_validation_from_default():
    """Test Case 2: Validate a completely legal budget generation policy configuration."""
    config = MVPRunConfig(
        game="CaliforniaHousing",
        game_family="local_xai",
        index="SV",
        max_order=1,
        n_players=14,
        game_params={
            "model_name": "decision_tree",
            "imputer": "marginal",
            "x": 0,
        },
        budget_policy={
            "strategy": "range",
            "start": "n+1",
            "end": "2^n-1",
            "steps": 20,
        },
        ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
        approximators=["PermutationSamplingSV"],
        budgets=[500],
        seeds=[0],
    )
    assert config.budget_policy["strategy"] == "range"


def test_game_family_validation_rejects_mismatched_family_mushroom():
    """Test Case 3: Verify that Mushroom does not support global_xai and is explicitly intercepted."""
    with pytest.raises(ValidationError, match="not available as a global_xai game"):
        MVPRunConfig(
            game="Mushroom",
            game_family="global_xai",
            index="SV",
            max_order=1,
            n_players=14,
            game_params={"model_name": "decision_tree", "imputer": "marginal"},
            ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            seeds=[0],
        )


@pytest.mark.parametrize("bad_budget", [10, 20000])
def test_budget_validation_rejects_out_of_range_bounds(bad_budget: int):
    """Test Case 4: Budgets below 15 or above 16383 should be intercepted (granular multi-boundary testing)."""
    with pytest.raises(ValidationError, match="Invalid budget value"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            max_order=1,
            n_players=14,
            game_params={
                "model_name": "decision_tree",
                "imputer": "marginal",
                "x": 0,
            },
            ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
            approximators=["PermutationSamplingSV"],
            budgets=[bad_budget],
            seeds=[0],
        )


def test_budget_policy_steps_zero_raises_error():
    """Test Case 5 (Continued): Explicitly verify interception when budget_policy steps are negative or zero."""
    with pytest.raises(ValidationError, match=r"budget_policy\.steps must be greater than 0"):
        MVPRunConfig(
            game="CaliforniaHousing",
            game_family="local_xai",
            index="SV",
            max_order=1,
            n_players=14,
            game_params={
                "model_name": "decision_tree",
                "imputer": "marginal",
                "x": 0,
            },
            budget_policy={"strategy": "range", "steps": 0},
            ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
            approximators=["PermutationSamplingSV"],
            budgets=[100],
            seeds=[0],
        )
