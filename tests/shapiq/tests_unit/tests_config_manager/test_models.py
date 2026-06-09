from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from config_manager import GroundTruthConfig, MVPRunConfig


def test_budget_validation_accepts_range_for_n_14():
    config = MVPRunConfig(
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

    assert config.n_players == 14
    assert config.budgets == [15, 100, 500, 1000, 16383]


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
