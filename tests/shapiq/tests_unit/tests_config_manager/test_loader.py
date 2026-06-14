from __future__ import annotations

from pathlib import Path

from config_manager import GroundTruthConfig, MVPRunConfig
from config_manager.loader import load_and_validate_config
from leaderboard.runner.runner_with_config_demo import expand_validated_config


def test_expand_validated_config_creates_correct_cartesian_product():
    """Verify that multiple approximators and budgets expand into a correct matrix of tasks."""
    config = MVPRunConfig(
        game="CaliforniaHousing",
        game_family="local_xai",
        index="SV",
        max_order=1,
        n_players=14,
        game_seed=100,
        approximators=["OwenSamplingSV", "StratifiedSamplingSV"],  # 2 items
        budgets=[100, 500, 1000],  # 3 items
        seeds=[1, 2],
        ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
    )

    # Execute the expansion logic
    expanded_list = expand_validated_config(config)

    # 1. Assert correct length: 2 approximators * 3 budgets = 6 run configs
    assert len(expanded_list) == 6

    # 2. Assert structural correctness of the first item
    first_task = expanded_list[0]
    assert first_task["approximator"] == "OwenSamplingSV"
    assert first_task["budget"] == 100
    assert first_task["game_seed"] == 100
    assert first_task["seeds"] == [1, 2]

    # 3. Assert uniqueness of the product combinations
    combinations = {(task["approximator"], task["budget"]) for task in expanded_list}
    assert len(combinations) == 6


def test_loader_successfully_parses_real_default_yaml():
    """Ensure that the production default_run.yaml file can be correctly discovered and loaded."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4]

    # Construct the clean path to the target configuration file
    yaml_path = project_root / "configs" / "default_run.yaml"

    # Optional debugging check: ensure your path calculation actually finds the file
    assert yaml_path.exists(), f"Configuration file not found at calculated path: {yaml_path}"

    # Act
    config_obj = load_and_validate_config(yaml_path)

    # Assert
    assert config_obj is not None
    assert config_obj.game == "CaliforniaHousing"
    assert "OwenSamplingSV" in config_obj.approximators
    assert config_obj.game_params.get("imputer") == "marginal"
