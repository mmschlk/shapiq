from config_manager import MVPRunConfig, GroundTruthConfig
from leaderboard.runner.runner_with_config_demo import expand_validated_config

def test_expand_validated_config():
    """Default test case to check if the run_configs contain all expected data."""
    mvp_run_config = MVPRunConfig(
        game="CaliforniaHousing",
        index="SV",
        max_order=1,
        game_seed=0,
        approximators=["PermutationSamplingSV"],
        budgets=[100, 500],
        seeds=[0, 1, 2],
        ground_truth=GroundTruthConfig(
            strategy="compute",
            method="ExactComputer")
    )
    run_configs = expand_validated_config(mvp_run_config)
    assert len(run_configs) == 2

    budgets = [run_config["budget"] for run_config in run_configs]
    assert budgets == [100, 500]

    for run_config in run_configs:
        assert run_config["game"] == "CaliforniaHousing"
        assert run_config["index"] == "SV"
        assert run_config["approximator"] == "PermutationSamplingSV"
        assert run_config["max_order"] == 1
        assert run_config["seeds"] == [0, 1, 2]
        assert run_config["game_seed"] == 0

def test_expand_validated_config_with_multiple_approximators():
    """Test that correct run_configs are produced for multiple approximators."""
    mvp_run_config = MVPRunConfig(
        game="BikeSharing",
        index="SV",
        max_order=1,
        game_seed=0,
        approximators=["StratifiedSamplingSV", "PermutationSamplingSV"],
        budgets=[100, 200, 500],
        seeds=[0, 1, 2],
        ground_truth=GroundTruthConfig(
            strategy="compute",
            method="ExactComputer"),
    )

    run_configs = expand_validated_config(mvp_run_config)
    assert len(run_configs) == 6

    combinations = set()
    for run_config in run_configs:
        combination = (
            run_config["approximator"],
            run_config["budget"],
        )
        combinations.add(combination)
    assert combinations == {
        ("StratifiedSamplingSV", 100),
        ("StratifiedSamplingSV", 200),
        ("StratifiedSamplingSV", 500),
        ("PermutationSamplingSV", 100),
        ("PermutationSamplingSV", 200),
        ("PermutationSamplingSV", 500),
    }

def test_expand_validated_config_preserves_concrete_seeds():
    """This test checks if the concrete seeds are preserved and not just the number of seeds."""
    mvp_run_config = MVPRunConfig(
        game="CaliforniaHousing",
        index="SV",
        max_order=1,
        game_seed=99,
        approximators=["PermutationSamplingSV"],
        budgets=[100],
        seeds=[42, 123, 999],
        ground_truth=GroundTruthConfig(
            strategy="compute",
            method="ExactComputer",
        ),
    )

    run_configs = expand_validated_config(mvp_run_config)

    assert len(run_configs) == 1
    assert run_configs[0]["seeds"] == [42, 123, 999]
    assert run_configs[0]["game_seed"] == 99
