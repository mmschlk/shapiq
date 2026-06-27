from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from leaderboard.runner.approximator_registry import get_approximator_class
from leaderboard.runner.benchmark_runner import run_benchmark
from leaderboard.runner.custom_types import InteractionIndex
from leaderboard.runner.game_factory import create_game_from_config

# Resolve the project root dynamically from the current file's location
project_root = Path(__file__).resolve().parents[3]
config_path = project_root / "configs" / "template_visual.yaml"

# Load the visual game configuration from the YAML file
with open(config_path, "r") as f:
    raw_config = yaml.safe_load(f)

# 2. Transform the visual template into the input format expected by the runner.
# Manually construct a structure approximating base_config and run_config.
base_config = {
    "game_params": raw_config["game_params"],
    "game_family": raw_config["game_family"],
}

# --- 💡 CRITICAL FIX: Force conversion of x_explain_path to an absolute path ---
if "x_explain_path" in base_config["game_params"]:
    raw_img_path = Path(base_config["game_params"]["x_explain_path"])
    # Ensure we use project_root to join the path, rather than the current working directory.
    absolute_img_path = project_root / raw_img_path

    # Verify path existence to prevent runtime errors.
    if not absolute_img_path.exists():
        raise FileNotFoundError(f"Image not found at: {absolute_img_path}")

    base_config["game_params"]["x_explain_path"] = str(absolute_img_path)
    print(f"✅ Image path resolved: {base_config['game_params']['x_explain_path']}")

run_config = {
    "game": raw_config["game"],
    "game_seed": 42,
    "max_order": raw_config["max_order"],
    "x": 0,  # Default instance index for local explanation
}

# 3. Manually invoke the Factory to instantiate the game
game, game_params = create_game_from_config(run_config, base_config)

# 4. Execute the benchmark directly
# Run based on configurations defined in template_visual.yaml
results = run_benchmark(
    game=game,
    game_name=raw_config["game"],
    game_params=game_params,
    max_order=raw_config["max_order"],
    approx_seeds=raw_config["seeds"],
    budget=raw_config["budgets"][0],
    index=cast(InteractionIndex, raw_config["index"]),
    approximator_class=get_approximator_class(raw_config["approximators"][0]),
    ground_truth_method=raw_config["ground_truth"]["method"],
)

print("Visual game run successful!")
print(results)
