import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from shapiq.approximator import Approximator
from shapiq.game import Game
from environment_info import get_hardware_info

def create_run_record(
    *,
    game: Game,
    game_name: str,
    game_params: dict,
    approximator_class: type[Approximator],
    approximator_params: dict,
    index: str,
    max_order: int,
    budget: int,
    approx_seed: int,
    metrics: dict[str, float] | None,
    runtime_seconds: float | None,
    run_failed: bool,
    error_message: str | None,
    notes: str = "",
) -> dict:
    return {
        "run_id": str(uuid.uuid4()), #TODO: needs to be checked

        "game_name": game_name,
        "game_id": game.game_id,
        "game_params": game_params,
        "n_players": game.n_players,

        "approximator_name": approximator_class.__name__,
        "approximator_params": approximator_params,
        "shapiq_version": version("shapiq"),

        "index": index,
        "max_order": max_order,
        "budget": budget,
        "approx_seed": approx_seed,

        "ground_truth_method": "ExactComputer", #TODO: this needs to be determined during the process

        "run_failed": run_failed,
        "error_message": error_message,

        "metrics": metrics if metrics is not None else {
            "mse": None,
            "mae": None,
            "mse_normalized": None,
            "spearman": None,
            "kendall_tau": None,
            "precision_at_k": None,
        },

        "runtime_seconds": runtime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": get_hardware_info(),
        "notes": notes,
    }