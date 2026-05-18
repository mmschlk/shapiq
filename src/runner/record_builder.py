import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from shapiq.approximator import Approximator
from shapiq.game import Game
from environment_info import get_hardware_info
from typing import Any
from custom_types import InteractionIndex

def create_run_record(
    *,
    game: Game,
    game_name: str,
    game_params: dict[str, Any],
    approximator_class: type[Approximator],
    approximator_params: dict[str, Any],
    index: InteractionIndex,
    max_order: int,
    budget: int,
    approx_seed: int | None,
    metrics: dict[str, float | None] | None,
    runtime_seconds: float | None,
    run_failed: bool,
    error_message: str | None,
    notes: str = "",
) -> dict[str, Any]:
    """Create a standardized run record for one benchmark run.

    Args:
        game: The game used in the benchmark run.
        game_name: The name of the game.
        game_params: The parameters used to initialize the game.
        approximator_class: The approximator class used for the run.
        approximator_params: The parameters used to initialize the approximator.
        index: The interaction index used in the run.
        max_order: The maximum interaction order computed.
        budget: The evaluation budget available to the approximator.
        approx_seed: The approximation seed used for the run, or "None" for
            aggregate records.
        metrics: The computed metric values, or "None" if the run failed.
        runtime_seconds: The runtime of the run in seconds, if available.
        run_failed: Whether the run failed.
        error_message: The error message if the run failed, otherwise "None".
        notes: Optional notes for the run record.

    Returns:
        A dictionary containing benchmark metadata, metric values, runtime
        information, environment information, and failure status.
    """
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