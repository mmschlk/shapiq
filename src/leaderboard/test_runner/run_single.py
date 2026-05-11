import time
import shapiq
from result_store import make_record, write_record
from shapiq_games.synthetic import DummyGame
from importlib.metadata import version
SHAPIQ_VERSION = version("shapiq")

RESULTS_PATH = "../ui/results_raw.jsonl"

# --- Feste Parameter ---
N_PLAYERS = 10
BUDGET = 256
SEED = 42
INDEX = "SV"
MAX_ORDER = 1

# --- Game ---
game = DummyGame(n=N_PLAYERS)

# --- Ground Truth ---
# game wird direkt übergeben, n_players wird automatisch aus game.n_players gelesen
exact = shapiq.ExactComputer(game=game)
gt = exact(index="SV", order=MAX_ORDER)  # gibt InteractionValues zurück

# --- Approximator ---
approximator = shapiq.approximator.KernelSHAP(
    n=N_PLAYERS,
    max_order=MAX_ORDER,
    random_state=SEED
)

try:
    t0 = time.time()
    result = approximator.approximate(budget=BUDGET, game=game)
    runtime = time.time() - t0

    # InteractionValues haben .values als numpy array
    # beide müssen gleiche Einträge haben (gleiche Koalitionen)
    gt_values = gt.values
    approx_values = result.values

    mse = float(((approx_values - gt_values) ** 2).mean())
    mae = float((abs(approx_values - gt_values)).mean())

    metrics = {
        "mse": mse,
        "mae": mae,
    }

    record = make_record(
        game_name="DummyGame",
        game_params={"n_players": N_PLAYERS, "interaction": False},
        n_players=N_PLAYERS,
        approximator_name="KernelSHAP",
        approximator_params={},
        shapiq_version=SHAPIQ_VERSION,
        index=INDEX,
        max_order=MAX_ORDER,
        budget=BUDGET,
        seed=SEED,
        ground_truth_method="ExactComputer",
        metrics=metrics,
        runtime_seconds=round(runtime, 4),
    )

except Exception as e:
    record = make_record(
        game_name="DummyGame",
        game_params={"n_players": N_PLAYERS, "interaction": False},
        n_players=N_PLAYERS,
        approximator_name="KernelSHAP",
        approximator_params={},
        shapiq_version=SHAPIQ_VERSION,
        index=INDEX,
        max_order=MAX_ORDER,
        budget=BUDGET,
        seed=SEED,
        ground_truth_method="ExactComputer",
        metrics=None,
        runtime_seconds=None,
        run_failed=True,
        error_message=type(e).__name__,
    )

write_record(RESULTS_PATH, record)

if record["run_failed"]:
    print(f"Run FAILED: {record['error_message']}")
else:
    print(f"Run OK — MSE: {record['metrics']['mse']:.20f}, MAE: {record['metrics']['mae']:.20f}")
    print(f"Runtime: {record['runtime_seconds']}s")
