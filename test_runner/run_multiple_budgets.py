import time
from importlib.metadata import version
from typing import cast

from shapiq_games.synthetic import DummyGame
from shapiq_games.synthetic import SOUM
from shapiq.game_theory.exact import ExactComputer
from shapiq.typing import IndexType
from shapiq.approximator import KernelSHAP
from shapiq.approximator import SHAPIQ

from result_store import make_record, write_record

RESULTS_PATH = "../ui/results_raw.jsonl"

# --- Feste Parameter ---
N_PLAYERS = 10
N_BASIS_GAMES = 5
BUDGETS = [16, 32, 64, 128, 256]
SEEDS = list(range(5))
INDEX: IndexType = cast(IndexType, "SV")
MAX_ORDER = 1
SHAPIQ_VERSION = version("shapiq")
# GAME = "DummyGame"
GAME = "Soum"
# APPROXIMATOR = "KernelSHAP"
APPROXIMATOR = "SHAPIQ"

# --- Game & Ground Truth (einmal erstellen) ---
if GAME == "Soum":
    game = SOUM(n=N_PLAYERS, n_basis_games=N_BASIS_GAMES, random_state=42)
    game_params = {"n_players": N_PLAYERS, "n_basis_games": N_BASIS_GAMES}
else:
    game = DummyGame(n=N_PLAYERS)
    game_params = {"n_players": N_PLAYERS}
exact = ExactComputer(game=game)
gt = exact(index="SV", order=MAX_ORDER)

# --- Sweep ---
for budget in BUDGETS:
    for seed in SEEDS:
        if APPROXIMATOR == "KernelSHAP":
            approximator = KernelSHAP(n=N_PLAYERS, max_order=MAX_ORDER, random_state=seed)
        else:
            approximator = SHAPIQ(n=N_PLAYERS, max_order=MAX_ORDER, random_state=seed)

        try:
            t0 = time.time()
            result = approximator.approximate(budget=budget, game=game)
            runtime = time.time() - t0

            mse = float(((result.values - gt.values) ** 2).mean())
            mae = float((abs(result.values - gt.values)).mean())

            metrics = {"mse": mse, "mae": mae}

            record = make_record(
                game_name=GAME,
                game_params=game_params,
                n_players=N_PLAYERS,
                approximator_name=APPROXIMATOR,
                approximator_params={},
                shapiq_version=SHAPIQ_VERSION,
                index=INDEX,
                max_order=MAX_ORDER,
                budget=budget,
                seed=seed,
                ground_truth_method="ExactComputer",
                metrics=metrics,
                runtime_seconds=round(runtime, 4),
            )

        except Exception as e:
            record = make_record(
                game_name=GAME,
                game_params=game_params,
                n_players=N_PLAYERS,
                approximator_name=APPROXIMATOR,
                approximator_params={},
                shapiq_version=SHAPIQ_VERSION,
                index=INDEX,
                max_order=MAX_ORDER,
                budget=budget,
                seed=seed,
                ground_truth_method="ExactComputer",
                metrics=None,
                runtime_seconds=None,
                run_failed=True,
                error_message=type(e).__name__,
            )

        write_record(RESULTS_PATH, record)
        if record["run_failed"]:
            print(f"budget={budget:4d} | seed={seed} | FAILED: {record['error_message']}")
        else:
            print(f"budget={budget:4d} | seed={seed} | MSE={record['metrics']['mse']:.2e}")
