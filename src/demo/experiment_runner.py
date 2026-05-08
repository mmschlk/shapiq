import time

from approximator_runner import approximate
from metrics_computer import compute_metrics
from record_builder import create_run_record


def run_experiment(
    *,
    game,
    game_name: str,
    game_params: dict,
    game_seed,
    ground_truth,
    approximator_class,
    index,
    max_order,
    budget,
    approx_seeds,
    metrics,
) -> list[dict]:
    results = []
    for approx_seed in approx_seeds:
        start_time = time.perf_counter()
        try:
            approx_values = approximate(
                game=game,
                approximator_class=approximator_class,
                index=index,
                max_order=max_order,
                budget=budget,
                seed=approx_seed,
            )

            # calculate metrics for each run
            metric_results: dict[str, float] = compute_metrics(
                ground_truth=ground_truth,
                approximation=approx_values,
                metrics=metrics
            )

            runtime_seconds = time.perf_counter() - start_time

            run_record = create_run_record(
                game=game,
                game_name=game_name,
                game_params=game_params,
                approximator_class=approximator_class,
                approximator_params={
                    "random_state": approx_seed,
                },
                index=index,
                max_order=max_order,
                budget=budget,
                approx_seed=approx_seed,
                metrics={
                    "mse": metric_results.get("mse"),
                    "mae": metric_results.get("mae"),
                    "mse_normalized": None,
                    "spearman": None,
                    "kendall_tau": None,
                    "precision_at_k": None,
                },
                runtime_seconds=runtime_seconds,
                run_failed=False,
                error_message=None,
                notes="",
            )

        except Exception as error:
            runtime_seconds = time.perf_counter() - start_time

            run_record = create_run_record(
                game=game,
                game_name=game_name,
                game_params=game_params,
                approximator_class=approximator_class,
                approximator_params={
                    "random_state": approx_seed,
                },
                index=index,
                max_order=max_order,
                budget=budget,
                approx_seed=approx_seed,
                metrics=None,
                runtime_seconds=runtime_seconds,
                run_failed=True,
                error_message=str(error),
                notes="",
            )

        results.append(run_record)
        return results