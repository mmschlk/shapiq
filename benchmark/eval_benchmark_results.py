"""Evaluate and summarize benchmark results.

This script evaluates and summarizes all benchmark results by iterating over all result
dataframes and computing summary statistics such as 'percentage of approximator being the best' or
ranking at highest budget. The results are then saved to a csv file.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

try:
    from shapiq.games.benchmark.plot import abbreviate_application_name, create_application_name
except ImportError:  # add shapiq to the path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.makedirs("eval", exist_ok=True)
    from shapiq.benchmark.plot import abbreviate_application_name, create_application_name

EVAL_DIR = Path(__file__).parent / "eval"
BENCHMARK_RESULTS_DIR = Path(__file__).parent / "results"


METRICS = {
    "Precision@10": "max",
    "Precision@5": "max",
    "MSE": "min",
    "MAE": "min",
    "SSE": "min",
    "SAE": "min",
}


APPLICATION_ORDERING = {
    "LocalExplanation": 1,
    "GlobalExplanation": 2,
    "FeatureSelection": 3,
    "DataValuation": 4,
    "DatasetValuation": 5,
    "EnsembleSelection": 6,
    "ClusterExplanation": 7,
    "UnsupervisedData": 8,
    "UncertaintyExplanation": 9,
    "SOUM (low)": 10,
    "SOUM (high)": 11,
}

SV_APPROXIMATORS_ORDERING = {
    "KernelSHAP": 1,
    "kADDSHAP": 2,
    "UnbiasedKernelSHAP": 3,
    "PermutationSamplingSV": 4,
    "StratifiedSamplingSV": 5,
    "OwenSamplingSV": 6,
    "SVARM": 7,
}


SI_APPROXIMATORS_ORDERING = {
    "KernelSHAPIQ": 1,
    "InconsistentKernelSHAPIQ": 2,
    "SHAPIQ": 3,
    "PermutationSamplingSII": 4,
    "SVARMIQ": 5,
}


def sort_values(list_to_sort: list[str], ordering: dict[str, int]) -> list[str]:
    """Sort the application names according to the APPLICATION_ORDERING."""
    sorted_list = [name for name in list_to_sort if name in ordering]
    sorted_list = sorted(sorted_list, key=lambda x: ordering[x])
    for name in list_to_sort:
        if name not in sorted_list:
            warnings.warn(f"Item {name} not in {ordering}. Appending.", stacklevel=2)
            sorted_list.append(name)
    return sorted_list


def _get_best_approximator(df: pd.DataFrame) -> dict[str, list[tuple]]:
    """Get the best (approximator, budget) for each budget for a set of metrics."""
    best_approximators = {}  # will store for each metric the approximator name that performed best
    for metric, metric_type in METRICS.items():
        if metric_type == "max":
            best_value = df.groupby("budget")[metric].max()
            best_at_budget = []
            for budget, value in best_value.items():
                best_approx_per_budget = df[(df["budget"] == budget) & (df[metric] >= value)][
                    "approximator"
                ].unique()
                for approx in best_approx_per_budget:
                    best_at_budget.append((approx, budget))  # noqa: PERF401
        else:
            best_value = df.groupby("budget")[metric].min()
            best_at_budget = []
            for budget, value in best_value.items():
                best_approx_per_budget = df[(df["budget"] == budget) & (df[metric] <= value)][
                    "approximator"
                ].unique()
                for approx in best_approx_per_budget:
                    best_at_budget.append((approx, budget))
        best_approximators[metric] = best_at_budget
    return best_approximators


def create_eval_csv(n_evals: int | None = None) -> pd.DataFrame:
    """Create a summary csv file from all benchmark results."""
    from shapiq.benchmark.run import load_benchmark_results

    # get all files in the benchmark results directory
    all_benchmark_results = list(os.listdir(BENCHMARK_RESULTS_DIR))
    all_benchmark_results = [result for result in all_benchmark_results if result.endswith(".json")]
    # iterate over all benchmark results
    all_results: list[dict] = []
    for eval_i, benchmark_result in tqdm(
        enumerate(all_benchmark_results),
        total=len(all_benchmark_results),
        unit=" files",
    ):
        # parse the file_name
        file_name = benchmark_result.split(".")[0]
        file_name, n_games = file_name.split("_n_games=")

        # get the parameters
        n_games = int(n_games)
        parameters = file_name.split("_")
        order = int(parameters[-1])
        index = parameters[-2]
        setup = "_".join(parameters[:-2])

        # get the game name
        if "SOUM" not in setup:
            application_name = create_application_name(setup)
        elif "max_interaction_size=5" in setup:
            application_name = "SOUM (low)"
        else:
            application_name = "SOUM (high)"
        run_id = file_name

        # load the benchmark results
        path = os.path.join(BENCHMARK_RESULTS_DIR, benchmark_result)
        results_df, _ = load_benchmark_results(path)

        n_players = int(results_df["n_players"].unique()[0])

        # get the best approximator
        try:
            best_approximators: dict = _get_best_approximator(results_df)
        except Exception as e:
            message = f"Error occurred while getting the best approximator for {file_name}: {e}"
            logging.exception(message)
            continue
        for metric, metric_values in best_approximators.items():
            for approximator, budget in metric_values:
                results = {
                    "run_id": run_id,
                    "application_name": application_name,
                    "index": index,
                    "order": order,
                    "n_games": n_games,
                    "metric": metric,
                    "best_approximator": approximator,
                    "budget": budget,
                    "n_player": n_players,
                    "full_budget": budget >= 2**n_players,
                }
                all_results.append(results)

        if n_evals is not None and eval_i >= n_evals:
            break
    # create a dataframe from the results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(EVAL_DIR / "benchmark_results_summary.csv", index=False)

    return results_df


def plot_stacked_bar(df: pd.DataFrame, setting: str = "high", *, save: bool = False) -> None:
    """Summarizes the benchmark results by plotting a collection of stacked bar plots.

    For each metric, this function plots a stacked bar plot showing the percentage of the best
    approximator being the best for each application. Hence, on the x-axis we have the application
    names and on the y-axis the percentage of the best approximator being the best. For each
    application, we plot the results for the SV and k-SII indices at separate bars. Thus, there
    are 2 * n_applications bars in total in each plot (one plot for each metric).

    Args:
        df: The dataframe containing the benchmark results. The dataframe should have the following
            columns:
            - run_id
            - application_name
            - index
            - order
            - n_games
            - metric
            - best_approximator
            - budget
            - n_player
            - full_budget
        setting: The budget setting to use. Can be 'all', 'high', or 'low'.
        save: Whether to save the plot to a file.

    """
    if setting not in ["all", "high", "low"]:
        msg = "Budget setting must be 'all', 'high', or 'low'."
        raise ValueError(msg)

    import matplotlib.pyplot as plt

    from shapiq.benchmark.plot import STYLE_DICT  # maps approx names to colors and markers

    # get all unique applications and metrics and index
    all_applications = df["application_name"].unique()
    all_applications = sort_values(all_applications, APPLICATION_ORDERING)
    all_metrics = df["metric"].unique()
    all_indices = df["index"].unique()

    index_approximators = {}
    for index in all_indices:
        index_approximators[index] = sorted(df[df["index"] == index]["best_approximator"].unique())

    # iterate over all metrics
    for metric in all_metrics:
        metric_df = df[(df["metric"] == metric) & (~df["full_budget"])]
        if setting != "all":
            if setting == "low":
                max_budget_values_run_id = metric_df.groupby("run_id")["budget"].median()
            else:  # budget_setting == "high":
                max_budget_values_run_id = metric_df.groupby("run_id")["budget"].max()
            high_budget_dfs = []
            for run_id, max_budget in max_budget_values_run_id.items():
                high_budget_dfs.append(
                    metric_df[
                        (metric_df["run_id"] == run_id) & (metric_df["budget"] == max_budget)
                    ].copy(),
                )
            metric_df = pd.concat(high_budget_dfs)
        fig, ax = plt.subplots()
        width = 0.4
        padding = 0.15
        sep = 0.6
        x = list(range(len(all_applications)))
        x_ticks_index, x_tick_labels_index, x_ticks_app, x_ticks_labels_app = [], [], [], []
        for app_i, application in enumerate(all_applications):
            position = x[app_i]
            all_pos = []
            for index_i, index in enumerate(all_indices):
                n_values = len(
                    metric_df[
                        (metric_df["application_name"] == application)
                        & (metric_df["index"] == index)
                    ],
                )
                approximators = index_approximators[index]
                if index == "SV":
                    approximators_sorted = sort_values(approximators, SV_APPROXIMATORS_ORDERING)
                else:
                    approximators_sorted = sort_values(approximators, SI_APPROXIMATORS_ORDERING)
                start, height = 0, 0
                pos = position + index_i * (width + padding) + (sep * position)
                all_pos.append(pos)
                for approximator in approximators_sorted:
                    color = STYLE_DICT[approximator]["color"]
                    count = len(
                        metric_df[
                            (metric_df["application_name"] == application)
                            & (metric_df["best_approximator"] == approximator)
                            & (metric_df["index"] == index)
                        ],
                    )
                    percent_best = count / n_values
                    height = percent_best * 100
                    ax.bar(pos, height, width, bottom=start, color=color)
                    start += height
                x_ticks_index.append(pos)
                index_title = f"{index}"
                if index == "k-SII":
                    index_title = r"SI"
                x_tick_labels_index.append(index_title)
            pos_mean = sum(all_pos) / len(all_pos)
            x_ticks_app.append(pos_mean)
            x_ticks_labels_app.append(abbreviate_application_name(application, new_line=True))

        # add the x-ticks for the indices
        ax.set_xticks(x_ticks_index)
        ax.set_xticklabels(x_tick_labels_index)

        # add a second x-axis for the application names with the same limits
        ax2 = ax.twiny()
        ax2.set_xticks(x_ticks_app)
        ax2.set_xticklabels(x_ticks_labels_app)
        ax2.set_xlim(ax.get_xlim())

        # add a title for the axis
        ax2.set_xlabel("Application")
        ax.set_ylabel(f"Perc. of approximator being best: {metric}")

        # set y-axis to max 105
        ax.set_ylim(0, 105)

        plt.tight_layout()
        if save:
            save_path = EVAL_DIR / f"stacked_bar_{metric}.pdf"
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    create_eval = False
    budget_setting = "high"  # can be 'all', 'high', 'low'

    eval_path = EVAL_DIR / "benchmark_results_summary.csv"
    if create_eval or not os.path.exists(eval_path):
        eval_df = create_eval_csv(n_evals=None)
    else:
        eval_df = pd.read_csv(eval_path)

    # data frame has the following columns:
    # run_id, application_name, index, order, n_games, metric, best_approximator, budget, n_player, full_budget

    # print unique applications

    # for all metrics compute the percentage of the approximator being the best
    for _metric in eval_df["metric"].unique():
        _metric_df = eval_df[eval_df["metric"] == _metric]
        _best_approx = _metric_df["best_approximator"].value_counts(normalize=True)

    # plot the results
    plot_stacked_bar(eval_df, setting=budget_setting, save=True)
