import pandas as pd
from shapiq.benchmark import plot_approximation_quality

if __name__ == "__main__":
    # Load the results from the CSV file
    results_df = pd.read_csv("results_benchmark.csv")
    results_df = results_df.sort_values(by="n_players")

    results_df = results_df[(results_df["approximator"] == "ShapleyGAX-3ADD-Lev1") | (results_df["approximator"] == "ShapleyGAX-3ADD") | (results_df["approximator"] == "KernelSHAP") |  (results_df["approximator"] == "LeverageSHAP") |  (results_df["approximator"] == "PermutationSampling")]

    GAME_IDS = results_df["game_id"].unique()
    GAME_TYPES = results_df["game_type"].unique()
    metric = "MSE"

    config_id = 37

    if config_id is not None:
        results_df = results_df[results_df["id_config_approximator"] == config_id]



    for game_type in GAME_TYPES:
        results_game_type = results_df[results_df["game_type"] == game_type]
        for game_id in GAME_IDS:
            results_game_id = results_game_type[results_game_type["game_id"] == game_id]
            if len(results_game_id) >0:
                fig, ax = plot_approximation_quality(data=results_game_id, metric=metric, log_scale_y = True)
                ax.set_title(f"{metric} for {game_type}, {game_id}")
                fig.savefig(f"plots/{game_type}/{game_id}_{metric}.png")
                fig.show()