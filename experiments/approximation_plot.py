from __future__ import annotations

import pandas as pd

from shapiq.benchmark import plot_approximation_quality

if __name__ == "__main__":
    # Load the results from the CSV file
    results_df = pd.read_csv("results_benchmark.csv")
    results_df = results_df.sort_values(by="n_players")

    results_df = results_df[
        # (results_df["approximator"] == "ShapleyGAX-4ADD") |
        # (results_df["approximator"] == "ShapleyGAX-5ADD") |
        # (results_df["approximator"] == "ShapleyGAX-2ADD-10%") |
        # (results_df["approximator"] == "ShapleyGAX-2ADD-20%") |
        # (results_df["approximator"] == "ShapleyGAX-2ADD-50%") |
        # (results_df["approximator"] == "ShapleyGAX-6ADD") |
        # (results_df["approximator"] == "ShapleyGAX-7ADD") |
        # (results_df["approximator"] == "ShapleyGAX-3ADD")
        # (results_df["approximator"] == "ShapleyGAX-1SYM-Lev1")
        # | (results_df["approximator"] == "ShapleyGAX-2SYM-Lev1")
        # (results_df["approximator"] == "ShapleyGAX-3ADD-Lev1")
        # (results_df["approximator"] == "ShapleyGAX-2ADD-P10%")
        # | (results_df["approximator"] == "ShapleyGAX-2ADD-P20%")
        # | (results_df["approximator"] == "ShapleyGAX-2ADD-P50%")
        # | (results_df["approximator"] == "ShapleyGAX-2ADD-P100%")
        # (results_df["approximator"] == "ShapleyGAX-2ADD-P10%")
        # | (results_df["approximator"] == "ShapleyGAX-2ADD-P20%")
        # | (results_df["approximator"] == "ShapleyGAX-2ADD-P50%")
        # (results_df["approximator"] == "ShapleyGAX-2ADD-P100%")
        # (results_df["approximator"] == "ShapleyGAX-3ADDWO2-P10%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADDWO2-P20%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADDWO2-P50%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADDWO2-P100%")
        (results_df["approximator"] == "ShapleyGAX-3ADD-P10%")
        | (results_df["approximator"] == "ShapleyGAX-3ADD-P20%")
        | (results_df["approximator"] == "ShapleyGAX-3ADD-P50%")
        | (results_df["approximator"] == "ShapleyGAX-3ADD-P100%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADD-10%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADD-20%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADD-50%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADD-WO2")
        # | (results_df["approximator"] == "ShapleyGAX-3ADDWO2-10%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADDWO2-20%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADDWO2-50%")
        # | (results_df["approximator"] == "KernelSHAP")
        | (results_df["approximator"] == "LeverageSHAP")
        # | (results_df["approximator"] == "ShapleyGAX-2ADD-Lev1")
        | (results_df["approximator"] == "PermutationSampling")
        # | (results_df["approximator"] == "ShapleyGAX-4ADD")
        | (results_df["approximator"] == "ShapleyGAX-4ADD-Lev1")
        # | (results_df["approximator"] == "ShapleyGAX-3ADD-P10%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADD-P20%")
        # | (results_df["approximator"] == "ShapleyGAX-3ADD-P50%")
    ]

    GAME_IDS = results_df["game_id"].unique()
    GAME_TYPES = results_df["game_type"].unique()
    metric = "MSE"

    config_id = None
    config_id = [37]

    if config_id is not None:
        results_df = results_df[results_df["id_config_approximator"].isin(config_id)]

    for game_type in GAME_TYPES:
        results_game_type = results_df[results_df["game_type"] == game_type]
        for game_id in GAME_IDS:
            results_game_id = results_game_type[results_game_type["game_id"] == game_id]
            if len(results_game_id) > 0:
                fig, ax = plot_approximation_quality(
                    data=results_game_id, metric=metric, log_scale_y=True
                )
                ax.set_title(f"{metric} for {game_type}, {game_id}")
                fig.savefig(f"plots/{game_type}/{game_id}_{metric}.png")
                # fig.show()
