import pandas as pd
from shapiq.benchmark import plot_approximation_quality

if __name__ == "__main__":
    # Load the results from the CSV file
    results_df = pd.read_csv("results_benchmark.csv")

    GAME_IDS = results_df["game_id"].unique()
    metric = "MSE"
    for game_id in GAME_IDS:
        results_game_id = results_df[results_df["game_id"] == game_id]
        fig, ax = plot_approximation_quality(data=results_game_id, metric=metric, log_scale_y = True)
        ax.set_title(f"{metric} for {game_id}")
        fig.savefig(f"plots/{game_id}_{metric}.png")
        fig.show()