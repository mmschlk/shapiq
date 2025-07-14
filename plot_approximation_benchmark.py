# plot the results
from shapiq.benchmark import plot_approximation_quality
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

if __name__ == "__main__":
    GAMES_IDENTIFIER = ["AdultCensusLocalXAI"] #["SentimentAnalysisLocalXAI","ImageClassifierLocalXAI","AdultCensusLocalXAI"]
    DATA_DIR = "data"
    METRICS_TO_PLOT = ["Precision@10", "MSE", "KendallTau"]

    for game_identifier in GAMES_IDENTIFIER:
        print(f"Looking for result files for {game_identifier}...")

        # Build a regex to match the desired filename pattern
        pattern = re.compile(rf"{re.escape(game_identifier)}_(.+?)_results\.json$")

        n_player_ids = []

        # Scan the directory
        for filename in os.listdir(DATA_DIR):
            match = pattern.match(filename)
            if match:
                n_player_id = match.group(1)
                n_player_ids.append(n_player_id)

        print(f"Found n_player_ids for {game_identifier}: {n_player_ids}")

        for n_player_id in n_player_ids:
            print(f"Plotting benchmark results for {game_identifier}...")
            # load the game files from disk / or download
            save_path = "data/" + game_identifier + "_" + n_player_id + "_results.json"

            results = pd.read_json(save_path)

            for metric in METRICS_TO_PLOT:
                fig, ax = plot_approximation_quality(results, metric=metric,log_scale_y=True)                # set title for the plot
                ax.set_title(f"{game_identifier} with configuration {n_player_id}")
                plt.tight_layout()
                fig.savefig("experiments/plots/" + game_identifier +  "_" + n_player_id + "_"+metric+".png")
                fig.show()
