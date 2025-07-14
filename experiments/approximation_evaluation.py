from shapiq import InteractionValues
import pandas as pd
import glob
from shapiq.benchmark.metrics import get_all_metrics


if __name__ == "__main__":
    GAME_IDS = ["adult_census", "california_housing", "bike_sharing","forest_fires","nhanesi","real_estate","communities_and_crime"]#,"breast_cancer"]
    MODELS = ["gradient_boosting"]#, "random_forest"]
    RANDOM_STATE = 40
    N_EXPLANATIONS = 10
    APPROXIMATORS = [
        "KernelSHAP",
        "LeverageSHAP",
        "PermutationSampling",
        "ShapleyGAX-2ADD",
        "ShapleyGAX-2ADD-Lev1",
        "ShapleyGAX-2ADD-Leverage2"
    ]

    approximation_metrics = pd.DataFrame()

    results = []

    for game in GAME_IDS:
        for model in MODELS:
            game_id = game + "_" + model
            for id_explain in range(N_EXPLANATIONS):
                save_path_ground_truth = "ground_truth/" + game_id + "_" + str(RANDOM_STATE) + "_" + str(id_explain) + "_exact_values.json"
                # load interaction values from save_path_ground_truth
                ground_truth = InteractionValues.load(save_path_ground_truth)
                # File pattern with wildcard for budget
                file_pattern = f"approximations/{game_id}_{RANDOM_STATE}_{id_explain}_*.json"
                # Get matching file paths
                file_paths = glob.glob(file_pattern)
                for file in file_paths:
                    approximator = file.split("_")[-2]
                    budget = int(file.split("_")[-1].replace(".json", ""))
                    result = {
                        "game": game,
                        "model": model,
                        "game_id": game_id,
                        "n_players": ground_truth.n_players,
                        "budget": budget,
                        "budget_relative": round(budget / (2 ** ground_truth.n_players), 6),
                        "approximator": approximator,
                        "used_budget": budget,
                        "iteration": 1
                    }
                    approximated_values = InteractionValues.load(file)
                    all_metrics = get_all_metrics(ground_truth, approximated_values)
                    result.update(all_metrics)
                    results.append(result)
    results_df = pd.DataFrame(results)

    results_df.to_csv("results_benchmark.csv")