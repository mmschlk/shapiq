from shapiq import InteractionValues
import pandas as pd
import glob
from shapiq.benchmark.metrics import get_all_metrics

import tqdm

if __name__ == "__main__":
    GAME_IDS = ["SentimentAnalysisLocalXAI",
                "adult_census",
                "california_housing",
                "bike_sharing",
                "forest_fires",
                "nhanesi",
                "real_estate",
                "communities_and_crime",
                "breast_cancer",
                "independentlinear60",
                "corrgroups60"
                ]
    MODELS = ["gradient_boosting", "random_forest","1"]
    RANDOM_STATE = 40
    ID_EXPLANATIONS = range(10) # ids of test instances to explain
    APPROXIMATORS = [
        #"SPEX",
        "KernelSHAP",
        "LeverageSHAP",
        "PermutationSampling",
        "ShapleyGAX-2ADD",
        "ShapleyGAX-2ADD-Lev1",
        "ShapleyGAX-3ADD",
        "ShapleyGAX-3ADD-Lev1"
    ]
    GAME_TYPES = ["baseline","interventional","pathdependent"]

    approximation_metrics = pd.DataFrame()

    results = []

    for game_type in GAME_TYPES:
        print(f"Loading {game_type}")
        for game in GAME_IDS:
            print(f"Loading {game}")
            for model in MODELS:
                game_id = game + "_" + model
                for id_explain in ID_EXPLANATIONS:
                    save_path_ground_truth = "ground_truth/"+game_type+"/" + game_id + "_" + str(RANDOM_STATE) + "_" + str(id_explain) + "_exact_values.json"

                    try:
                        # load interaction values from save_path_ground_truth
                        ground_truth = InteractionValues.load(save_path_ground_truth)
                    except Exception as e:
                        print(f"Error loading ground truth for {game_type}/{game_id}")
                        continue
                    # File pattern with wildcard for budget
                    file_pattern = f"approximations/{game_type}/{game_id}_*_{id_explain}_*.json"
                    # Get matching file paths
                    file_paths = glob.glob(file_pattern)
                    for file in file_paths:
                        id_config_approximator = file.split("_")[-4]
                        approximator = file.split("_")[-2]
                        budget = int(file.split("_")[-1].replace(".json", ""))
                        result = {
                            "game_type": game_type,
                            "game": game,
                            "model": model,
                            "game_id": game_id,
                            "id_explain": id_explain,
                            "n_players": ground_truth.n_players,
                            "budget": budget,
                            "budget_relative": round(budget / (2 ** ground_truth.n_players), 6),
                            "approximator": approximator,
                            "used_budget": budget,
                            "iteration": 1,
                            "id_config_approximator": id_config_approximator,
                        }
                        approximated_values = InteractionValues.load(file)
                        assert(len(approximated_values.values)-1==ground_truth.n_players)
                        all_metrics = get_all_metrics(ground_truth, approximated_values)
                        result.update(all_metrics)
                        results.append(result)
    results_df = pd.DataFrame(results)

    results_df.to_csv("results_benchmark.csv")