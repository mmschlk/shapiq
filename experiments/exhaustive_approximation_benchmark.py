from shapiq.games.benchmark.treeshap_interventional_xai import TreeSHAPInterventionalXAI
from shapiq import TreeExplainer, InteractionValues
import numpy as np
import multiprocessing as mp
from init_approximator import get_approximators

from shapiq.games.benchmark.local_xai.benchmark_language import SentimentAnalysis
from shapiq.games.benchmark.local_xai.benchmark_image import ImageClassifier


from shapiq.benchmark import load_games_from_configuration


if __name__ == "__main__":
    # this code runs interventional treeshap from the shap package for ground truth and uses the TreeSHAPInterventionalXAI class
    ID_EXPLANATIONS = range(10) #range(10,30) # ids of test instances to explain, can be used to compute new ids
    RANDOM_STATE = 40 # random state for the games
    # ID_CONFIG_APPROXIMATORS = 40, PAIRING=False, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 39, PAIRING_False, REPLACEMENT=False
    # ID_CONFIG_APPROXIMATORS = 38, PAIRING=True, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 37, PAIRING=True, REPLACEMENT=False
    ID_CONFIG_APPROXIMATORS = 37 # used for different approximator configurations

    if ID_CONFIG_APPROXIMATORS == 40:
        REPLACEMENT = True
        PAIRING = False
    if ID_CONFIG_APPROXIMATORS == 39:
        REPLACEMENT = False
        PAIRING = False
    if ID_CONFIG_APPROXIMATORS == 38:
        REPLACEMENT = True
        PAIRING = True
    if ID_CONFIG_APPROXIMATORS == 37:
        REPLACEMENT = False
        PAIRING = True

    RUN_GROUND_TRUTH = False
    RUN_APPROXIMATION = True



    GAME_IDENTIFIERS = ["SentimentIMDBDistilBERT14","ResNet18w14Superpixel","ViT3by3Patches","ViT4by4Patches"]#, "ImageClassifierLocalXAI"]
    # game_identifier = "SOUM"
    config_id = 1
    n_games = 10

    GAMES = {}

    for game_identifier in GAME_IDENTIFIERS:
        if game_identifier == "SentimentIMDBDistilBERT14":
            sentiment_analysis = load_games_from_configuration(
                game_class="SentimentAnalysisLocalXAI", n_player_id=0, config_id=config_id, n_games=n_games
            )
            GAMES[game_identifier] = sentiment_analysis
        if game_identifier == "ResNet18w14Superpixel":
            image_classifier = load_games_from_configuration(
                game_class="ImageClassifierLocalXAI", n_player_id=0, config_id=config_id, n_games=n_games
            )
            GAMES[game_identifier] = image_classifier
        if game_identifier == "ViT3by3Patches":
            image_classifier = load_games_from_configuration(
                game_class="ImageClassifierLocalXAI", n_player_id=1, config_id=config_id, n_games=n_games
            )
        if game_identifier == "ViT4by4Patches":
            image_classifier = load_games_from_configuration(
                game_class="ImageClassifierLocalXAI", n_player_id=2, config_id=config_id, n_games=n_games
            )


    if RUN_GROUND_TRUTH:
    # Compute the ground truth values for the games
        for game_identifier, game in GAMES.items():
            for id_explain,game_instance in enumerate(game):
                game_id = game_identifier + "_" + str(config_id)
                save_path = "ground_truth/exhaustive/" + game_id + "_" + str(RANDOM_STATE) + "_" + str(id_explain) + "_exact_values.json"
                ground_truth = game_instance.exact_values(index="SV", order=1)
                ground_truth.save(save_path)
                print(f"Exact: {ground_truth} saved to {save_path}")


    APPROXIMATORS = ["ShapleyGAX-3ADD","ShapleyGAX-3ADD-Lev1"] #["PermutationSampling", "KernelSHAP", "LeverageSHAP", "ShapleyGAX-2ADD", "ShapleyGAX-2ADD-Lev1", "ShapleyGAX-3ADD","ShapleyGAX-3ADD-Lev1"]

    MAX_BUDGET = 20000
    N_BUDGET_STEPS = 10

    def explain_instance(args):
        game_id, id_explain, game_instance = args
        approximators = get_approximators(APPROXIMATORS, game_instance.n_players, RANDOM_STATE, PAIRING, REPLACEMENT)
        #budget_range = np.linspace(min(500,2**game_instance.n_players/10), min(2 ** game_instance.n_players, MAX_BUDGET), N_BUDGET_STEPS).astype(int)
        min_budget = min(50, 2 ** game_instance.n_players / 10)
        max_budget = min(2 ** game_instance.n_players, MAX_BUDGET)
        budget_range = np.logspace(np.log10(min_budget), np.log10(max_budget), N_BUDGET_STEPS).astype(int)
        for approximator in approximators:
            print("Computing approximations for", approximator.name, "on game", game_id, "explanation id", id_explain)
            for budget in budget_range:
                shap_approx = approximator.approximate(budget=budget, game=game_instance)
                save_path = "approximations/exhaustive/" + game_id + "_" + str(ID_CONFIG_APPROXIMATORS) + "_" + str(
                    id_explain) + "_" + approximator.name + "_" + str(budget) + ".json"
                shap_approx.save(save_path)


    if RUN_APPROXIMATION:
        N_JOBS = 5
        # Compute the ground truth values for the games
        for game_identifier, game in GAMES.items():
            game_id = game_identifier + "_" + str(config_id)
            args_list = [(game_id,id_explain,game_instance) for id_explain, game_instance in enumerate(game)]
            with mp.Pool() as pool:
                pool.map(explain_instance, args_list)
