from shapiq.games.benchmark.local_xai import AdultCensus, CaliforniaHousing, BikeSharing
from shapiq.games.benchmark.local_xai.benchmark_tabular import ForestFires, RealEstate, BreastCancer, NHANESI, WineQuality, CommunitiesAndCrime

from shapiq.games.benchmark.treeshapiq_xai import TreeSHAPIQXAI

from shapiq.explainer.tree import TreeSHAPIQ

from shapiq import TreeExplainer
import numpy as np

from shapiq import ExactComputer

from shapiq import KernelSHAP, PermutationSamplingSV
from shapiq.approximator.regression.shapleygax import ShapleyGAX, ExplanationBasisGenerator

from shapiq.utils.empirical_leverage_scores import get_leverage_scores

from scipy.special import binom

import multiprocessing as mp
import tqdm


def get_approximators(n_players):
    # Create the approximators for the game



    # get leverage scores for order 1 and 2
    leverage_weights_1 = np.ones(n_players + 1)
    lev_scores_2 = get_leverage_scores(n_players, 2)
    leverage_weights_2 = np.zeros(n_players + 1)
    for size, score in lev_scores_2.items():
        leverage_weights_2[size] = binom(n_players, size) * score


    approximators = []

    explanation_basis = ExplanationBasisGenerator(N=set(range(n_players)))
    kadd = explanation_basis.generate_kadd_explanation_basis(max_order=2)

    if "KernelSHAP" in APPROXIMATORS:
        # KernelSHAP
        kernel_shap = KernelSHAP(n=n_players, random_state=RANDOM_STATE)
        kernel_shap.name = "KernelSHAP"
        approximators.append(kernel_shap)
    if "LeverageSHAP" in APPROXIMATORS:
        # LeverageSHAP
        leverage_shap = KernelSHAP(n=n_players, random_state=RANDOM_STATE, sampling_weights=leverage_weights_1)
        leverage_shap.name = "LeverageSHAP"
        approximators.append(leverage_shap)
    if "PermutationSampling" in APPROXIMATORS:
        # Permutation Sampling
        permutation_sampling = PermutationSamplingSV(n=n_players, random_state=RANDOM_STATE)
        permutation_sampling.name = "PermutationSampling"
        approximators.append(permutation_sampling)
    if "ShapleyGAX-2ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        shapley_gax_kadd = ShapleyGAX(n=n_players, explanation_basis=kadd, random_state=RANDOM_STATE)
        shapley_gax_kadd.name = "ShapleyGAX-2ADD"
        approximators.append(shapley_gax_kadd)
    if "ShapleyGAX-2ADD-Lev1" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 1
        shapley_gax_kadd_lev1 = ShapleyGAX(n=n_players, explanation_basis=kadd, random_state=RANDOM_STATE,
                                      sampling_weights=leverage_weights_1)
        shapley_gax_kadd_lev1.name = "ShapleyGAX-2ADD-Lev1"
        approximators.append(shapley_gax_kadd_lev1)
    if "ShapleyGAX-2ADD-Lev2" in APPROXIMATORS:
        # ShapleyGAX with leverage weights for order 2
        shapley_gax_kadd_lev2 = ShapleyGAX(n=n_players, explanation_basis=kadd, random_state=RANDOM_STATE,
                                           sampling_weights=leverage_weights_2)
        shapley_gax_kadd_lev2.name = "ShapleyGAX-2ADD-Leverage2"
        approximators.append(shapley_gax_kadd_lev2)

    return approximators



if __name__ == "__main__":
    N_EXPLANATIONS = 10
    RANDOM_STATE = 40
    RUN_GROUND_TRUTH = False
    RUN_APPROXIMATION = True

    # run the benchmark for the games
    GAMES = [
        #AdultCensus(model_name="random_forest", imputer="baseline",random_state=RANDOM_STATE),
        #AdultCensus(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #CaliforniaHousing(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        CaliforniaHousing(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #BikeSharing(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        #BikeSharing(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #ForestFires(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        #ForestFires(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #RealEstate(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        #RealEstate(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #NHANESI(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        #NHANESI(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #BreastCancer(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        #BreastCancer(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #WineQuality(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        #WineQuality(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #CommunitiesAndCrime(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        #CommunitiesAndCrime(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        #SentimentAnalysis(),
        #ImageClassifier()
    ]

    if RUN_GROUND_TRUTH:
    # Compute the ground truth values for the games
        for game in GAMES:
            game_id = game.setup.dataset_name + "_" + game.setup.model_name
            print(game.setup.dataset_name, game.setup.model_name)
            game.setup.print_train_performance()
            for id_explain in range(N_EXPLANATIONS):
                x_explain = game.setup.x_test[id_explain, :]
                tree_game = TreeSHAPIQXAI(x_explain, game.setup.model, verbose=False)
                save_path = "ground_truth/" + game_id + "_" + str(RANDOM_STATE) + "_" + str(id_explain) + "_exact_values.json"
                shap_ground_truth = tree_game.exact_values(index="SV",order=1)
                shap_ground_truth.save(save_path)
                print(f"Exact: {shap_ground_truth.values} saved to {save_path}")


    APPROXIMATORS = ["PermutationSampling", "KernelSHAP", "LeverageSHAP", "ShapleyGAX-2ADD", "ShapleyGAX-2ADD-Lev1", "ShapleyGAX-2ADD-Lev2"]

    MAX_BUDGET = 20000
    N_BUDGET_STEPS = 10

    def explain_instance(args):
        game_id, id_explain = args
        tree_game = TREE_GAMES[id_explain]
        approximators = get_approximators(tree_game.n_players)
        budget_range = np.linspace(min(500,2**game.n_players/10), min(2 ** game.n_players, MAX_BUDGET), N_BUDGET_STEPS).astype(int)
        for approximator in approximators:
            print("Computing approximations for", approximator.name, "on game", game_id, "explanation id", id_explain)
            for budget in budget_range:
                shap_approx = approximator.approximate(budget=budget, game=tree_game)
                save_path = "approximations/"+game_id+"_"+str(RANDOM_STATE)+"_"+str(id_explain)+"_"+approximator.name+"_"+str(budget)+".json"
                shap_approx.save(save_path)

    if RUN_APPROXIMATION:
        N_JOBS = 5
        for game in GAMES:
            game_id = game.setup.dataset_name + "_" + game.setup.model_name
            TREE_GAMES = []
            for id_explain in range(N_EXPLANATIONS):
                x_explain = game.setup.x_test[id_explain, :]
                tree_game = TreeSHAPIQXAI(x_explain, game.setup.model, verbose=False)
                TREE_GAMES.append(tree_game)
            print("Tree games initialized for", game.setup.dataset_name, game.setup.model_name)
            args_list = [(game_id,id_explain) for id_explain in range(N_EXPLANATIONS)]
            with mp.Pool() as pool:
                pool.map(explain_instance, args_list)

        """
        # Approximate the values for the games
        for game in GAMES:
            approximators = get_approximators(game)
            print("Approximation for ",game.setup.dataset_name, game.setup.model_name)
            game.setup.print_train_performance()
            budget_range = np.linspace(500, min(2**game.n_players, MAX_BUDGET), N_BUDGET_STEPS).astype(int)
            for id_explain in tqdm.tqdm(range(N_EXPLANATIONS), desc=f"Approximating {game.setup.dataset_name} {game.setup.model_name}"):
                x_explain = game.setup.x_test[id_explain, :]
                tree_game = TreeSHAPIQXAI(x_explain, game.setup.model, verbose=False)
                for approximator in approximators:
                    for budget in budget_range:
                        shap_approx = approximator.approximate(budget=budget, game=tree_game)
                        save_path = f"approximations/{game.setup.dataset_name}_{game.setup.model_name}_{RANDOM_STATE}_{id_explain}_{approximator.name}_{budget}.json"
                        shap_approx.save(save_path)
        """