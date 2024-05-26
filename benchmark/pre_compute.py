"""This script pre-computes the games provided the benchmark configurations for certain parameters.
"""

import argparse
import sys
from pathlib import Path

# add shapiq to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


if __name__ == "__main__":

    from shapiq.games.benchmark import (
        AdultCensusClusterExplanation,  # 2 configs
        AdultCensusDatasetValuation,  # 3 configs
        AdultCensusEnsembleSelection,  # 1 config
        AdultCensusFeatureSelection,  # 3 configs
        AdultCensusGlobalXAI,  # 3 configs
        AdultCensusLocalXAI,  # 3 configs
        AdultCensusRandomForestEnsembleSelection,  # 1 config
        AdultCensusUnsupervisedData,  # 1 config
        BikeSharingClusterExplanation,  # 2 configs
        BikeSharingDatasetValuation,  # 3 configs
        BikeSharingEnsembleSelection,  # 1 config
        BikeSharingFeatureSelection,  # 3 configs
        BikeSharingGlobalXAI,  # 3 configs
        BikeSharingLocalXAI,  # 3 configs
        BikeSharingRandomForestEnsembleSelection,  # 1 config
        BikeSharingUnsupervisedData,  # 1 config
        CaliforniaHousingClusterExplanation,  # 2 configs
        CaliforniaHousingDatasetValuation,  # 3 configs
        CaliforniaHousingEnsembleSelection,  # 1 config
        CaliforniaHousingFeatureSelection,  # 3 configs
        CaliforniaHousingGlobalXAI,  # 4 configs  (neural network not in the others)
        CaliforniaHousingLocalXAI,  # 4 configs  (neural network not in the others)
        CaliforniaHousingRandomForestEnsembleSelection,  # 1 config
        CaliforniaHousingUnsupervisedData,  # 1 config
    )
    from shapiq.games.benchmark.benchmark_config import BENCHMARK_CONFIGURATIONS
    from shapiq.games.benchmark.precompute import pre_compute_from_configuration

    game_name_to_class = {
        "AdultCensusClusterExplanation": AdultCensusClusterExplanation,
        "AdultCensusDatasetValuation": AdultCensusDatasetValuation,
        "AdultCensusEnsembleSelection": AdultCensusEnsembleSelection,
        "AdultCensusFeatureSelection": AdultCensusFeatureSelection,
        "AdultCensusGlobalXAI": AdultCensusGlobalXAI,
        "AdultCensusLocalXAI": AdultCensusLocalXAI,
        "AdultCensusRandomForestEnsembleSelection": AdultCensusRandomForestEnsembleSelection,
        "AdultCensusUnsupervisedData": AdultCensusUnsupervisedData,
        "BikeSharingClusterExplanation": BikeSharingClusterExplanation,
        "BikeSharingDatasetValuation": BikeSharingDatasetValuation,
        "BikeSharingEnsembleSelection": BikeSharingEnsembleSelection,
        "BikeSharingFeatureSelection": BikeSharingFeatureSelection,
        "BikeSharingGlobalXAI": BikeSharingGlobalXAI,
        "BikeSharingLocalXAI": BikeSharingLocalXAI,
        "BikeSharingRandomForestEnsembleSelection": BikeSharingRandomForestEnsembleSelection,
        "BikeSharingUnsupervisedData": BikeSharingUnsupervisedData,
        "CaliforniaHousingClusterExplanation": CaliforniaHousingClusterExplanation,
        "CaliforniaHousingDatasetValuation": CaliforniaHousingDatasetValuation,
        "CaliforniaHousingEnsembleSelection": CaliforniaHousingEnsembleSelection,
        "CaliforniaHousingFeatureSelection": CaliforniaHousingFeatureSelection,
        "CaliforniaHousingGlobalXAI": CaliforniaHousingGlobalXAI,
        "CaliforniaHousingLocalXAI": CaliforniaHousingLocalXAI,
        "CaliforniaHousingRandomForestEnsembleSelection": CaliforniaHousingRandomForestEnsembleSelection,
        "CaliforniaHousingUnsupervisedData": CaliforniaHousingUnsupervisedData,
    }

    # example python run commands for the different games
    # python pre_compute.py --game AdultCensusClusterExplanation --config_id 1
    # python pre_compute.py --game AdultCensusClusterExplanation --config_id 2
    # python pre_compute.py --game CaliforniaHousingDatasetValuation --config_id 1
    # python pre_compute.py --game CaliforniaHousingDatasetValuation --config_id 2 & python pre_compute.py --game CaliforniaHousingDatasetValuation --config_id 3

    # parse arguments
    parser = argparse.ArgumentParser()
    game_choices = list(game_name_to_class.keys())
    parser.add_argument("--game", type=str, required=True, choices=game_choices)
    parser.add_argument("--config_id", type=int, required=True, help="The configuration ID to use.")
    parser.add_argument("--n_jobs", type=int, required=False, default=1)
    args = parser.parse_args()

    # get the game class
    game_class = game_name_to_class[args.game]

    # get the configuration
    all_game_configs = BENCHMARK_CONFIGURATIONS[game_class]["configurations"]
    n_configs = len(all_game_configs)
    if args.config_id < 1 or args.config_id > n_configs:
        raise ValueError(
            f"Invalid configuration ID. Must be in [1, {n_configs}] for game {args.game} which has "
            f"{all_game_configs} configurations."
        )

    game_config = all_game_configs[args.config_id - 1]

    # run the pre-computation
    pre_compute_from_configuration(game_class, configuration=game_config, n_jobs=args.n_jobs)
