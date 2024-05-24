"""This module contains the configuration of all benchmark games and how they are set up."""

from . import BikeSharingLocalXAI

# stores the configurations of all the benchmark games and how they are set up
BENCHMARK_CONFIGURATIONS: dict[str, list[dict]] = {
    # TODO: add configurations for all games
    # TODO: fix that the names for the games are not hardcoded but also unique for the game
    BikeSharingLocalXAI.__name__: [
        {
            "x": None,  # TODO: think about how to fix this that this is the _n_ of the game
            "model_name": "decision_tree",
            "normalize": True,
            "verbose": True,
            "random_state": 42,
        }
    ],
}
