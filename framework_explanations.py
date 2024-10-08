"""This module contains functionality for computing various feature-based explanations."""

from shapiq import ExactComputer, Game, InteractionValues, powerset


def _compute_pure(
    moebius_values: InteractionValues, feature_set: tuple[int, ...], joint: bool = False
) -> float:
    """Computes the pure feature influence from the Moebius values.

    Args:
        moebius_values: The Moebius values.

    Returns:
        The pure feature influence up to order
    """
    pure_value = moebius_values[feature_set]
    if not joint:
        return pure_value
    for feature_subset in powerset(feature_set, max_size=len(feature_set) - 1):
        pure_value += moebius_values[feature_subset]
    return pure_value


def compute_explanation(
    game: Game,
    feature_influence: str = "distributed",
    measurement: str = "individual",
) -> InteractionValues:
    """Computes the feature-based explanation for a given game.

    Args:
        game: The game for which the explanation should be computed.
        feature_influence: The type higher-order interaction influence to compute the explanation
            with. Either `pure`, `distributed`, or `full`. Default is `distributed`.
        measurement: The feature effect to measure. Either `individual`, `joint`, or `interaction`.
            Default is `individual`.

    Returns:
        The feature-based explanation.
    """
    assert feature_influence in ["pure", "distributed", "full"]
    assert measurement in ["individual", "joint", "interaction"]

    computer = ExactComputer(game_fun=game, n_players=game.n_players)
    moebius_values = computer(index="Moebius", order=game.n_players)
    print(moebius_values)
