"""This module contains functionality for computing various feature-based explanations."""

from typing import Callable, Optional, Union

from shapiq import ExactComputer, Game, InteractionValues, powerset


def _sii_weights(moebius_size: int, interaction_size: int) -> float:
    return 1 / (moebius_size - interaction_size + 1)


def _compute_pure(
    moebius_values: InteractionValues, feature_set: tuple[int, ...], joint: bool = False
) -> float:
    """Computes the pure feature influence from the Moebius values.

    Args:
        moebius_values: The Moebius values.

    Returns:
        The pure feature influence of the given feature set.
    """
    pure_value = moebius_values[feature_set]
    if not joint:
        return pure_value
    for feature_subset in powerset(feature_set, max_size=len(feature_set) - 1):
        pure_value += moebius_values[feature_subset]
    return pure_value


def _compute_partial(
    moebius_values: InteractionValues,
    feature_set: tuple[int, ...],
    player_set: tuple[int, ...],
    joint: bool = False,
    weights: Optional[Callable[[int, int], float]] = None,
) -> float:
    """Computes the pure feature influence from the Moebius values.

    Args:
        moebius_values: The Moebius values.

    Returns:
        The pure feature influence of the given feature set.
    """
    if weights is None:
        weights = _sii_weights
    partial_value = moebius_values[feature_set]
    feature_set = set(feature_set)
    for subset in powerset(player_set, min_size=1):
        if not joint and feature_set.issubset(subset) and feature_set != subset:
            weight = weights(len(subset), len(feature_set))
            partial_value += moebius_values[subset] * weight
        elif joint and feature_set.intersection(subset) not in [set(), feature_set]:
            weight = weights(len(subset), len(feature_set))
            partial_value += moebius_values[subset] * weight
    return partial_value


def _compute_full(
    moebius_values: InteractionValues,
    feature_set: tuple[int, ...],
    player_set: tuple[int, ...],
    joint: bool = False,
) -> float:
    """Computes the full feature influence from the Moebius values.

    Args:
        moebius_values: The Moebius values.

    Returns:
        The full feature influence of the given feature set.
    """
    full_value = moebius_values[feature_set]
    feature_set = set(feature_set)
    for subset in powerset(player_set, min_size=1):
        if not joint and feature_set.issubset(subset) and feature_set != subset:
            full_value += moebius_values[subset]
        elif joint and feature_set.intersection(subset) not in [set(), feature_set]:
            full_value += moebius_values[subset]
    return full_value


def compute_explanation_with_mi(
    game: Game,
    feature_sets: Union[list[tuple[int, ...]], tuple[int, ...]],
    feature_influence: str = "partial",
    entity: str = "individual",
) -> dict[tuple[int, ...], float]:
    """Computes the feature-based explanation for a given game.

    Args:
        game: The game for which the explanation should be computed.
        feature_sets: The feature sets for which the explanation should be computed.
        feature_influence: The type higher-order interaction influence to compute the explanation
            with. Either `pure`, `distributed`, or `full`. Default is `distributed`.
        entity: The feature effect to measure. Either `individual`, `joint`, or `interaction`.
            Default is `individual`.

    Returns:
        The feature-based explanation.
    """
    assert feature_influence in ["pure", "partial", "full"]
    assert entity in ["individual", "joint", "interaction"]

    if isinstance(feature_sets, tuple):
        feature_sets = [feature_sets]

    computer = ExactComputer(game_fun=game, n_players=game.n_players)
    moebius_values = computer(index="Moebius", order=game.n_players)
    player_set = tuple(range(game.n_players))

    explanation = {}
    for feature_set in feature_sets:
        if feature_influence == "pure":
            explanation[feature_set] = _compute_pure(
                moebius_values=moebius_values, feature_set=feature_set, joint=entity == "joint"
            )
        elif feature_influence == "partial":
            explanation[feature_set] = _compute_partial(
                moebius_values=moebius_values,
                feature_set=feature_set,
                joint=entity == "joint",
                player_set=player_set,
            )
        elif feature_influence == "full":
            explanation[feature_set] = _compute_full(
                moebius_values=moebius_values,
                feature_set=feature_set,
                joint=entity == "joint",
                player_set=player_set,
            )
    return explanation


def compute_explanation(
    game: Game,
    feature_sets: Union[list[tuple[int, ...]], tuple[int, ...]],
    feature_influence: str = "partial",
    entity: str = "individual",
) -> dict[tuple[int, ...], float]:
    assert feature_influence in ["pure", "partial", "full"]
    assert entity in ["individual", "joint", "interaction"]

    if isinstance(feature_sets, tuple):
        feature_sets = [feature_sets]

    computer = ExactComputer(game_fun=game, n_players=game.n_players)

    explanation = {}
    # entity = individual
    if entity == "individual" and feature_influence == "pure":
        for feature_set in feature_sets:
            explanation[feature_set] = game[feature_set] - game.empty_coalition_value
    elif entity == "individual" and feature_influence == "partial":
        sv_values = computer(index="SV", order=1)
        for feature_set in feature_sets:
            explanation[feature_set] = sv_values[feature_set]
    elif entity == "individual" and feature_influence == "full":
        for feature_set in feature_sets:
            explanation[feature_set] = game.grand_coalition_value - game[feature_set]

    # entity = joint
    elif entity == "joint" and feature_influence == "pure":
        for feature_set in feature_sets:
            explanation[feature_set] = game[feature_set] - game.empty_coalition_value
    elif entity == "joint" and feature_influence == "partial":
        joint_sv_values = computer(index="SGV", order=game.n_players)
        for feature_set in feature_sets:
            explanation[feature_set] = joint_sv_values[feature_set]
    elif entity == "joint" and feature_influence == "full":
        for feature_set in feature_sets:
            explanation[feature_set] = game.grand_coalition_value - game[feature_set]

    # entity = interaction
    elif entity == "interaction" and feature_influence == "pure":
        mi_values = computer(index="Moebius", order=game.n_players)
        for feature_set in feature_sets:
            explanation[feature_set] = mi_values[feature_set]
    elif entity == "interaction" and feature_influence == "partial":
        sii_values = computer(index="SII", order=game.n_players)
        for feature_set in feature_sets:
            explanation[feature_set] = sii_values[feature_set]
    elif entity == "interaction" and feature_influence == "full":
        co_mi_values = computer(index="Co-Moebius", order=game.n_players)
        for feature_set in feature_sets:
            explanation[feature_set] = co_mi_values[feature_set]

    return explanation
