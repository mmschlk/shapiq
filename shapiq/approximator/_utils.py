from shapiq.utils import powerset


def _generate_interaction_lookup(n: int, min_order: int, max_order: int) -> dict[tuple[int], int]:
    """Generates a lookup dictionary for interactions.

    Args:
        n: The number of players.
        min_order: The minimum order of the approximation.
        max_order: The maximum order of the approximation.

    Returns:
        A dictionary that maps interactions to their index in the values vector.
    """
    interaction_lookup = {
        interaction: i
        for i, interaction in enumerate(
            powerset(set(range(n)), min_size=min_order, max_size=max_order)
        )
    }
    return interaction_lookup
