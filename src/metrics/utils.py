import copy

from shapiq.interaction_values import InteractionValues


def remove_empty_value_if_needed(value):
    if not isinstance(value, InteractionValues):
        return value

    try:
        new_value = copy.deepcopy(value)
        empty_index = new_value.interaction_lookup[()]
        new_value.values[empty_index] = 0
        return new_value
    except KeyError:
        return value