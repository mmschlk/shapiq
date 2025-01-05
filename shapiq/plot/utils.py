"""This utility module contains helper functions for plotting."""

import re
from collections.abc import Iterable

__all__ = ["abbreviate_feature_names", "format_value"]


def format_value(s, format_str):
    """Strips trailing zeros and uses a unicode minus sign."""
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r"\.?0+$", "", s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s


def format_labels(feature_mapping, feature_tuple):
    if len(feature_tuple) == 0:
        return "Baseval."
    elif len(feature_tuple) == 1:
        return str(feature_mapping[feature_tuple[0]])
    else:
        return " x ".join([feature_mapping[f] for f in feature_tuple])


def abbreviate_feature_names(feature_names: Iterable[str]) -> list[str]:
    """A rudimentary function to abbreviate feature names for plotting.

    Args:
        feature_names: The feature names to be abbreviated.

    Returns:
        list[str]: The abbreviated feature names.
    """
    abbreviated_names = []
    for name in feature_names:
        name = str(name)
        name = name.strip()
        capital_letters = sum(1 for c in name if c.isupper())
        seperator_chars = (" ", "_", "-", ".")
        is_seperator_in_name = any([c in seperator_chars for c in name[:-1]])
        if is_seperator_in_name:
            for seperator in seperator_chars:
                name = name.replace(seperator, ".")
            name_parts = name.split(".")
            new_name = ""
            for part in name_parts:
                if part:
                    new_name += part[0].upper()
            abbreviated_names.append(new_name)
        elif capital_letters > 1:
            new_name = "".join([c for c in name if c.isupper()])
            abbreviated_names.append(new_name[0:3])
        else:
            abbreviated_names.append(name.strip()[0:3] + ".")
    return abbreviated_names
