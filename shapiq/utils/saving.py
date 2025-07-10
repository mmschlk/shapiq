"""Utils io module for saving and loading data from disk."""

from __future__ import annotations

import datetime
import json
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Literal

    from shapiq.typing import JSONType, MetadataBlock


def safe_tuple_to_str(t: tuple[int, ...]) -> str:
    """Converts a tuple of integers into a string representation for saving purposes."""
    if len(t) == 0:
        return "Empty"
    # make tuple into a hash
    return ",".join(map(str, t))


def safe_str_to_tuple(s: str) -> tuple[int, ...]:
    """Converts a string representation of integers back into a tuple of integers."""
    if s == "Empty":
        return ()
    return tuple(map(int, s.split(",")))


def interactions_to_dict(
    interactions: Mapping[tuple[int, ...], float],
) -> dict[str, float]:
    """Converts a mapping of interactions to a dictionary for saving."""
    return {safe_tuple_to_str(tup): value for tup, value in interactions.items()}


def dict_to_interactions(
    interaction_dict: dict[str, float],
) -> dict[tuple[int, ...], float]:
    """Converts a dictionary of interaction values back to a mapping of tuples to float values."""
    return {safe_str_to_tuple(tup_str): value for tup_str, value in interaction_dict.items()}


def lookup_and_values_to_dict(
    interaction_lookup: Mapping[tuple[int, ...], int],
    interaction_values: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Converts a pair of interaction lookup and values into a dictionary for saving.

    Args:
        interaction_lookup: A mapping from tuples of integers to indices.
        interaction_values: A sequence of float values corresponding to the indices in the lookup.

    Returns:
        A dictionary mapping string representations of tuples to their corresponding interaction
            values.
    """
    return {
        safe_tuple_to_str(tup): interaction_values[interaction_lookup[tup]]
        for tup in interaction_lookup
    }


def dict_to_lookup_and_values(
    interaction_dict: dict[str, float],
) -> tuple[dict[tuple[int, ...], int], np.ndarray]:
    """Converts a dictionary of interaction values back to a lookup and values.

    Args:
        interaction_dict: A dictionary mapping string representations of tuples to float values.

    Returns:
        A tuple containing a mapping from tuples of integers to indices and a sequence of float
            values.
    """
    interaction_lookup = {
        safe_str_to_tuple(tup_str): idx for idx, tup_str in enumerate(interaction_dict)
    }
    interaction_values = [interaction_dict[tup_str] for tup_str in interaction_dict]
    interaction_values = np.array(interaction_values, dtype=float)
    return interaction_lookup, interaction_values


def make_file_metadata(
    object_to_store: object,
    *,
    data_type: Literal["interaction_values", "game"] | None = None,
    desc: str | None = None,
    created_from: object | None = None,
    parameters: JSONType = None,
) -> MetadataBlock:
    """Creates a metadata block for saving interaction values or games."""
    return {
        "object_name": object_to_store.__class__.__name__,
        "data_type": data_type,
        "version": version("shapiq"),
        "created_from": repr(created_from) if created_from else None,
        "description": desc,
        "parameters": parameters or {},
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat() + "Z",
    }


def save_json(data: JSONType, path: Path) -> None:
    """Saves data to a JSON file."""
    if not path.name.endswith(".json"):
        path = path.with_suffix(".json")

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    with path.open("w") as file:
        file.write(json_str)
