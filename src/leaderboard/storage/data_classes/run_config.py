"""Module defining the RunConfig data class, which represents the immutable configuration of a run in the leaderboard storage system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Immutable configuration that uniquely identifies a family of runs.

    Two runs sharing the same ``RunConfig`` differ only in their random seed
    (or other non-config factors) and are aggregated together for metric
    reporting.
    """

    game_name: str
    n_players: int
    approximator_name: str
    index: str
    max_order: int
    budget: int
    ground_truth_method: str
    game_params: dict[str, Any] = field(default_factory=dict)
    approximator_params: dict[str, Any] = field(default_factory=dict)

    # Serialisation
    def to_dict(self) -> dict[str, Any]:
        """Convert config to a plain dict (usable as a MongoDB query)."""
        return {
            "game_name": self.game_name,
            "n_players": self.n_players,
            "approximator_name": self.approximator_name,
            "index": self.index,
            "max_order": self.max_order,
            "budget": self.budget,
            "ground_truth_method": self.ground_truth_method,
            "game_params": self.game_params,
            "approximator_params": self.approximator_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Construct a ``RunConfig`` from a plain dictionary."""
        return cls(
            game_name=data["game_name"],
            n_players=data["n_players"],
            approximator_name=data["approximator_name"],
            index=data["index"],
            max_order=data["max_order"],
            budget=data["budget"],
            ground_truth_method=data["ground_truth_method"],
            game_params=data.get("game_params", {}),
            approximator_params=data.get("approximator_params", {}),
        )

    # Overwrite default string representation for better readability in logs
    def __repr__(self) -> str:
        """Provide a concise and informative string representation of the RunConfig for debugging and logging purposes."""
        return (
            f"RunConfig("
            f"game={self.game_name!r}, "
            f"n_players={self.n_players}, "
            f"approximator={self.approximator_name!r}, "
            f"index={self.index!r}, "
            f"max_order={self.max_order}, "
            f"budget={self.budget}, "
            f"ground_truth={self.ground_truth_method!r}"
            f")"
        )
