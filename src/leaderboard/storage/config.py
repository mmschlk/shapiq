from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class RunConfig:
    game_name: str
    n_players: int
    approximator_name: str
    index: str
    max_order: int
    budget: int
    ground_truth_method: str
    game_params: Dict[str, Any] = field(default_factory=dict)
    approximator_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config into a MongoDB query dictionary."""
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