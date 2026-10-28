"""Batched player sets and their coalition compatibility contracts."""

from .finite import FinitePlayers
from .latent import LatentPlayers
from .players import Players

__all__ = ["FinitePlayers", "LatentPlayers", "Players"]
