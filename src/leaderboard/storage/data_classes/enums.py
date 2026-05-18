"""Module defining enums for run status, approximator types, and ground truth methods used in the leaderboard storage data classes."""

from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    """Status of a completed run."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    USER_INPUT_MISCONFIGURED = "USER_INPUT_MISCONFIGURED"

    @classmethod
    def from_str(cls, status_str: str) -> RunStatus:
        try:
            return cls(status_str)
        except ValueError:
            raise ValueError(
                f"Invalid RunStatus: {status_str!r}. Valid values: {[e.value for e in cls]}"
            )


class ApproximatorType(str, Enum):
    """Type of approximator used in a run."""

    KERNEL_SHAP = "KernelSHAP"
    PERMUTATION = "Permutation"
    SAMPLING = "Sampling"
    EXACT = "Exact"

    @classmethod
    def from_str(cls, approximator_str: str) -> ApproximatorType:
        try:
            return cls(approximator_str)
        except ValueError:
            raise ValueError(
                f"Invalid ApproximatorType: {approximator_str!r}. Valid values: {[e.value for e in cls]}"
            )


class GroundTruthMethod(str, Enum):
    """Method used to compute ground truth values."""

    EXACT = "ExactComputer"
    MONTE_CARLO = "MonteCarlo"

    @classmethod
    def from_str(cls, method_str: str) -> GroundTruthMethod:
        try:
            return cls(method_str)
        except ValueError:
            raise ValueError(
                f"Invalid GroundTruthMethod: {method_str!r}. Valid values: {[e.value for e in cls]}"
            )
