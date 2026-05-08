from enum import Enum


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    USER_INPUT_MISCONFIGURED = "USER_INPUT_MISCONFIGURED"


    @classmethod
    def from_str(cls, status_str: str) -> "RunStatus":
        try:
            return cls(status_str)
        except ValueError:
            raise ValueError(f"Invalid RunStatus: {status_str}")


class ApproximatorType(str, Enum):
    KERNEL_SHAP = "KernelSHAP"
    PERMUTATION = "Permutation"
    SAMPLING = "Sampling"
    EXACT = "Exact"

    @classmethod
    def from_str(cls, approximator_str: str) -> "ApproximatorType":
        try:
            return cls(approximator_str)
        except ValueError:
            raise ValueError(f"Invalid ApproximatorType: {approximator_str}")


class GroundTruthMethod(str, Enum):
    EXACT = "ExactComputer"
    MONTE_CARLO = "MonteCarlo"

    @classmethod
    def from_str(cls, method_str: str) -> "GroundTruthMethod":
        try:
            return cls(method_str)
        except ValueError:
            raise ValueError(f"Invalid GroundTruthMethod: {method_str}")