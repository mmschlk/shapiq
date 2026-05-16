from abc import ABC, abstractmethod
from .result import MetricResult

class Metric(ABC):

    name = "base"
    higher_is_better = False

    @abstractmethod
    def compute(self, ground_truth, estimated) -> MetricResult:
        pass