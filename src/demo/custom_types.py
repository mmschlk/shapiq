from typing import Callable, Literal
from shapiq import InteractionValues

InteractionIndex = Literal[
    "SV",
    "BV",
    "SII",
    "BII",
    "k-SII",
    "STII",
    "FBII",
    "FSII",
    "kADD-SHAP",
    "CHII",
]

MetricFunction = Callable[[InteractionValues, InteractionValues], float]