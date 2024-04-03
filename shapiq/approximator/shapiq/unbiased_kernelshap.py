"""This module contains the Unbiased KernelSHAP approximation method for the Shapley value (SV).
The Unbiased KernelSHAP method is a variant of the KernelSHAP. However, it was shown that
Unbiased KernelSHAP is a more specific variant of the ShapIQ interaction method.
"""
from typing import Optional

from .shapiq import ShapIQ


class UnbiasedKernelSHAP(ShapIQ):
    def __init__(
        self,
        n: int,
        random_state: Optional[int] = None,
    ):
        super().__init__(n, 1, "SV", False, random_state)
