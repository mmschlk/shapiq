from __future__ import annotations

from leaderboard.runner.approximator_registry import get_approximator_class
from shapiq import KernelSHAPIQ, PermutationSamplingSV
from shapiq.approximator import ProxySHAP


def test_get_approximator_class_positive():
    assert get_approximator_class("ProxySHAP") is ProxySHAP
    assert get_approximator_class("KernelSHAPIQ") is KernelSHAPIQ
    assert get_approximator_class("PermutationSamplingSV") is PermutationSamplingSV
