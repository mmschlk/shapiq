"""Environment information retrieval for the leaderboard runner."""

from __future__ import annotations

import platform


# TO DO: Placeholder-implementation
def get_hardware_info() -> dict:
    """Return basic hardware and Python runtime information.

    Returns:
        A dictionary containing CPU information, RAM information if available,
        and the Python version.
    """
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "python_version": platform.python_version(),
    }
