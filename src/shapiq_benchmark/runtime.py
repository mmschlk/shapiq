"""Portable execution settings and authenticated, per-method source overlays.

These are operational inputs, separate from frozen games and estimator settings.
Overlay files are trusted Python code, just like a local estimator plugin.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType


@dataclass(frozen=True)
class ExecutionSettings:
    """Final allowance for a fresh run; historical attempts retain their own limits."""

    approximate_seconds: float = 3600
    startup_seconds: float = 600
    output_seconds: float = 600
    memory_gib: float = 8

    def __post_init__(self) -> None:
        """Reject invalid limits before allocating a worker."""
        for value in asdict(self).values():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                message = "Execution settings must be positive finite numbers."
                raise ValueError(message)

    @property
    def wall_seconds(self) -> float:
        """Reserve all three phases before starting a cell."""
        return self.startup_seconds + self.approximate_seconds + self.output_seconds

    @classmethod
    def load(cls, path: Path) -> ExecutionSettings:
        """Load explicit limits, rejecting misspelled or unsupported fields."""
        return cls(**json.loads(path.read_text()))


def recovery_memory(method: str, order: int, *, previous_gib: float = 8) -> float:
    """Memory-failure policy; a timeout-only retry instead retains previous_gib."""
    dense = {"InconsistentKernelSHAPIQ", "KernelSHAPIQ", "RegressionFSII", "RegressionFBII"}
    return max(previous_gib, 512 if method in dense and order > 1 else 128)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_overlays(path: Path) -> dict:
    """Resolve a portable method-to-module manifest and authenticate its source files."""
    overlays = json.loads(path.read_text())
    if not isinstance(overlays, dict):
        message = "Overlays must map method names to source descriptors."
        raise TypeError(message)
    result = {}
    for method, descriptor in overlays.items():
        if (
            not isinstance(method, str)
            or not isinstance(descriptor, dict)
            or set(descriptor) != {"module", "path", "sha256", "base_sha256"}
            or not isinstance(descriptor["path"], str)
            or not isinstance(descriptor["module"], str)
            or not descriptor["module"].startswith("shapiq.approximator.")
            or any(
                not isinstance(descriptor[key], str)
                or len(descriptor[key]) != 64
                or any(c not in "0123456789abcdef" for c in descriptor[key])
                for key in ("sha256", "base_sha256")
            )
        ):
            message = "Invalid estimator source overlay descriptor."
            raise ValueError(message)
        source = (path.parent / descriptor["path"]).resolve()
        if _digest(source) != descriptor["sha256"]:
            message = f"Overlay source changed: {method}"
            raise ValueError(message)
        result[method] = {**descriptor, "path": str(source)}
    return result


class _Overlay(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, descriptor: dict) -> None:
        self.descriptor = descriptor

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname != self.descriptor["module"]:
            return None
        original = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if (
            original is None
            or original.origin is None
            or _digest(Path(original.origin)) != self.descriptor["base_sha256"]
        ):
            message = "Overlay does not match the underlying estimator source."
            raise ValueError(message)
        return importlib.util.spec_from_loader(fullname, self, origin=self.descriptor["path"])

    def create_module(self, spec: ModuleSpec) -> None:
        del spec  # The import system creates an ordinary module.

    def exec_module(self, module: ModuleType) -> None:
        source = Path(self.descriptor["path"])
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.descriptor["sha256"]:
            message = "Overlay source changed before import."
            raise ValueError(message)
        module.__file__ = str(source)
        exec(compile(content, str(source), "exec"), module.__dict__)  # noqa: S102 -- authenticated trusted plugin


def install_overlay(descriptor: dict) -> None:
    """Install before scientific imports in a fresh worker; never modify the parent."""
    if descriptor["module"] in sys.modules:
        message = "Source overlays require a fresh worker before estimator import."
        raise ValueError(message)
    sys.meta_path.insert(0, _Overlay(descriptor))
