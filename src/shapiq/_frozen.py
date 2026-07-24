"""Enforced frozenness for the configuration tier.

Games, samplers, and explainers are configuration: values that never
mutate after construction. The library's process design leans on that
invariant everywhere — estimates carry all evolving state, budgets split
exactly because the policy underneath cannot drift — so it is enforced,
not merely documented: instances lock once the most-derived ``__init__``
returns, and assignment afterwards raises a teaching error.

Value-equivalent lazy caches remain possible on frozen objects through
container attributes created during construction (a dict filled later
does not rebind an attribute); that is the same allowance the frozen
evidence caches use, and it never changes observable behavior.
"""

from __future__ import annotations

from abc import ABCMeta
from typing import Any


class FrozenABCMeta(ABCMeta):
    """Metaclass stamping instances as frozen after full construction.

    The stamp happens after the most-derived ``__init__`` returns, so
    subclasses may freely assign configuration attributes anywhere in
    their constructor chain.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - a passthrough; anything narrower degrades constructor types
        """Construct an instance, then lock it."""
        instance = super().__call__(*args, **kwargs)
        object.__setattr__(instance, "_shapiq_frozen", True)
        return instance


class Frozen(metaclass=FrozenABCMeta):
    """Base for configuration values: construction is the only write.

    Assignment or deletion after construction raises ``AttributeError``
    with the intended alternative: build a new value.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        """Assign during construction; teach afterwards."""
        if getattr(self, "_shapiq_frozen", False):
            msg = (
                f"{type(self).__name__} is frozen: configuration never changes after "
                f"construction, so all process state can ride the estimates it "
                f"produces; build a new {type(self).__name__} instead of assigning "
                f"{name!r}"
            )
            raise AttributeError(msg)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Refuse deletion on frozen instances."""
        if getattr(self, "_shapiq_frozen", False):
            msg = (
                f"{type(self).__name__} is frozen: configuration never changes after "
                f"construction; build a new {type(self).__name__} instead of deleting "
                f"{name!r}"
            )
            raise AttributeError(msg)
        super().__delattr__(name)
