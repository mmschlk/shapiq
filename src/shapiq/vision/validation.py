"""Shared validation helpers for vision model compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._error import _vision_protocol_error

if TYPE_CHECKING:
    from collections.abc import Iterable

    from shapiq.typing import Model


def validate_config_attributes(
    model: Model,
    attributes: Iterable[str],
    requester: str,
    hint: str = "",
) -> None:
    """Validate that ``model.config`` defines the given attributes.

    Args:
        model: Model whose ``config`` is inspected.
        attributes: Names of the config attributes that must be defined.
        requester: Name of the caller, used in the error message.
        hint: Optional trailing sentence appended to the error message.

    Raises:
        TypeError: If ``model`` exposes no ``config``, or if any of
            ``attributes`` is undefined on it.
    """
    config = getattr(model, "config", None)
    if config is None:
        msg = f"{requester} requires a model exposing a ``config``, got {type(model).__name__}."
        raise TypeError(msg)

    missing = [name for name in attributes if getattr(config, name, None) is None]
    if missing:
        missing_str = ", ".join(f"``{name}``" for name in missing)
        msg = f"{requester} requires {missing_str} to be defined on the model's ``config``."
        if hint:
            msg = f"{msg} {hint}"
        raise TypeError(msg)


class ModelCompatible:
    """Trait for strategies that validate a compatible model protocol.

    Subclasses declare the model protocol they accept via
    ``compatible_model_protocol`` and inherit a shared ``validate_model``
    implementation that raises a ``TypeError`` for incompatible models.
    """

    compatible_model_protocol: type | tuple[type, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure subclasses declare a compatible model protocol."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "compatible_model_protocol"):
            msg = f"{cls.__name__} must define 'compatible_model_protocol'."
            raise TypeError(msg)

    @classmethod
    def validate_model(cls, model: Model) -> None:
        """Validate that ``model`` satisfies the declared protocol.

        Validation with protocols will only check for the presence of attributes and methods,
        not their signatures or return types.

        Args:
            model: Object to validate against ``compatible_model_protocol``.

        Raises:
            TypeError: If ``model`` is not compatible with the declared
                protocol.
        """
        protocol = cls.compatible_model_protocol

        if not isinstance(model, protocol):
            if isinstance(protocol, tuple):
                expected = ", ".join(proto.__name__ for proto in protocol)
            else:
                expected = protocol.__name__

            raise _vision_protocol_error(model, expected)
