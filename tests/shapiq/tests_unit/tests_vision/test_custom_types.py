"""Tests for ``shapiq.vision.custom_types`` and the shared error helpers.

:class:`CoalitionDomain` is what keeps a pixel-space player strategy from being
paired with a token-space masker, and :class:`VisionModel` is the runtime-checkable
protocol every architecture validates user models against. Because both are
enforced with ``isinstance`` at construction time, the exact set of members they
check is behaviour rather than annotation, and is worth pinning down.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from shapiq.vision._error import _vision_protocol_error
from shapiq.vision.custom_types import ClassificationOutput, CoalitionDomain, VisionModel


class TestCoalitionDomain:
    def test_members(self) -> None:
        assert {domain.value for domain in CoalitionDomain} == {"pixel", "token"}

    def test_members_are_distinct(self) -> None:
        assert CoalitionDomain.PIXEL is not CoalitionDomain.TOKEN

    def test_identity_comparison_works(self) -> None:
        """The architecture compares domains with ``is``, so lookups must return singletons."""
        assert CoalitionDomain("pixel") is CoalitionDomain.PIXEL
        assert CoalitionDomain("token") is CoalitionDomain.TOKEN


class TestVisionModelProtocol:
    def test_callable_object_satisfies_protocol(self) -> None:
        class _Model:
            def __call__(self, x):
                return x

        assert isinstance(_Model(), VisionModel)

    def test_torch_module_satisfies_protocol(self) -> None:
        assert isinstance(torch.nn.Linear(2, 1), VisionModel)

    def test_plain_function_satisfies_protocol(self) -> None:
        assert isinstance(lambda x: x, VisionModel)

    def test_non_callable_object_fails_protocol(self) -> None:
        assert not isinstance(object(), VisionModel)

    def test_namespace_without_call_fails_protocol(self) -> None:
        """A config-only stub is the common mistake and must not pass."""
        assert not isinstance(SimpleNamespace(config=SimpleNamespace(patch_size=16)), VisionModel)


class TestClassificationOutputProtocol:
    def test_object_with_logits_satisfies_protocol(self) -> None:
        assert isinstance(SimpleNamespace(logits=torch.zeros(1, 2)), ClassificationOutput)

    def test_object_without_logits_fails_protocol(self) -> None:
        assert not isinstance(SimpleNamespace(predictions=torch.zeros(1, 2)), ClassificationOutput)


class TestVisionProtocolError:
    def test_returns_type_error(self) -> None:
        assert isinstance(_vision_protocol_error(object()), TypeError)

    def test_message_names_the_offending_type(self) -> None:
        assert "dict" in str(_vision_protocol_error({}))

    def test_expected_protocol_is_prefixed_when_given(self) -> None:
        message = str(_vision_protocol_error(object(), "VisionModel"))
        assert message.startswith("Expected compatibility with VisionModel.")

    def test_no_prefix_when_expected_omitted(self) -> None:
        assert not str(_vision_protocol_error(object())).startswith("Expected")
