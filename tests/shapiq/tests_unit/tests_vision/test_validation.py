"""Tests for ``shapiq.vision.validation``.

Two helpers guard every architecture and masking strategy against models they
cannot drive:

- :func:`validate_config_attributes` checks that a Hugging Face style ``config``
  defines the fields a strategy reads (e.g. ``patch_size``, ``hidden_size``).
- :class:`ModelCompatible` gives strategies a shared ``validate_model`` that
  checks a model against the protocol they declare.

Both are expected to fail loudly at construction time rather than deep inside a
forward pass, so these tests assert on the error type and on the message parts a
user needs to act on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from shapiq.vision.custom_types import VisionModel
from shapiq.vision.validation import ModelCompatible, validate_config_attributes


class _CallableModel:
    """Smallest object satisfying the ``VisionModel`` protocol."""

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(1, 2)


class TestValidateConfigAttributes:
    def test_passes_when_all_attributes_defined(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(image_size=224, patch_size=16))
        validate_config_attributes(model, ("image_size", "patch_size"), "requester")

    def test_passes_for_empty_attribute_list(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace())
        validate_config_attributes(model, (), "requester")

    def test_raises_when_model_has_no_config(self) -> None:
        with pytest.raises(TypeError, match="requires a model exposing a ``config``"):
            validate_config_attributes(_CallableModel(), ("image_size",), "requester")

    def test_missing_config_error_names_the_model_type(self) -> None:
        with pytest.raises(TypeError, match="_CallableModel"):
            validate_config_attributes(_CallableModel(), ("image_size",), "requester")

    def test_raises_when_attribute_absent(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(image_size=224))
        with pytest.raises(TypeError, match="``patch_size``"):
            validate_config_attributes(model, ("image_size", "patch_size"), "requester")

    def test_none_valued_attribute_counts_as_missing(self) -> None:
        """HF configs default optional fields to None, which is as unusable as absent."""
        model = SimpleNamespace(config=SimpleNamespace(patch_size=None))
        with pytest.raises(TypeError, match="``patch_size``"):
            validate_config_attributes(model, ("patch_size",), "requester")

    def test_reports_every_missing_attribute_at_once(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace())
        with pytest.raises(TypeError) as err:
            validate_config_attributes(model, ("image_size", "patch_size"), "requester")
        assert "``image_size``" in str(err.value)
        assert "``patch_size``" in str(err.value)

    def test_requester_name_is_included(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace())
        with pytest.raises(TypeError, match="PatchStrategy default"):
            validate_config_attributes(model, ("patch_size",), "PatchStrategy default")

    def test_hint_is_appended_when_given(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace())
        with pytest.raises(TypeError, match="Pass an explicit player_strategy"):
            validate_config_attributes(
                model, ("patch_size",), "requester", hint="Pass an explicit player_strategy."
            )

    def test_no_trailing_hint_when_omitted(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace())
        with pytest.raises(TypeError) as err:
            validate_config_attributes(model, ("patch_size",), "requester")
        assert str(err.value).endswith("``config``.")


class TestModelCompatible:
    def test_subclass_must_declare_a_protocol(self) -> None:
        with pytest.raises(TypeError, match="must define 'compatible_model_protocol'"):

            class _NoProtocol(ModelCompatible):
                pass

    def test_subclass_inherits_declared_protocol(self) -> None:
        class _Parent(ModelCompatible):
            compatible_model_protocol = VisionModel

        class _Child(_Parent):
            pass

        _Child.validate_model(_CallableModel())  # inherited declaration is enough

    def test_validate_model_accepts_compatible_model(self) -> None:
        class _Strategy(ModelCompatible):
            compatible_model_protocol = VisionModel

        _Strategy.validate_model(_CallableModel())

    def test_validate_model_rejects_incompatible_model(self) -> None:
        class _Strategy(ModelCompatible):
            compatible_model_protocol = VisionModel

        with pytest.raises(TypeError, match="VisionModel"):
            _Strategy.validate_model(object())

    def test_rejection_message_names_the_offending_type(self) -> None:
        class _Strategy(ModelCompatible):
            compatible_model_protocol = VisionModel

        with pytest.raises(TypeError, match="dict"):
            _Strategy.validate_model({})

    def test_tuple_protocol_accepts_any_member(self) -> None:
        class _Strategy(ModelCompatible):
            compatible_model_protocol = (VisionModel, torch.nn.Module)

        _Strategy.validate_model(_CallableModel())
        _Strategy.validate_model(torch.nn.Linear(2, 1))

    def test_tuple_protocol_error_lists_all_expected_names(self) -> None:
        class _Strategy(ModelCompatible):
            compatible_model_protocol = (VisionModel, torch.nn.Module)

        with pytest.raises(TypeError) as err:
            _Strategy.validate_model(object())
        assert "VisionModel" in str(err.value)
        assert "Module" in str(err.value)
