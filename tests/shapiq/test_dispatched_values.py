"""Tests for the dispatched prediction-to-values conversion."""

from __future__ import annotations

import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import to_values

LAZINESS_PROBE = """
import sys
import numpy as np
import shapiq

values = shapiq.to_values(np.asarray([1.0, 2.0]))
assert float(values.sum()) == 3.0
assert "torch" not in sys.modules, "importing shapiq or converting numpy must not import torch"

import torch

tensor_values = shapiq.to_values(torch.tensor([1.0, 2.0, 3.0]))
assert float(tensor_values.sum()) == 6.0
print("lazy ok")
"""


def test_fallback_converts_jax_numpy_and_python_values():
    assert to_values(jnp.asarray([1.0, 2.0])).shape == (2,)
    assert to_values(np.asarray([[1, 2], [3, 4]])).shape == (2, 2)
    assert float(to_values(1.5)) == 1.5
    assert to_values([1.0, 2.0, 3.0]).shape == (3,)


def test_torch_never_imports_until_a_tensor_arrives():
    result = subprocess.run(
        [sys.executable, "-c", LAZINESS_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "lazy ok" in result.stdout


def test_torch_tensors_dispatch_to_the_dlpack_conversion():
    torch = pytest.importorskip("torch")
    from shapiq.games.torch import to_jax  # noqa: PLC0415 - requires torch

    tensor = torch.tensor([[0.5, -1.0], [2.0, 0.0]])
    assert jnp.array_equal(to_values(tensor), to_jax(tensor))


def test_masked_game_defaults_to_dispatched_values():
    torch = pytest.importorskip("torch")
    from shapiq import (  # noqa: PLC0415 - keep this module importable without torch
        BaselineMasker,
        DenseCoalitionArray,
        MaskedGame,
        ModelMaskedPredictor,
    )
    from shapiq.games.torch import to_jax  # noqa: PLC0415 - requires torch

    masker = BaselineMasker(
        inputs=torch.tensor([1.0, -2.0, 3.0]),
        baseline=torch.zeros(3),
    )
    predictor = ModelMaskedPredictor(masker=masker, model=lambda x: x.sum(dim=-1))
    defaulted = MaskedGame(masked_predictor=predictor)
    explicit = MaskedGame(masked_predictor=predictor, link_function=to_jax)
    rows = DenseCoalitionArray(jnp.asarray([[True, False, True], [False, False, False]]))
    assert jnp.array_equal(defaulted(rows), explicit(rows))
