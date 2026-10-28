"""Selective runtime checks must work without pytest's typeguard import hook."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from jaxtyping import TypeCheckError


def _ordinary_identity(value: int) -> int:
    return value


def test_automatic_typechecking_is_opt_in(pytestconfig):
    if pytestconfig.getoption("--typeguard"):
        with pytest.raises(TypeCheckError):
            _ordinary_identity("not an integer")
    else:
        assert _ordinary_identity("not an integer") == "not an integer"


def test_runtime_checks_do_not_require_typeguard():
    script = dedent("""\
        import sys

        # Simulate a normal runtime installation without the test dependency.
        sys.modules["typeguard"] = None

        import json
        from inspect import Parameter, signature
        from pathlib import Path
        from typing import get_overloads

        import jax.numpy as jnp
        from jaxtyping import TypeCheckError
        from shapiq.coalition.coalition import (
            BoolCoalition,
            pack_bool_coalition,
            unpack_bool_coalition,
        )
        from shapiq.players import FinitePlayers

        # Check overloads without the import hook, which rewrites their registry keys.
        constructors = get_overloads(FinitePlayers.__init__)
        assert len(constructors) == 3
        for constructor in constructors:
            parameters = signature(constructor).parameters
            latent_count = parameters["latent_n_players"]
            assert latent_count.kind is Parameter.KEYWORD_ONLY
            assert latent_count.default is None

        coalition = jnp.ones((2, 9), dtype=bool)
        packed = pack_bool_coalition(coalition)
        assert bool(jnp.array_equal(unpack_bool_coalition(packed, 9), coalition))

        for attempt in (
            lambda: pack_bool_coalition(jnp.ones((2, 9))),
            lambda: unpack_bool_coalition(jnp.ones((2, 2)), 9),
            lambda: unpack_bool_coalition(jnp.ones((2, 1), dtype=jnp.uint8), 9),
        ):
            try:
                attempt()
            except TypeCheckError:
                pass
            else:
                raise AssertionError("A runtime coalition contract was not enforced")

        # Importing shapiq must not enable checking for unrelated functions.
        def unchecked(value: BoolCoalition) -> BoolCoalition:
            return value

        wrong_dtype = jnp.ones((9,))
        assert unchecked(wrong_dtype) is wrong_dtype

        # The notebook checks constructors and external coalition evaluation.
        notebook = json.loads(Path("examples/game_api.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                code = compile("".join(cell["source"]), cell["id"], "exec", dont_inherit=True)
                exec(code, globals())

        assert sys.modules["typeguard"] is None
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
