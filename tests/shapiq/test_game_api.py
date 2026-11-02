"""Execute the game API sketchbook using the public library types."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from shapiq.game import TableGame


@pytest.fixture(scope="module")
def sketchbook(request: pytest.FixtureRequest) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples" / "game_api.ipynb"
    notebook = json.loads(path.read_text())
    module = ModuleType(f"{__name__}.sketchbook")
    sys.modules[module.__name__] = module
    # A normal fixture stays recognizable after runtime-checking wrappers.
    request.addfinalizer(lambda: sys.modules.pop(module.__name__))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = compile(
                "".join(cell["source"]),
                f"{path}:{cell['id']}",
                "exec",
                dont_inherit=True,
            )
            exec(code, module.__dict__)
    return module


def test_sketchbook_uses_public_table_game(sketchbook):
    assert sketchbook.TableGame is TableGame
    assert isinstance(sketchbook.g, TableGame)
