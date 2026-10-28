"""Opt in to exhaustive typeguard checking without changing normal runtime tests."""

import sys

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--typeguard",
        action="store_true",
        help="Exhaustively check shapiq and test annotations with jaxtyping/typeguard.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--typeguard"):
        return

    if "shapiq" in sys.modules:
        msg = "--typeguard must be configured before shapiq is imported. Move shapiq imports out of early pytest setup."
        raise pytest.UsageError(msg)

    # Keep checker dependencies optional, and import shapiq only after the hook.
    from jaxtyping import install_import_hook, jaxtyped  # noqa: PLC0415

    try:
        from typeguard import typechecked  # noqa: PLC0415
    except ImportError as error:
        msg = "--typeguard requires the test-typeguard dependency group: uv run --group test-typeguard pytest --typeguard"
        raise pytest.UsageError(msg) from error

    if sys.version_info >= (3, 15):
        from importlib.util import cache_from_source  # noqa: PLC0415

        from jaxtyping import _import_hook  # noqa: PLC0415

        # jaxtyping 0.3.11 passes the removed debug_override argument (None).
        # Keep this Python 3.15 workaround confined to guarded tests.
        def compatible_cache_from_source(path, debug_override=None, **kwargs):
            del debug_override
            return cache_from_source(path, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            "jaxtyping._import_hook.cache_from_source", compatible_cache_from_source
        )
        config.add_cleanup(monkeypatch.undo)

        # Python 3.15 also passes a new fullname argument to source_to_code.
        source_to_code = _import_hook._JaxtypingLoader.source_to_code

        def compatible_source_to_code(loader, data, path, fullname=None, *, _optimize=-1):
            del fullname
            return source_to_code(loader, data, path, _optimize=_optimize)

        monkeypatch.setattr(
            _import_hook._JaxtypingLoader, "source_to_code", compatible_source_to_code
        )

    # Install before collection imports library boundaries or test modules.
    hook = install_import_hook(["shapiq", "tests"], "typeguard.typechecked")
    config.add_cleanup(hook.uninstall)

    import shapiq.typing  # noqa: PLC0415

    # Dataclass constructors already checked by beartype would not be rewrapped
    # by the import hook, so explicit boundaries must select typeguard too.
    runtime_decorator = shapiq.typing.typechecked
    config.add_cleanup(lambda: setattr(shapiq.typing, "typechecked", runtime_decorator))
    shapiq.typing.typechecked = jaxtyped(typechecker=typechecked)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--typeguard"):
        return

    skip = pytest.mark.skip(reason="requires exhaustive checks: run pytest --typeguard")
    for item in items:
        if item.get_closest_marker("typeguard") is not None:
            item.add_marker(skip)
