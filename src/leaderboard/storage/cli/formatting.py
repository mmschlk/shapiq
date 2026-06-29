"""Terminal formatting utilities for the SIL CLI."""

from __future__ import annotations


class _ColorConfig:
    """Mutable holder for the color-enabled flag."""

    enabled: bool = True


_cfg = _ColorConfig()


def set_color(*, enabled: bool) -> None:
    """Enable or disable ANSI color output for terminal formatting."""
    _cfg.enabled = enabled


def _c(code: str, text: str) -> str:
    """Apply ANSI color code *code* to *text* if color is enabled."""
    if not _cfg.enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    """Return *text* in bold formatting."""
    return _c("1", text)


def dim(text: str) -> str:
    """Return *text* in dim formatting."""
    return _c("2", text)


def green(text: str) -> str:
    """Return *text* in green formatting."""
    return _c("32", text)


def yellow(text: str) -> str:
    """Return *text* in yellow formatting."""
    return _c("33", text)


def red(text: str) -> str:
    """Return *text* in red formatting."""
    return _c("31", text)


def cyan(text: str) -> str:
    """Return *text* in cyan formatting."""
    return _c("36", text)


def magenta(text: str) -> str:
    """Return *text* in magenta formatting."""
    return _c("35", text)


def blue(text: str) -> str:
    """Return *text* in blue formatting."""
    return _c("34", text)


def header(text: str) -> str:
    """Return *text* formatted as a header (bold cyan)."""
    return bold(cyan(text))


def ok(text: str) -> str:
    """Return *text* formatted as a success message (green with checkmark)."""
    return green(f"✓ {text}")


def warn(text: str) -> str:
    """Return *text* formatted as a warning message (yellow with exclamation mark)."""
    return yellow(f"⚠ {text}")


def error(text: str) -> str:
    """Return *text* formatted as an error message (red with cross)."""
    return red(f"✗ {text}")


def info(text: str) -> str:
    """Return *text* formatted as an info message (dim)."""
    return dim(f"  {text}")


def storage_id(sid: str) -> str:
    """Render a storage ID prominently."""
    return bold(magenta(f"[{sid}]"))


def prompt(active_ids: list[str]) -> str:
    """Return the command prompt string, including active storage IDs if any."""
    if active_ids:
        ids = " ".join(magenta(f"[{s}]") for s in active_ids)
        return f"{ids} {bold('sil')}> "
    return f"{bold('sil')}> "
