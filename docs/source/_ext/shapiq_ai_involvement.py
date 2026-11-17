"""Sphinx extension for the AI involvement of a page: a thin line beside the text, part by part.

The levels are in ``_data/ai-involvement.toml``, from 0 (a person wrote it without AI) to 5 (AI
wrote it and no person checked it). A page reports its parts in a TOML file next to it and names
the file in the ``shapiq-ai-involvement`` directive, once, anywhere on the page::

    .. shapiq-ai-involvement:: ai-involvement.toml

The file has a ``[page]`` table, for the page as a whole, and one ``[[parts]]`` table per part.
A part names the element it covers by a CSS selector, usually the id of a section's label
(``"#fixlip-sec-summary"``) or of a demo's figure (``"#fixlip-figure-masks"``)::

    [page]
    level = 3
    summary = "One sentence on the page as a whole."

    [[parts]]
    target = "#fixlip-sec-summary"
    level = 3
    written = "Claude, from an outline by Max"
    models = ["Claude Opus"]
    human = "Max read it, asked for two changes and approved it."
    sources = "The abstract of the paper."
    checked = 2026-11-17

``target``, ``level`` and ``written`` are required. ``name`` (default, the part's heading or
caption number), ``models``, ``human``, ``sources``, ``checked`` and ``note`` are optional. A
``[defaults]`` table gives every part the fields it does not set itself. A selector that matches
several elements, e.g. ``"figure.fixlip-figure"``, makes each of them a part. A part inside another
part, e.g. a figure inside a section, shows its own level where it is. Of two parts on the same
element, the later one counts. A target given as an id that the page does not have warns.

The directive writes the levels, the page's parts and the URL of the page ``ai-involvement`` as
JSON into the page, and adds ``_static/js/ai-involvement.js``, which draws the line, and
``_static/css/ai-involvement.css``. Outside HTML, it writes nothing.

The ``shapiq-ai-levels`` directive lists the levels, each with the line at its place, for the
page that explains them.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata
    from sphinx.writers.html5 import HTML5Translator

logger = logging.getLogger(__name__)

_LEVELS = Path("_data") / "ai-involvement.toml"
_ABOUT = "ai-involvement"  # the page that explains the levels
_PAGE = {"level", "summary"}, {"written", "models", "human", "checked", "note"}
_PART = {"target", "level", "written"}, {"name", "models", "human", "sources", "checked", "note"}
_DEFAULTS = set(), (_PART[1] | {"level", "written"}) - {"name"}


class InvolvementNode(nodes.General, nodes.Element):
    """The data of the line: the levels and the page's parts, written as JSON."""


class LevelGlyphNode(nodes.Inline, nodes.Element):
    """A small drawing of the line at one level, for the list of levels."""


def _levels(srcdir: str) -> list[dict[str, Any]]:
    """Return the levels, in order."""
    with (Path(srcdir) / _LEVELS).open("rb") as file:
        return sorted(tomllib.load(file)["levels"], key=lambda level: level["level"])


def _plain(value: Any) -> Any:  # noqa: ANN401
    """Return a TOML value as JSON can hold it (dates as ISO strings)."""
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _problems(entry: dict[str, Any], fields: tuple[set[str], set[str]], top: int) -> list[str]:
    """Return what is wrong with the page's table or a part: missing or unknown fields, a level."""
    required, optional = fields
    problems = [f"missing field {key!r}" for key in sorted(required - entry.keys())]
    problems += [f"unknown field {key!r}" for key in sorted(entry.keys() - required - optional)]
    level = entry.get("level")
    if "level" in entry and not (isinstance(level, int) and 0 <= level <= top):
        problems.append(f"level {level!r} is not a whole number from 0 to {top}")
    if "models" in entry and not (
        isinstance(entry["models"], list) and all(isinstance(m, str) for m in entry["models"])
    ):
        problems.append("'models' is not a list of strings")
    return problems


class InvolvementDirective(SphinxDirective):
    """Report a page's AI involvement: ``.. shapiq-ai-involvement:: <file>``, next to the page."""

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        """Return the node with the page's data, after checking the file."""
        path = (Path(self.env.doc2path(self.env.docname)).parent / self.arguments[0]).resolve()
        if not path.is_file():
            msg = f"no file {self.arguments[0]!r} next to the page"
            raise self.error(msg)
        self.env.note_dependency(str(path))
        self.env.note_dependency(str(Path(self.env.srcdir) / _LEVELS))
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            msg = f"{path.name}: {error}"
            raise self.error(msg) from error
        top = _levels(self.env.srcdir)[-1]["level"]
        problems = [f"[page]: {p}" for p in _problems(data.get("page", {}), _PAGE, top)]
        defaults = data.get("defaults", {})
        problems += [f"[defaults]: {p}" for p in _problems(defaults, _DEFAULTS, top)]
        parts = [{**defaults, **part} for part in data.get("parts", [])]
        for number, part in enumerate(parts, start=1):
            label = part.get("target", f"part {number}")
            problems += [f"[[parts]] {label}: {p}" for p in _problems(part, _PART, top)]
        for problem in problems:
            logger.warning("%s: %s", path.name, problem, location=self.get_location())
        node = InvolvementNode(page=_plain(data.get("page", {})), parts=_plain(parts))
        node["source_file"] = path.name
        return [node]


def _resolve_targets(app: Sphinx, doctree: nodes.document, docname: str) -> None:  # noqa: ARG001
    """Point a part's target at the id that the HTML has, and warn for an id the page lacks.

    A section's label (``.. _fixlip-sec-summary:``) is one of several ids of the section, and the
    HTML writes only its first one (``summary``).
    """
    first = {}
    for element in doctree.findall(nodes.Element):
        for i in element.get("ids", []):
            first.setdefault(i, element["ids"][0])
    for node in doctree.findall(InvolvementNode):
        for part in node["parts"]:
            pieces = []
            for raw in part.get("target", "").split(","):
                piece = raw.strip()
                if re.fullmatch(r"#[\w-]+", piece) and piece[1:] not in first:
                    logger.warning(
                        "%s: no element with the id %r on the page",
                        node["source_file"],
                        piece[1:],
                        location=node,
                    )
                elif re.fullmatch(r"#[\w-]+", piece):
                    piece = "#" + first[piece[1:]]
                pieces.append(piece)
            part["target"] = ", ".join(pieces)


def _visit_involvement(self: HTML5Translator, node: InvolvementNode) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    data = {
        "levels": _levels(self.builder.srcdir),
        "page": node["page"],
        "parts": node["parts"],
        "about": relative_uri(page, self.builder.get_target_uri(_ABOUT)),
    }
    text = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    self.body.append(
        f'<script type="application/json" id="shapiq-ai-involvement">{text}</script>\n'
    )
    raise nodes.SkipNode


def _skip(self: nodes.NodeVisitor, node: nodes.Node) -> None:  # noqa: ARG001
    raise nodes.SkipNode


def _glyph(level: int, top: int) -> str:
    """Return the line at a level as a small SVG: faint lanes and the line in its lane."""
    gap, height = 3, 18
    width = gap * top + 3
    lanes = "".join(
        f'<line class="lane" x1="{1.5 + i * gap}" x2="{1.5 + i * gap}" y1="0" y2="{height}"/>'
        for i in range(top + 1)
    )
    x = 1.5 + level * gap
    return (
        f'<svg class="shapiq-ai-glyph" width="{width}" height="{height}" aria-hidden="true">'
        f'{lanes}<line class="line" x1="{x}" x2="{x}" y1="0" y2="{height}"/></svg>'
    )


class LevelsDirective(SphinxDirective):
    """List the levels: ``.. shapiq-ai-levels::``, as a table with the line at every level."""

    option_spec: ClassVar[dict[str, Any]] = {}

    def run(self) -> list[nodes.Node]:
        """Return a table of the levels: the line, the number and name, what it means."""
        self.env.note_dependency(str(Path(self.env.srcdir) / _LEVELS))
        levels = _levels(self.env.srcdir)
        top = levels[-1]["level"]
        table = nodes.table(classes=["shapiq-ai-levels"])
        group = nodes.tgroup(cols=3)
        table += group
        for width in (1, 3, 12):
            group += nodes.colspec(colwidth=width)
        head = nodes.thead()
        row = nodes.row()
        for title in ("", "Level", "What it means"):
            row += nodes.entry("", nodes.paragraph("", title))
        head += row
        group += head
        body = nodes.tbody()
        for level in levels:
            row = nodes.row()
            row += nodes.entry("", LevelGlyphNode(level=level["level"], top=top))
            name = nodes.paragraph()
            name += nodes.strong("", f"{level['level']} \N{MIDDLE DOT} {level['name']}")
            row += nodes.entry("", name)
            row += nodes.entry("", nodes.paragraph("", level["summary"]))
            body += row
        group += body
        return [table]


def _visit_glyph(self: HTML5Translator, node: LevelGlyphNode) -> None:
    self.body.append(_glyph(node["level"], node["top"]))
    raise nodes.SkipNode


def _add_assets(
    app: Sphinx,
    pagename: str,  # noqa: ARG001
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],  # noqa: ARG001
    doctree: nodes.document | None,
) -> None:
    """Add the line's stylesheet and script to a page that reports its AI involvement."""
    if doctree is None:
        return
    if next(iter(doctree.findall(InvolvementNode)), None) is not None:
        app.add_css_file("css/ai-involvement.css", priority=900)
        app.add_js_file("js/ai-involvement.js", loading_method="defer")
    elif next(iter(doctree.findall(LevelGlyphNode)), None) is not None:
        app.add_css_file("css/ai-involvement.css", priority=900)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the directives, their nodes and the assets."""
    skip = (_skip, None)
    app.add_node(
        InvolvementNode,
        html=(_visit_involvement, None),
        latex=skip,
        text=skip,
        man=skip,
        texinfo=skip,
    )
    app.add_node(
        LevelGlyphNode, html=(_visit_glyph, None), latex=skip, text=skip, man=skip, texinfo=skip
    )
    app.add_directive("shapiq-ai-involvement", InvolvementDirective)
    app.add_directive("shapiq-ai-levels", LevelsDirective)
    app.connect("doctree-resolved", _resolve_targets)
    app.connect("html-page-context", _add_assets)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
