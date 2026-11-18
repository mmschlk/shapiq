"""Sphinx extension for the AI involvement of a page: a thin line beside the text, part by part.

The levels are in ``_data/ai-involvement.toml``, from 0 (a person wrote it without AI) to 4 (AI
wrote it and no person went through all of it). A page reports its parts in a TOML file next to it and names
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
caption number), ``models``, ``human``, ``sources``, ``checked`` and ``note`` are optional.
``quoted = true`` marks a part taken from the work of others: it has no level and keeps the place
of the part around it, in gray. A
``[defaults]`` table gives every part the fields it does not set itself. A selector that matches
several elements, e.g. ``"figure.fixlip-figure"``, makes each of them a part. A part inside another
part, e.g. a figure inside a section, shows its own level where it is. Of two parts on the same
element, the later one counts. A target given as an id that the page does not have warns.

The directive writes the levels, the page's parts and the URL of the page ``ai-involvement`` as
JSON into the page, and adds ``_static/js/ai-involvement.js``, which draws the line, and
``_static/css/ai-involvement.css``. Outside HTML, it writes nothing.

For the page that explains them, the ``shapiq-ai-levels`` directive lists the levels as cards,
and ``shapiq-ai-figure`` draws the line beside a page (``page``) or the levels on their two inputs
(``levels``), with its content as the caption.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tomllib
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
_PAGE = {"level", "summary"}, {"written", "models", "human", "checked", "note", "mixer"}
_PART = (
    {"target", "written"},
    {"level", "name", "models", "human", "sources", "checked", "note", "quoted", "kind", "by"},
)
_KIND = "prose"  # the kind of a part that names none
_DEFAULTS = set(), (_PART[1] | {"written"}) - {"name"}


class InvolvementNode(nodes.General, nodes.Element):
    """The data of the line: the levels and the page's parts, written as JSON."""


class AboutNode(nodes.General, nodes.Element):
    """A drawing of the line or the cards of the levels, for the page that explains them."""


def _scale(srcdir: str) -> dict[str, Any]:
    """Return the levels and the steps of the two axes (``human``, ``ai``), each in order."""
    with (Path(srcdir) / _LEVELS).open("rb") as file:
        data = tomllib.load(file)
    return {
        "levels": sorted(data["levels"], key=lambda level: level["level"]),
        "human": sorted(data.get("human", []), key=lambda step: step["step"]),
        "ai": sorted(data.get("ai", []), key=lambda step: step["step"]),
        "kinds": {kind["kind"]: kind for kind in data.get("kinds", [])},
    }


def _place(
    part: dict[str, Any], kinds: dict[str, Any], top: int
) -> tuple[dict[str, Any], list[str]]:
    """Return the part with its place, its level and its answers, and what is wrong with it.

    Each question of the part's kind takes the level given in ``by``, or else the part's
    ``level``. The place is the weighted mean of the answers, from 0 to 1, and the level the
    nearest one (a tie goes to the higher level). A quoted part has neither.
    """
    if part.get("quoted"):
        return part, []
    kind = kinds.get(part.get("kind", _KIND))
    if kind is None:
        return part, [f"unknown kind {part.get('kind')!r}, use one of {', '.join(kinds)}"]
    by = part.get("by", {})
    keys = {criterion["key"] for criterion in kind["criteria"]}
    problems = [
        f"kind {kind['kind']!r} asks no question {key!r}" for key in sorted(by.keys() - keys)
    ]
    answers, total, weights = [], 0, 0
    for criterion in kind["criteria"]:
        level = by.get(criterion["key"], part.get("level"))
        if not (isinstance(level, int) and 0 <= level <= top):
            problems.append(f"no level from 0 to {top} for the question {criterion['key']!r}")
            continue
        answers.append({**criterion, "level": level, "given": criterion["key"] in by})
        total += criterion["weight"] * level
        weights += criterion["weight"]
    if problems or not weights:
        return part, problems
    place = total / (top * weights)
    level = (2 * top * total + top * weights) // (2 * top * weights)  # floor(top * place + 1/2)
    return {**part, "level": level, "place": place, "answers": answers, "kind": kind["kind"]}, []


def _levels(srcdir: str) -> list[dict[str, Any]]:
    """Return the levels, in order."""
    return _scale(srcdir)["levels"]


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
        scale = _scale(self.env.srcdir)
        top = scale["levels"][-1]["level"]
        page = data.get("page", {})
        problems = [f"[page]: {p}" for p in _problems(page, _PAGE, top)]
        if isinstance(page.get("level"), int):
            page = {**page, "place": page["level"] / top}
        defaults = data.get("defaults", {})
        problems += [f"[defaults]: {p}" for p in _problems(defaults, _DEFAULTS, top)]
        parts = []
        for number, own in enumerate(data.get("parts", []), start=1):
            part = {**defaults, **own}
            # the answers of the defaults hold only for parts of the same kind
            same = own.get("kind", defaults.get("kind", _KIND)) == defaults.get("kind", _KIND)
            part["by"] = {**(defaults.get("by", {}) if same else {}), **own.get("by", {})}
            if not part["by"]:
                del part["by"]
            label = part.get("target", f"part {number}")
            found = _problems(part, _PART, top)
            if not found:
                part, found = _place(part, scale["kinds"], top)
                if "by" in own and "level" in own and found == [] and part["level"] != own["level"]:
                    found = [
                        f"the answers give level {part['level']}, not the stated {own['level']}"
                    ]
            problems += [f"[[parts]] {label}: {p}" for p in found]
            parts.append(part)
        for problem in problems:
            logger.warning("%s: %s", path.name, problem, location=self.get_location())
        node = InvolvementNode(page=_plain(page), parts=_plain(parts))
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
        **_scale(self.builder.srcdir),
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


def _hue(place: float) -> str:
    """Return the hue of the line's palette "spectrum" at a place from 0 to 1 (as in the JS)."""
    return f"{160 - 152 * place:.0f}"


def _places(scale: dict[str, Any]) -> tuple[int, list[tuple[dict[str, Any], int, int, float]]]:
    """Return the top step of the two axes and each level with its two inputs and its place."""
    top = max(len(scale["human"]), len(scale["ai"]), 2) - 1
    rows = []
    for level in scale["levels"]:
        h, a = level.get("human", 0), level.get("ai", 0)
        rows.append((level, h, a, (a - h + top) / (2 * top)))
    return top, rows


def _esc(text: Any) -> str:  # noqa: ANN401
    """Return text escaped for HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mini(place: float, lanes: int) -> str:
    """Return the line at a place (0 to 1) as a small SVG: faint lanes, the shade and the line."""
    width, height = 44, 26
    step = (width - 4) / (lanes - 1)
    drawn = "".join(
        f'<line class="lane{" mid" if 2 * i == lanes - 1 else ""}" x1="{2 + i * step:.2f}"'
        f' x2="{2 + i * step:.2f}" y1="2" y2="{height - 2}"/>'
        for i in range(lanes)
    )
    x = 2 + place * (width - 4)
    return (
        f'<svg class="shapiq-ai-mini" viewBox="0 0 {width} {height}" width="{width}"'
        f' height="{height}" aria-hidden="true">{drawn}'
        f'<rect class="shade" x="0" y="0" width="{x:.2f}" height="{height}"/>'
        f'<line class="halo" x1="{x:.2f}" x2="{x:.2f}" y1="2" y2="{height - 2}"/>'
        f'<line class="line" x1="{x:.2f}" x2="{x:.2f}" y1="2" y2="{height - 2}"/></svg>'
    )


def _cards(scale: dict[str, Any]) -> str:
    """Return the levels as cards: the line at its place, the name, the two inputs, the meaning."""
    top, rows = _places(scale)
    human = {step["step"]: step["name"] for step in scale["human"]}
    ai = {step["step"]: step["name"] for step in scale["ai"]}

    def pips(value: int) -> str:
        return "".join(f'<i class="{"on" if i < value else ""}"></i>' for i in range(top))

    cards = []
    for level, h, a, place in rows:
        cards.append(
            f'<li class="shapiq-ai-level" style="--shapiq-ai-h: {_hue(place)}">'
            f'<div class="shapiq-ai-level-head">{_mini(place, 2 * top + 1)}'
            f'<span class="shapiq-ai-level-name">{_esc(level["name"])}</span></div>'
            f'<dl class="shapiq-ai-level-inputs">'
            f'<dt>Person</dt><dd><span class="pips">{pips(h)}</span>{_esc(human.get(h, h))}</dd>'
            f'<dt>AI</dt><dd><span class="pips">{pips(a)}</span>{_esc(ai.get(a, a))}</dd></dl>'
            f'<p class="shapiq-ai-level-summary">{_esc(level["summary"])}</p></li>'
        )
    return f'<ol class="shapiq-ai-levels">{"".join(cards)}</ol>\n'


def _curve(points: list[tuple[float, float, float]], bend: float) -> str:
    """Return an SVG path down through stretches (x, top, bottom), curving between them."""
    x, y0, _ = points[0]
    path = [f"M{x:.1f},{y0:.1f}"]
    for (x1, _, b1), (x2, t2, _) in pairwise(points):
        mid = (b1 + t2) / 2
        path.append(f"L{x1:.1f},{mid - bend:.1f}")
        path.append(f"C{x1:.1f},{mid:.1f} {x2:.1f},{mid:.1f} {x2:.1f},{mid + bend:.1f}")
    path.append(f"L{points[-1][0]:.1f},{points[-1][2]:.1f}")
    return " ".join(path)


def _figure_page(scale: dict[str, Any]) -> str:
    """Return a drawing of the line beside a page, with numbered marks for the list beside it."""
    top, rows = _places(scale)
    place = {level["level"]: p for level, _, _, p in rows}
    name = {level["level"]: level["name"] for level, _, _, _ in rows}
    lanes, left, span = 2 * top + 1, 14.0, 72.0

    def at(level: int) -> float:
        return left + place[level] * span

    # the parts of the drawn page: top, bottom and level
    parts = [(12, 94, 3), (104, 178, 4), (188, 240, 1), (250, 296, 3)]
    stretches = [(at(level), t, b) for t, b, level in parts]
    line = _curve(stretches, 9)
    stops = []
    for (_, b1, l1), (t2, _, l2) in pairwise(parts):
        mid = (b1 + t2) / 2
        stops.append((mid - 9, l1))
        stops.append((mid + 9, l2))
    stops = [(parts[0][0], parts[0][2]), *stops, (parts[-1][1], parts[-1][2])]
    gradient = "".join(
        f'<stop offset="{y / 310:.4f}" style="--shapiq-ai-h: {_hue(place[level])}"/>'
        for y, level in stops
    )
    out = [
        (
            '<svg class="shapiq-ai-drawing" viewBox="0 0 400 316" role="img"'
            ' aria-label="A page with the line on its left">'
            '<defs><linearGradient id="shapiq-ai-fig-down" gradientUnits="userSpaceOnUse"'
            f' x1="0" y1="0" x2="0" y2="310">{gradient}</linearGradient>'
            '<linearGradient id="shapiq-ai-fig-across" x1="0" x2="1" y1="0" y2="0">'
            '<stop offset="0" stop-color="#fff" stop-opacity="0.02"/>'
            '<stop offset="1" stop-color="#fff" stop-opacity="0.22"/></linearGradient>'
            '<mask id="shapiq-ai-fig-fade" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="316">'
            '<rect x="0" y="0" width="100" height="316" fill="url(#shapiq-ai-fig-across)"/></mask>'
            '<filter id="shapiq-ai-fig-blur" x="-50%" y="-5%" width="200%" height="110%">'
            '<feGaussianBlur stdDeviation="2"/></filter>'
            f'<linearGradient id="shapiq-ai-fig-tint" x1="0" x2="1" y1="0" y2="0"'
            f' style="--shapiq-ai-h: {_hue(place[4])}">'
            '<stop offset="0" class="tint" stop-opacity="0.2"/>'
            '<stop offset="0.25" class="tint" stop-opacity="0.07"/>'
            '<stop offset="1" class="tint" stop-opacity="0"/></linearGradient></defs>'
        )
    ]
    # the lanes, the middle one dashed
    for k in range(lanes):
        x = left + k * span / (lanes - 1)
        out.append(
            f'<line class="lane{" mid" if 2 * k == lanes - 1 else ""}" x1="{x:.1f}" x2="{x:.1f}"'
            ' y1="10" y2="298"/>'
        )
    out.append(
        f'<text class="axis" x="{left - 4}" y="311" text-anchor="start">Person</text>'
        f'<text class="axis" x="{left + span + 4}" y="311" text-anchor="end">AI</text>'
    )
    # the tint of the part pointed at (the figure)
    out.append(
        '<rect x="98" y="100" width="300" height="82" rx="6" fill="url(#shapiq-ai-fig-tint)"/>'
    )
    # the page: bars of text, a figure, a definition
    text = [(0, 18, 150, 9, "head")]
    for i, w in enumerate((282, 270, 288, 252, 180)):
        text.append((0, 38 + 12 * i, w, 5, ""))
    for i, w in enumerate((286, 274, 240)):
        text.append((0, 256 + 12 * i, w, 5, ""))
    for _, y, w, hgt, cls in text:
        out.append(f'<rect class="bar {cls}" x="104" y="{y}" width="{w}" height="{hgt}" rx="2.5"/>')
    out.append('<rect class="box" x="132" y="108" width="236" height="62" rx="5"/>')
    for i, hgt in enumerate((18, 30, 24, 40, 34, 46, 28, 38, 22, 44)):
        out.append(
            f'<rect class="bars" x="{146 + 21 * i}" y="{162 - hgt}" width="12" height="{hgt}" rx="1.5"/>'
        )
    out.append('<rect class="def" x="104" y="190" width="288" height="48" rx="5"/>')
    out.append('<rect class="def-edge" x="104" y="190" width="3" height="48" rx="1.5"/>')
    for i, w in enumerate((90, 250, 230)):
        out.append(
            f'<rect class="bar{" head" if i == 0 else ""}" x="118" y="{198 + 12 * i}"'
            f' width="{w}" height="{6 if i == 0 else 5}" rx="2.5"/>'
        )
    # the line: its shade, its glow and itself
    out.append(
        f'<path class="shade" d="{line} L4,296 L4,12 Z" fill="url(#shapiq-ai-fig-down)"'
        ' mask="url(#shapiq-ai-fig-fade)"/>'
        f'<path class="halo" d="{line}" stroke="url(#shapiq-ai-fig-down)" filter="url(#shapiq-ai-fig-blur)"/>'
        f'<path class="line" d="{line}" stroke="url(#shapiq-ai-fig-down)"/>'
    )
    # the dot under the pointer, on the figure's stretch
    out.append(
        f'<circle class="dot" cx="{at(4):.1f}" cy="142" r="3.6" style="--shapiq-ai-h: {_hue(place[4])}"/>'
    )
    # the readout beside the dot: a small scale with the part's place, and its kind and level
    rx, ry = at(4) + 9, 134
    out.append(
        f'<g class="readout"><rect x="{rx:.1f}" y="{ry}" width="112" height="16" rx="8"/>'
        f'<line class="base" x1="{rx + 7:.1f}" x2="{rx + 39:.1f}" y1="{ry + 8}" y2="{ry + 8}"/>'
        + "".join(
            f'<line class="tick" x1="{rx + 7 + 8 * k:.1f}" x2="{rx + 7 + 8 * k:.1f}"'
            f' y1="{ry + 5.5}" y2="{ry + 10.5}"/>'
            for k in range(top * 2 + 1)
        )
        + f'<circle class="place" cx="{rx + 39:.1f}" cy="{ry + 8}" r="2.6"'
        f' style="--shapiq-ai-h: {_hue(place[4])}"/>'
        f'<text x="{rx + 45:.1f}" y="{ry + 10.8}">Figure · {_esc(name[4])}</text></g>'
    )
    # the tag, which names the level of the part being read
    mid = left + span / 2
    out.append(
        f'<g class="tag" style="--shapiq-ai-h: {_hue(place[3])}">'
        f'<rect x="{mid - 7.5}" y="22" width="15" height="62" rx="7.5"/>'
        f'<text transform="translate({mid + 2.6},53) rotate(90)" text-anchor="middle">'
        f"{_esc(name[3])}</text></g>"
    )
    # the numbered marks of the list beside the drawing
    for number, (x, y) in enumerate(((mid, 12), (95, 99), (at(4) + 130, 142), (95, 214)), start=1):
        out.append(
            f'<g class="mark"><circle cx="{x:.1f}" cy="{y}" r="6.5"/>'
            f'<text x="{x:.1f}" y="{y + 3}" text-anchor="middle">{number}</text></g>'
        )
    out.append("</svg>")
    return "".join(out)


def _figure_map(scale: dict[str, Any]) -> tuple[str, str]:
    """Return the levels on the two inputs (a grid) and at their place on the line (a rail)."""
    top, rows = _places(scale)
    human = {step["step"]: step["name"] for step in scale["human"]}
    ai = {step["step"]: step["name"] for step in scale["ai"]}
    cell, x0, y0 = 174 / top, 92, 34

    def gx(a: int) -> float:
        return x0 + a * cell

    def gy(h: int) -> float:
        return y0 + (top - h) * cell

    size = top * cell
    out = [
        (
            f'<svg class="shapiq-ai-drawing shapiq-ai-grid" viewBox="0 0 {x0 + size + 78} {y0 + size + 52}"'
            ' role="img" aria-label="The levels on a grid of what a person did and what AI did">'
        )
    ]
    # the balance of the two inputs: one diagonal per place of the line
    for k in range(-top, top + 1):
        a1, a2 = max(0, k), min(top, top + k)
        p = (k + top) / (2 * top)
        out.append(
            f'<line class="diagonal{" mid" if k == 0 else ""}" style="--shapiq-ai-h: {_hue(p)}"'
            f' x1="{gx(a1)}" y1="{gy(a1 - k)}" x2="{gx(a2)}" y2="{gy(a2 - k)}"/>'
        )
    steps = range(top + 1)
    out.extend(
        f'<circle class="node" cx="{gx(a)}" cy="{gy(h)}" r="1.6"/>' for a in steps for h in steps
    )
    out.extend(
        f'<text class="step" x="{gx(a)}" y="{y0 + size + 20}" text-anchor="middle">'
        f"{_esc(ai.get(a, a))}</text>"
        for a in steps
    )
    out.extend(
        f'<text class="step" x="{x0 - 12}" y="{gy(h) + 3.5}" text-anchor="end">'
        f"{_esc(human.get(h, h))}</text>"
        for h in steps
    )
    out.append(
        f'<text class="axis" x="{x0 + size / 2}" y="{y0 + size + 42}" text-anchor="middle">'
        "What AI did</text>"
        f'<text class="axis" transform="translate(14,{y0 + size / 2}) rotate(-90)"'
        ' text-anchor="middle">What a person did</text>'
    )
    # the path of the levels, from a person alone to AI alone
    for (_, h1, a1, p1), (_, h2, a2, p2) in pairwise(rows):
        gid = f"shapiq-ai-map-{a1}{h1}"
        out.append(
            f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{gx(a1)}" y1="{gy(h1)}"'
            f' x2="{gx(a2)}" y2="{gy(h2)}"><stop offset="0" style="--shapiq-ai-h: {_hue(p1)}"/>'
            f'<stop offset="1" style="--shapiq-ai-h: {_hue(p2)}"/></linearGradient>'
            f'<line class="path" x1="{gx(a1)}" y1="{gy(h1)}" x2="{gx(a2)}" y2="{gy(h2)}"'
            f' stroke="url(#{gid})"/>'
        )
    for level, h, a, p in rows:
        x, y = gx(a), gy(h)
        out.append(
            f'<g class="point" style="--shapiq-ai-h: {_hue(p)}"><circle cx="{x}" cy="{y}" r="8"/>'
            f'<text class="number" x="{x}" y="{y + 3.4}" text-anchor="middle">{level["level"]}</text>'
        )
        if h == top and a < top:
            out.append(
                f'<text x="{x}" y="{y - 14}" text-anchor="middle">{_esc(level["name"])}</text>'
            )
        else:
            out.append(f'<text x="{x + 14}" y="{y + 3.5}">{_esc(level["name"])}</text>')
        out.append("</g>")
    out.append("</svg>")
    grid = "".join(out)

    # the rail: the lanes of the line, each level at its place
    lanes, width, left = 2 * top + 1, 320, 40
    step = (width - 2 * left) / (lanes - 1)
    out = [
        f'<svg class="shapiq-ai-drawing shapiq-ai-rail" viewBox="0 0 {width} 132" role="img"'
        ' aria-label="The levels at their places on the line, from a person alone to AI alone">'
        f'<linearGradient id="shapiq-ai-rail-band" x1="0" x2="1" y1="0" y2="0">'
        + "".join(
            f'<stop offset="{k / (lanes - 1):.3f}" style="--shapiq-ai-h: {_hue(k / (lanes - 1))}"/>'
            for k in range(lanes)
        )
        + "</linearGradient>"
        f'<rect class="band" x="{left - 8}" y="22" width="{width - 2 * left + 16}" height="58" rx="8"'
        ' fill="url(#shapiq-ai-rail-band)"/>'
    ]
    for k in range(lanes):
        x = left + k * step
        out.append(
            f'<line class="lane{" mid" if 2 * k == lanes - 1 else ""}" x1="{x:.1f}" x2="{x:.1f}"'
            ' y1="26" y2="76"/>'
        )
    for i, (level, _, _, p) in enumerate(rows):
        x = left + p * (lanes - 1) * step
        y = 98 if i % 2 == 0 else 112
        out.append(
            f'<g class="point" style="--shapiq-ai-h: {_hue(p)}">'
            f'<line class="halo" x1="{x:.1f}" x2="{x:.1f}" y1="30" y2="72"/>'
            f'<line class="line" x1="{x:.1f}" x2="{x:.1f}" y1="30" y2="72"/>'
            f'<circle cx="{x:.1f}" cy="14" r="8"/>'
            f'<text class="number" x="{x:.1f}" y="17.4" text-anchor="middle">{level["level"]}</text>'
            f'<line class="tick" x1="{x:.1f}" x2="{x:.1f}" y1="80" y2="{y - 9}"/>'
            f'<text x="{x:.1f}" y="{y}" text-anchor="middle">{_esc(level["name"])}</text></g>'
        )
    out.append(
        f'<text class="axis" x="{left - 8}" y="130" text-anchor="start">A person did it all</text>'
        f'<text class="axis" x="{width - left + 8}" y="130" text-anchor="end">AI did it all</text>'
        "</svg>"
    )
    rail = "".join(out)
    return grid, rail


def _figure_kinds(scale: dict[str, Any]) -> str:
    """Return a figure's answers to the questions of its kind, joined into one place."""
    levels = scale["levels"]
    top = levels[-1]["level"]
    kind = scale["kinds"]["figure"]
    example = {"design": 3, "code": 4, "check": 3}  # as the figures of the FIxLIP page
    left, right, base = 112, 404, 104

    def x(t: float) -> float:
        return left + t * (right - left)

    total = sum(c["weight"] * example[c["key"]] for c in kind["criteria"])
    weights = sum(c["weight"] for c in kind["criteria"])
    place = total / (top * weights)
    level = levels[int(top * place + 0.5)]
    out = [
        (
            '<svg class="shapiq-ai-drawing shapiq-ai-kinds-drawing" viewBox="0 0 430 150" role="img"'
            ' aria-label="The three answers of a figure and their weighted mean on the scale">'
        ),
        '<linearGradient id="shapiq-ai-kinds-band" x1="0" x2="1" y1="0" y2="0">'
        + "".join(
            f'<stop offset="{k / top:.3f}" style="--shapiq-ai-h: {_hue(k / top)}"/>'
            for k in range(top + 1)
        )
        + "</linearGradient>",
        (
            f'<rect class="band" x="{left - 6}" y="{base - 5}" width="{right - left + 12}"'
            ' height="10" rx="5" fill="url(#shapiq-ai-kinds-band)"/>'
        ),
    ]
    for k, entry in enumerate(levels):
        out.append(
            f'<line class="lane" x1="{x(k / top):.1f}" x2="{x(k / top):.1f}" y1="10"'
            f' y2="{base + 6}"/><text class="axis" x="{x(k / top):.1f}" y="{base + 22}"'
            f' text-anchor="middle">{_esc(entry["name"])}</text>'
        )
    for i, criterion in enumerate(kind["criteria"]):
        y = 22 + 22 * i
        t = example[criterion["key"]] / top
        out.append(
            f'<text x="{left - 14}" y="{y + 3.5}" text-anchor="end">{_esc(criterion["name"])}'
            f'<tspan class="weight"> counts {criterion["weight"]}</tspan></text>'
            f'<line class="join" x1="{x(t):.1f}" y1="{y + 4}" x2="{x(place):.1f}" y2="{base - 6}"/>'
            f'<circle class="answer" cx="{x(t):.1f}" cy="{y}" r="{2 + criterion["weight"] * 0.7:.1f}"/>'
        )
    out.append(
        f'<g class="point" style="--shapiq-ai-h: {_hue(place)}">'
        f'<circle cx="{x(place):.1f}" cy="{base}" r="6.5"/></g>'
        f'<text class="axis" x="{left - 14}" y="{base + 3.5}" text-anchor="end">Place</text>'
        f'<text class="result" x="{x(place):.1f}" y="{base + 40}" text-anchor="middle">'
        f"{_esc(level['name'])}, the nearest level</text></svg>"
    )
    return "".join(out)


_MARKS = (
    "The tag names the level of the part being read. It stays at the top while the page scrolls.",
    "Where a part changes, the line curves into the place of the new part.",
    (
        "Pointing at the line shows a dot on it and tints its part. A small scale beside the dot"
        " shows where the part sits. A click opens the card of the part."
    ),
    "A figure or a definition inside a section can have its own place. The line swings out and back.",
)


class FigureDirective(SphinxDirective):
    """Draw the line: ``.. shapiq-ai-figure:: page`` (on a page) or ``levels`` (the levels)."""

    required_arguments = 1
    has_content = True

    def run(self) -> list[nodes.Node]:
        """Return the drawing with its caption, the content of the directive."""
        self.env.note_dependency(str(Path(self.env.srcdir) / _LEVELS))
        scale = _scale(self.env.srcdir)
        kind = self.arguments[0]
        if kind == "page":
            items = "".join(f"<li>{_esc(text)}</li>" for text in _MARKS)
            body = (
                f'<div class="shapiq-ai-figure-page">{_figure_page(scale)}'
                f'<ol class="shapiq-ai-marks">{items}</ol></div>'
            )
        elif kind == "levels":
            grid, rail = _figure_map(scale)
            body = (
                '<div class="shapiq-ai-figure-levels">'
                f'<div class="shapiq-ai-panel"><p class="shapiq-ai-panel-title">Two inputs</p>{grid}</div>'
                '<div class="shapiq-ai-arrow" aria-hidden="true"></div>'
                f'<div class="shapiq-ai-panel"><p class="shapiq-ai-panel-title">One place</p>{rail}</div>'
                "</div>"
            )
        elif kind == "kinds":
            body = f'<div class="shapiq-ai-figure-kinds">{_figure_kinds(scale)}</div>'
        else:
            msg = f"unknown drawing {kind!r}, use 'page', 'levels' or 'kinds'"
            raise self.error(msg)
        figure = nodes.container(classes=["shapiq-ai-figure"])
        figure += AboutNode(html=body)
        if self.content:
            caption = nodes.container(classes=["shapiq-ai-figure-caption"])
            self.state.nested_parse(self.content, self.content_offset, caption)
            figure += caption
        return [figure]


class KindsDirective(SphinxDirective):
    """List the kinds of content and their questions: ``.. shapiq-ai-kinds::``, as a table."""

    def run(self) -> list[nodes.Node]:
        """Return a table with a row for each question of each kind."""
        self.env.note_dependency(str(Path(self.env.srcdir) / _LEVELS))
        table = nodes.table(classes=["shapiq-ai-kinds"])
        group = nodes.tgroup(cols=4)
        table += group
        for width in (2, 2, 8, 2):
            group += nodes.colspec(colwidth=width)
        head = nodes.thead()
        row = nodes.row()
        for title in ("Kind", "Question", "Asks", "Counts"):
            row += nodes.entry("", nodes.paragraph("", title))
        head += row
        group += head
        body = nodes.tbody()
        for kind in _scale(self.env.srcdir)["kinds"].values():
            for i, criterion in enumerate(kind["criteria"]):
                row = nodes.row()
                name = nodes.paragraph()
                if i == 0:
                    name += nodes.strong("", kind["name"])
                row += nodes.entry("", name)
                row += nodes.entry("", nodes.paragraph("", criterion["name"]))
                row += nodes.entry("", nodes.paragraph("", criterion["question"]))
                row += nodes.entry("", nodes.paragraph("", f"{criterion['weight']} of 10"))
                body += row
        group += body
        return [table]


class LevelsDirective(SphinxDirective):
    """List the levels: ``.. shapiq-ai-levels::``, as cards with the line at every level."""

    def run(self) -> list[nodes.Node]:
        """Return the cards of the levels."""
        self.env.note_dependency(str(Path(self.env.srcdir) / _LEVELS))
        return [AboutNode(html=_cards(_scale(self.env.srcdir)))]


def _visit_about(self: HTML5Translator, node: AboutNode) -> None:
    self.body.append(node["html"])
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
    elif next(iter(doctree.findall(AboutNode)), None) is not None:
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
        AboutNode, html=(_visit_about, None), latex=skip, text=skip, man=skip, texinfo=skip
    )
    app.add_directive("shapiq-ai-involvement", InvolvementDirective)
    app.add_directive("shapiq-ai-levels", LevelsDirective)
    app.add_directive("shapiq-ai-figure", FigureDirective)
    app.add_directive("shapiq-ai-kinds", KindsDirective)
    app.connect("doctree-resolved", _resolve_targets)
    app.connect("html-page-context", _add_assets)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
