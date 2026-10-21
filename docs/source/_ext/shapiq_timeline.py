"""Sphinx extension for the timeline of Shapley-based methods (``timeline.rst``).

The ``shapiq-timeline`` directive reads the lanes and entries of a TOML file
(``_data/timeline.toml``) and renders one card per entry, grouped by year. The texts of a card are
reStructuredText parsed by Sphinx, so its API cross-references (``:py:obj:``), gallery links
(``:doc:``) and citations (``:cite:``) are resolved and checked like everywhere else in the
docs. A missing API object warns even though the docs are not built in nitpicky mode, and every
warning points to the entry in the TOML file.

``_static/js/timeline.js`` adds filters and turns the list into swimlanes or a table by lane, and
``_static/css/timeline.css`` styles the views; they are only added to pages with a timeline.
Without JavaScript, the page is a list of all cards with all their details.
"""

from __future__ import annotations

import json
import re
import tomllib
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docutils import nodes
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective, switch_source_input

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata
    from sphinx.writers.html5 import HTML5Translator

logger = logging.getLogger(__name__)

_REQUIRED = {"id", "name", "year", "lane", "summary", "description"}
_OPTIONAL = {"cite", "api", "examples", "links", "related"}
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MAX_LANES = 9  # the eight colors of the categorical palette in timeline.css and a neutral one


class TimelineNode(nodes.General, nodes.Element):
    """The timeline: a legend of the lanes and the years with their cards."""


class TimelineYearNode(nodes.General, nodes.Element):
    """The cards of one year."""


class TimelineCardNode(nodes.General, nodes.Element):
    """One card; its children are the parsed summary and details."""


def _column_labels(years: set[int], eras: list[list[int]]) -> dict[int, str]:
    """Return the swimlane column of every year: the years of its era (e.g. 1981-2000) or itself."""
    labels = {year: str(year) for year in years}
    for start, end in eras:
        inside = sorted(year for year in years if start <= year <= end)
        if inside:
            label = f"{inside[0]}\N{EN DASH}{inside[-1]}" if len(inside) > 1 else str(inside[0])
            labels.update(dict.fromkeys(inside, label))
    return labels


def _entry_lines(text: str) -> dict[str, int]:
    """Return the line of the ``id`` field of every entry in the TOML source (for warnings)."""
    lines, in_entries = {}, False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("[["):
            in_entries = line.strip() == "[[entries]]"
        elif in_entries and (match := re.match(r'id\s*=\s*"([^"]*)"', line)):
            lines[match.group(1)] = number
    return lines


def _problems(entry: dict[str, Any], lanes: dict[str, Any], seen: set[str]) -> list[str]:
    """Return what is wrong with an entry: missing or unknown fields, a bad id or lane."""
    problems = [f"missing field {key!r}" for key in sorted(_REQUIRED - entry.keys())]
    problems += [f"unknown field {key!r}" for key in sorted(entry.keys() - _REQUIRED - _OPTIONAL)]
    if "id" in entry and not _ID.fullmatch(entry["id"]):
        problems.append(f"id {entry['id']!r} is not lowercase-with-dashes")
    if entry.get("id") in seen:
        problems.append(f"duplicate id {entry['id']!r}")
    if "lane" in entry and entry["lane"] not in lanes:
        problems.append(f"unknown lane {entry['lane']!r}")
    if "year" in entry and not isinstance(entry["year"], int):
        problems.append(f"year {entry['year']!r} is not a number")
    for key in ("cite", "api", "examples", "related"):
        if key in entry and not (
            isinstance(entry[key], list) and all(isinstance(item, str) for item in entry[key])
        ):
            problems.append(f"{key!r} is not a list of strings")
    return problems


def _api_reference(target: str) -> str:
    """Return a cross-reference showing the target from its class on, e.g. ``ExactComputer.x``."""
    parts = target.split(".")
    first = next((i for i, part in enumerate(parts) if part[:1].isupper()), len(parts) - 1)
    return f":py:obj:`{'.'.join(parts[first:])} <{target}>`"


def _details_rst(entry: dict[str, Any], names: dict[str, str]) -> str:
    """Return the reStructuredText of a card's details: its description and a field list."""
    fields = []
    if cite := entry.get("cite"):
        label = "Reference" if len(cite) == 1 else "References"
        fields.append((label, f":cite:p:`{','.join(cite)}`"))  # [Author et al., 2022], as elsewhere
    if api := entry.get("api"):
        fields.append(("In shapiq", ", ".join(_api_reference(target) for target in api)))
    if examples := entry.get("examples"):
        label = "Example" if len(examples) == 1 else "Examples"
        fields.append((label, ", ".join(f":doc:`/{doc}`" for doc in examples)))
    if links := entry.get("links"):
        fields.append(("Links", ", ".join(f"`{x['label']} <{x['url']}>`__" for x in links)))
    if related := [key for key in entry.get("related", []) if key in names]:
        fields.append(("Related", ", ".join(f"`{names[key]} <#{key}>`__" for key in related)))
    description = entry["description"].strip()
    if not fields:  # an empty rst-class would pass its class on to the next card
        return f"{description}\n"
    field_list = "\n".join(f":{label}: {body}" for label, body in fields)
    return f"{description}\n\n.. rst-class:: tl-card-fields\n\n{field_list}\n"


class TimelineDirective(SphinxDirective):
    """Render the timeline of a TOML file: ``.. shapiq-timeline:: _data/timeline.toml``."""

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        """Return the timeline node with one year node per year and one card per entry."""
        rel_path, path = self.env.relfn2path(self.arguments[0])
        self.env.note_dependency(rel_path)
        try:
            text = Path(path).read_text(encoding="utf-8")
            data = tomllib.loads(text)
        except (OSError, tomllib.TOMLDecodeError) as error:
            msg = f"cannot read the timeline {rel_path}: {error}"
            raise self.error(msg) from error

        lanes = {lane["id"]: {**lane, "slot": slot} for slot, lane in enumerate(data["lanes"], 1)}
        if len(lanes) > _MAX_LANES:
            logger.warning("the timeline has more than %d lanes", _MAX_LANES, location=f"{path}:1")
        lines, entries, seen = _entry_lines(text), [], set()
        for entry in data["entries"]:
            problems = _problems(entry, lanes, seen)
            for problem in problems:
                location = f"{path}:{lines.get(entry.get('id', ''), 1)}"
                logger.warning("timeline entry %s", problem, location=location)
            seen.add(entry.get("id", ""))
            if not problems:
                entries.append(entry)
        lane_order = list(lanes)
        entries.sort(key=lambda e: (e["year"], lane_order.index(e["lane"]), e["name"].casefold()))

        names = {entry["id"]: entry["name"] for entry in entries}
        years = sorted({entry["year"] for entry in entries})
        columns = _column_labels(set(years), data.get("eras", []))
        timeline = TimelineNode(
            lanes=[
                {key: lane[key] for key in ("id", "name", "description", "slot")}
                for lane in lanes.values()
            ],
            columns=list(dict.fromkeys(columns[year] for year in years)),
        )
        for year in years:
            year_node = TimelineYearNode(year=year)
            for entry in (entry for entry in entries if entry["year"] == year):
                source = (path, lines.get(entry["id"], 1))
                year_node += self._card(entry, lanes[entry["lane"]], columns[year], names, source)
            timeline += year_node
        return [timeline]

    def _card(
        self,
        entry: dict[str, Any],
        lane: dict[str, Any],
        column: str,
        names: dict[str, str],
        source: tuple[str, int],
    ) -> TimelineCardNode:
        """Return the card of an entry with its parsed summary and details."""
        for key in entry.get("related", []):
            if key not in names:
                logger.warning("unknown related entry %r", key, location="{}:{}".format(*source))
        card = TimelineCardNode(
            ids=[entry["id"]],
            name=entry["name"],
            year=entry["year"],
            column=column,
            lane=lane["id"],
            lane_name=lane["name"],
            slot=lane["slot"],
            implemented=bool(entry.get("api")),
        )
        summary = self._parse(entry["summary"], source)
        details = self._parse(_details_rst(entry, names), source)
        card += nodes.container("", *summary, classes=["tl-card-summary"])
        card += nodes.container(  # the region that the card's toggle (timeline.js) expands
            "",
            nodes.container("", *details, classes=["tl-card-details-inner"]),
            classes=["tl-card-details"],
            ids=[f"{entry['id']}-details"],
        )
        return card

    def _parse(self, text: str, source: tuple[str, int]) -> list[nodes.Node]:
        """Parse reStructuredText; warnings point to ``source``, missing API objects warn too."""
        lines = text.strip().splitlines()
        path, line = source
        content = StringList(lines, items=[(path, line - 1)] * len(lines))  # 0-based lines
        with switch_source_input(self.state, content):  # report the TOML file, as autodoc does
            parsed = self.parse_text_to_nodes(content, offset=0)
        for node in parsed:
            for reference in node.findall(addnodes.pending_xref):
                reference["refwarn"] = True  # warn about a missing target, as in nitpicky mode
        return parsed


# -- HTML ------------------------------------------------------------------------------------------


def _visit_timeline(self: HTML5Translator, node: TimelineNode) -> None:
    data = {"data-lanes": json.dumps(node["lanes"]), "data-columns": json.dumps(node["columns"])}
    self.body.append(self.starttag(node, "div", CLASS="shapiq-timeline", **data))
    legend = "".join(
        f'<li data-slot="{lane["slot"]}"><span class="tl-legend-name">{escape(lane["name"])}'
        f"</span> {escape(lane['description'])}</li>"
        for lane in node["lanes"]
    )
    self.body.append(f'<ul class="tl-legend" aria-label="Lanes">{legend}</ul>\n')
    self.body.append('<div class="tl-list">\n')


def _depart_timeline(self: HTML5Translator, node: TimelineNode) -> None:  # noqa: ARG001
    self.body.append("</div>\n</div>\n")


def _visit_year(self: HTML5Translator, node: TimelineYearNode) -> None:
    self.body.append(self.starttag(node, "section", CLASS="tl-year", **{"data-year": node["year"]}))
    self.body.append(f'<h2 class="tl-year-heading">{node["year"]}</h2>\n')


def _depart_year(self: HTML5Translator, node: TimelineYearNode) -> None:  # noqa: ARG001
    self.body.append("</section>\n")


def _visit_card(self: HTML5Translator, node: TimelineCardNode) -> None:
    data = {
        "data-lane": node["lane"],
        "data-slot": node["slot"],
        "data-year": node["year"],
        "data-column": node["column"],
        "data-implemented": str(node["implemented"]).lower(),
    }
    self.body.append(self.starttag(node, "article", CLASS="tl-card", **data))
    entry_id, name = node["ids"][0], escape(node["name"])
    header = [
        '<header class="tl-card-header"><p class="tl-card-meta">',
        f'<span class="tl-card-year">{node["year"]}</span>',
        f'<span class="tl-card-lane">{escape(node["lane_name"])}</span></p>',
        f'<h3 class="tl-card-title"><span class="tl-card-name">{name}</span>',
        f'<a class="headerlink" href="#{entry_id}" title="Link to {name}">#</a></h3>',
    ]
    if node["implemented"]:
        header.append(
            '<p class="tl-badges"><span class="tl-badge tl-badge-shapiq">In shapiq</span></p>'
        )
    self.body.append("".join(header) + "</header>\n")


def _depart_card(self: HTML5Translator, node: TimelineCardNode) -> None:  # noqa: ARG001
    self.body.append("</article>\n")


def _add_assets(
    app: Sphinx,
    pagename: str,  # noqa: ARG001
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],  # noqa: ARG001
    doctree: nodes.document | None,
) -> None:
    """Add the stylesheet and the script to the pages with a timeline."""
    if doctree is not None and next(doctree.findall(TimelineNode), None) is not None:
        app.add_css_file("css/timeline.css", priority=900)  # after html_css_files (tokens.css)
        app.add_js_file("js/timeline.js", loading_method="defer")


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the directive, its nodes and the assets."""
    app.add_node(TimelineNode, html=(_visit_timeline, _depart_timeline))
    app.add_node(TimelineYearNode, html=(_visit_year, _depart_year))
    app.add_node(TimelineCardNode, html=(_visit_card, _depart_card))
    app.add_directive("shapiq-timeline", TimelineDirective)
    app.connect("html-page-context", _add_assets)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
