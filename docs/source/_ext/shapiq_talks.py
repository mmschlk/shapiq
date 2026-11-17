"""Sphinx extension for talks (``talks/``): a page of slides that is also a slide deck.

A talk is a page of the docs, ``talks/<name>/index.rst``, made of slides. Read on the page, the
slides follow each other like the sections of a handout, each with its notes below it. Presented
(the button "Present", or the key ``P``), one slide fills the screen at a time, and the keyboard, a
presenter's clicker or the buttons in the corner move through them. ``_static/js/talk.js`` and
``_static/css/talk.css`` do both, on every page with slides.

The ``shapiq-slide`` directive makes one slide. Its argument is the slide's title, its content
the slide. Slides are numbered in the order of the page and get the id ``slide-<number>``, so that
``#slide-3`` links to the third slide, on the page and in the deck::

    .. shapiq-slide:: One Score per Feature
       :layout: wide
       :steps: 3

       The content: text, figures, demos (``shapiq-demo``) and figures of the talk.

       .. shapiq-notes::

          What to say. Shown below the slide on the page, and on the presenter's screen.

``:layout:`` is ``title`` (the talk's first slide), ``section`` (a divider with a large title),
``wide`` (a demo or a figure that needs the full width) or, without it, a slide with a title and
its content. ``:steps:`` is the number of steps a slide builds up in, one key press each. A part of
the slide with the class ``shapiq-from-<k>`` (``.. rst-class:: shapiq-from-2``) shows from step
``k`` on, counted from 0, and the slide's ``data-talk-step`` holds the step it is at, for the figures
of the talk to follow. Without ``:steps:``, a slide has as many steps as its parts ask for.

The ``shapiq-talk-figure`` directive adds a figure that the talk's own script draws, from
``talks/<name>/assets/figures.js`` and ``figures.css`` (copied to ``_static/talks/<name>/``)::

    .. shapiq-talk-figure:: same-shap
       :alt: What the figure shows, for screen readers.

It writes ``<div class="shapiq-talk-figure" data-figure="same-shap" data-root="...">``. Outside
HTML, slides keep their title and content, and notes and figures are left out.
"""

from __future__ import annotations

import shutil
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata
    from sphinx.writers.html5 import HTML5Translator

_LAYOUTS = ("title", "section", "wide")
_FIGURES = ("figures.js", "figures.css")  # the assets of a talk with figures of its own


def _talk(docname: str) -> str:
    """Return the name of the talk a page belongs to, its folder in ``talks/``."""
    parts = docname.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "talks" else ""


class SlideNode(nodes.General, nodes.Element):
    """A slide; its children are its content and its notes."""


class NotesNode(nodes.General, nodes.Element):
    """The notes of a slide."""


class TalkFigureNode(nodes.General, nodes.Element):
    """A figure that the talk's script draws."""


class SlideDirective(SphinxDirective):
    """Add a slide: ``.. shapiq-slide:: <title>``, with its content."""

    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = True
    option_spec: ClassVar[dict[str, Any]] = {
        "layout": lambda value: directives.choice(value, _LAYOUTS),
        "steps": directives.positive_int,
        "name": directives.unchanged_required,  # a second id, to link to the slide by name
        "class": directives.class_option,
    }

    def run(self) -> list[nodes.Node]:
        """Return the slide with its parsed content, numbered in the order of the page."""
        number = self.env.temp_data.get("shapiq_slides", 0) + 1
        self.env.temp_data["shapiq_slides"] = number
        node = SlideNode(
            title=self.arguments[0] if self.arguments else "",
            layout=self.options.get("layout", ""),
            steps=self.options.get("steps", 0),
            number=number,
        )
        node["classes"] += self.options.get("class", [])
        node["ids"].append(f"slide-{number}")
        if "name" in self.options:
            node["ids"].append(nodes.make_id(self.options["name"]))
        if node["title"]:
            title = nodes.paragraph(classes=["shapiq-slide-title"])
            title += self.parse_inline(node["title"], lineno=self.lineno)[0]
            node += title
        body = nodes.container(classes=["shapiq-slide-body"])
        self.state.nested_parse(self.content, self.content_offset, body)
        # the notes go after the body, wherever they were written
        notes = list(body.findall(NotesNode))
        for note in notes:
            note.parent.remove(note)
        node += body
        node.extend(notes)
        return [node]


class NotesDirective(SphinxDirective):
    """Add notes to a slide: ``.. shapiq-notes::``, with what to say."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        """Return the notes with their parsed content."""
        node = NotesNode()
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class TalkFigureDirective(SphinxDirective):
    """Add a figure of the talk: ``.. shapiq-talk-figure:: <name>``."""

    required_arguments = 1
    option_spec: ClassVar[dict[str, Any]] = {
        "alt": directives.unchanged_required,
        "class": directives.class_option,
    }

    def run(self) -> list[nodes.Node]:
        """Return the figure, after checking that the talk has the assets to draw it."""
        talk = _talk(self.env.docname)
        assets = Path(self.env.srcdir) / "talks" / talk / "assets"
        if not talk or (missing := [file for file in _FIGURES if not (assets / file).is_file()]):
            msg = f"the page {self.env.docname!r} is not a talk with {', '.join(_FIGURES)} in assets/"
            raise self.error(msg if not talk else f"{msg} (missing: {', '.join(missing)})")
        for file in _FIGURES:
            self.env.note_dependency(str(assets / file))
        node = TalkFigureNode(talk=talk, figure=self.arguments[0], alt=self.options.get("alt", ""))
        node["classes"] += self.options.get("class", [])
        return [node]


def _visit_slide(self: HTML5Translator, node: SlideNode) -> None:
    classes = ["shapiq-slide"] + ([f"shapiq-slide--{node['layout']}"] if node["layout"] else [])
    attributes = {"data-title": node["title"], "data-number": str(node["number"])}
    if node["steps"]:
        attributes["data-steps"] = str(node["steps"])
    self.body.append(self.starttag(node, "section", CLASS=" ".join(classes), **attributes))


def _depart_slide(self: HTML5Translator, node: SlideNode) -> None:  # noqa: ARG001
    self.body.append("</section>\n")


def _visit_notes(self: HTML5Translator, node: NotesNode) -> None:
    self.body.append(self.starttag(node, "aside", CLASS="shapiq-slide-notes"))


def _depart_notes(self: HTML5Translator, node: NotesNode) -> None:  # noqa: ARG001
    self.body.append("</aside>\n")


def _visit_figure(self: HTML5Translator, node: TalkFigureNode) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    root = relative_uri(page, f"_static/talks/{node['talk']}/")
    self.body.append(
        self.starttag(
            node,
            "div",
            CLASS="shapiq-talk-figure",
            role="img",
            **{"data-figure": node["figure"], "data-root": root, "aria-label": node["alt"]},
        )
    )
    self.body.append(f'<noscript><p>{escape(node["alt"])}</p></noscript></div>\n')
    raise nodes.SkipNode


def _visit_slide_other(self: nodes.NodeVisitor, node: SlideNode) -> None:
    """Keep a slide's content outside HTML (its title is a paragraph of it)."""


def _depart_slide_other(self: nodes.NodeVisitor, node: SlideNode) -> None:
    """Close a slide outside HTML."""


def _skip(self: nodes.NodeVisitor, node: nodes.Node) -> None:  # noqa: ARG001
    """Leave notes and figures out of a build that is not HTML."""
    raise nodes.SkipNode


def _add_assets(
    app: Sphinx,
    pagename: str,
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],  # noqa: ARG001
    doctree: nodes.document | None,
) -> None:
    """Add the deck's stylesheet and script, and the talk's figures, to every page with slides."""
    if doctree is None or next(iter(doctree.findall(SlideNode)), None) is None:
        return
    app.add_css_file("css/talk.css", priority=900)  # after html_css_files (tokens.css)
    app.add_js_file("js/talk.js", loading_method="defer", priority=600)  # after the demos' scripts
    talk = _talk(pagename)
    if talk and next(iter(doctree.findall(TalkFigureNode)), None) is not None:
        app.add_css_file(f"talks/{talk}/figures.css", priority=901)
        app.add_js_file(f"talks/{talk}/figures.js", loading_method="defer", priority=550)


def _copy_assets(app: Sphinx, env: object) -> None:  # noqa: ARG001
    """Copy every talk's assets to ``_static/talks/<name>/``, before the pages are written."""
    if app.builder.format != "html":
        return
    for assets in sorted((Path(app.srcdir) / "talks").glob("*/assets")):
        target = Path(app.outdir) / "_static" / "talks" / assets.parent.name
        shutil.copytree(assets, target, dirs_exist_ok=True)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the directives, their nodes and the assets."""
    other = (_visit_slide_other, _depart_slide_other)
    skip = (_skip, None)
    app.add_node(SlideNode, html=(_visit_slide, _depart_slide), latex=other, text=other, man=other, texinfo=other)
    app.add_node(NotesNode, html=(_visit_notes, _depart_notes), latex=skip, text=skip, man=skip, texinfo=skip)
    app.add_node(TalkFigureNode, html=(_visit_figure, None), latex=skip, text=skip, man=skip, texinfo=skip)
    app.add_directive("shapiq-slide", SlideDirective)
    app.add_directive("shapiq-notes", NotesDirective)
    app.add_directive("shapiq-talk-figure", TalkFigureDirective)
    app.connect("html-page-context", _add_assets)
    app.connect("env-updated", _copy_assets)
    return {
        "version": "1.0",
        "env_version": 1,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
