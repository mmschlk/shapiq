"""Sphinx extension for the demos (``demos/``): one-time computations with ``shapiq``, on a page.

Unlike the examples, which run with every build of the docs and show how to use the API, a demo
was computed once, and its page shows the result. Every demo has a folder ``demos/<name>/``:

- ``index.rst``: the demo's page, listed in the toctree of ``demos/index.rst``;
- ``assets/``: what the page loads, ``demo.js`` and ``demo.css`` and the demo's data, copied to
  ``_static/demos/<name>/`` of the built docs;
- ``compute/``: the scripts that computed the data, and how to run them again (not part of the
  docs).

The ``shapiq-demo`` directive embeds a demo in a page, its own or any other (the landing page
embeds the FIxLIP demo)::

    .. shapiq-demo:: fixlip
       :anchor: fixlip-demo
       :fallback: images/hero_siglip2.png
       :alt: What the fallback image shows.

A demo that builds up step by step shows the step its ``:step:`` names, as the container's
``data-step`` (the orders demo's ``problem``, ``mechanism``, ``shapley``, ``interactions`` and
``full``; without it, all of it), and points at rest at what its ``:select:`` names, as
``data-select`` (the orders demo's ``none``, ``mechanism``, ``full`` or an order, ``1``, ``2``,
...; without it, what the step adds). A demo with examples shows those its ``:examples:`` names, as
``data-examples``, in that order (the orders demo's ``molecule``, ``house``, ``tabpfn``,
``houses`` and ``fixlip``, apart by commas or spaces; without it, all of them; with one, no tabs).
The demos of one page show the same example: choosing one in any of them chooses it in all that
have it.

It writes the demo's container, ``<div class="<name>-demo" data-root="..." data-version="...">``,
whose ``data-root`` is the URL of the demo's assets relative to the page and ``data-version`` a
checksum of its ``data.js`` (for the script to load the data by, past browsers' caches, where the
page has not loaded it), and adds ``demo.css``, ``data.js`` (if the demo has one) and ``demo.js`` to
the page, the data deferred just before the script, so that both load at once. Without JavaScript,
the container shows the fallback image, a file of the built ``_static`` folder. Outside HTML, a demo
is left out.

The assets are copied before the pages are written, so that Sphinx adds its checksums to the links
of ``demo.css`` and ``demo.js``, and they are dependencies of the pages that embed the demo, so that a
change to them writes these pages again.

The ``shapiq-demo-figure`` directive adds a figure that a demo's script draws on the page, e.g.
one that follows what the demo shows. Its arguments are the demo and the name of the figure, and
its content is the caption::

    .. shapiq-demo-figure:: fixlip masks
       :alt: What the figure shows.

       **The game assigns a similarity to every mask.** The caption.

It writes ``<figure class="<demo>-figure" data-figure="<name>" data-root="...">`` with an empty
stage, which is labeled with the alt text and shows it without JavaScript, and the caption. A page
with such figures gets ``figures.css``, the demo's ``data.js``, ``figures-data.js`` (if the demo has
one) and ``figures.js`` of the demo's assets. Outside HTML, only the caption is kept.

On a page with such figures, the captions are numbered in the order of the page. The figures of
the directive and the ``figure`` directives with a caption count as ``Figure 1``, ``Figure 2``,
and so on, and the tables with a title as ``Table 1``, and so on. The other pages of the docs keep
their captions as they are. The role ``shapiq-figure`` refers to a figure or a table of the page by
its number, as a link. Its target is the name of a demo's figure, or the ``:name:`` of a ``figure``
or a table::

    :shapiq-figure:`masks` shows eight masks.
"""

from __future__ import annotations

import shutil
import zlib
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata
    from sphinx.writers.html5 import HTML5Translator

logger = logging.getLogger(__name__)

_ASSETS = ("demo.js", "demo.css")  # the files that every demo's assets folder has
_DATA = "data.js"  # the demo's data, which its script loads, if it has any
_FIGURES = ("figures.js", "figures.css")  # the files of a demo with figures on a page
_FIGURES_DATA = "figures-data.js"  # what the figures show beyond the demo's data, if anything


class DemoNode(nodes.General, nodes.Element):
    """A demo's container; the demo's script fills it."""


class DemoDirective(SphinxDirective):
    """Embed a demo: ``.. shapiq-demo:: <name>``, the name of its folder in ``demos/``."""

    required_arguments = 1
    option_spec: ClassVar[dict[str, Any]] = {
        "anchor": directives.unchanged_required,  # the container's id, to link to it
        "fallback": directives.unchanged_required,  # an image of _static, shown without scripts
        "alt": directives.unchanged_required,  # the fallback image's text
        "step": directives.unchanged_required,  # of a demo built up step by step, the one to show
        "select": directives.unchanged_required,  # what the demo points at, at rest
        "examples": directives.unchanged_required,  # of a demo with examples, those it shows
    }

    def run(self) -> list[nodes.Node]:
        """Return the demo's container, after checking that the demo has its assets."""
        name = self.arguments[0]
        assets = Path(self.env.srcdir) / "demos" / name / "assets"
        if missing := [file for file in _ASSETS if not (assets / file).is_file()]:
            msg = f"the demo {name!r} has no {', '.join(missing)} in demos/{name}/assets"
            raise self.error(msg)
        for file in (*_ASSETS, _DATA):
            if (assets / file).is_file():
                self.env.note_dependency(str(assets / file))
        data = assets / _DATA
        version = f"{zlib.crc32(data.read_bytes()):08x}" if data.is_file() else ""
        node = DemoNode(demo=name, fallback=self.options.get("fallback"), version=version)
        node["alt"] = self.options.get("alt", "")
        node["step"] = self.options.get("step", "")
        node["select"] = self.options.get("select", "")
        node["examples"] = self.options.get("examples", "")
        if "anchor" in self.options:
            node["ids"].append(self.options["anchor"])
        return [node]


class FigureNode(nodes.General, nodes.Element):
    """A figure that a demo's script draws; its children are the caption."""


class FigureDirective(SphinxDirective):
    """Add a figure of a demo: ``.. shapiq-demo-figure:: <demo> <name>``, with its caption."""

    required_arguments = 2
    has_content = True
    option_spec: ClassVar[dict[str, Any]] = {
        "alt": directives.unchanged_required,  # what the figure shows, in words
    }

    def run(self) -> list[nodes.Node]:
        """Return the figure with its parsed caption, after checking the demo's assets."""
        demo, name = self.arguments
        assets = Path(self.env.srcdir) / "demos" / demo / "assets"
        if missing := [file for file in _FIGURES if not (assets / file).is_file()]:
            msg = f"the demo {demo!r} has no {', '.join(missing)} in demos/{demo}/assets"
            raise self.error(msg)
        for file in (*_FIGURES, _DATA, _FIGURES_DATA):
            if (assets / file).is_file():
                self.env.note_dependency(str(assets / file))
        node = FigureNode(demo=demo, figure=name, alt=self.options.get("alt", ""))
        node["ids"].append(f"{demo}-figure-{name}")
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class FigureReference(nodes.Inline, nodes.Element):
    """A reference to a numbered figure or table of the page, resolved when the page is written."""


_CAPTIONED = (FigureNode, nodes.figure, nodes.table)  # what a page with figures numbers


def _visit_demo(self: HTML5Translator, node: DemoNode) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    root = relative_uri(page, f"_static/demos/{node['demo']}/")
    attributes = {"data-root": root, "data-version": node.get("version", "")}
    for option in ("step", "select", "examples"):
        if node.get(option):
            attributes[f"data-{option}"] = node[option]
    self.body.append(self.starttag(node, "div", CLASS=f"{node['demo']}-demo", **attributes))
    if node["fallback"]:
        src = relative_uri(page, f"_static/{node['fallback']}")
        self.body.append(
            f'<noscript><img class="{node["demo"]}-demo-fallback" src="{escape(src)}"'
            f' alt="{escape(node["alt"])}"></noscript>'
        )


def _depart_demo(self: HTML5Translator, node: DemoNode) -> None:  # noqa: ARG001
    self.body.append("</div>\n")


def _skip_demo(self: nodes.NodeVisitor, node: DemoNode) -> None:  # noqa: ARG001
    """Leave a demo out of a build that is not HTML (text, LaTeX, ...)."""
    raise nodes.SkipNode


def _visit_figure(self: HTML5Translator, node: FigureNode) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    root = relative_uri(page, f"_static/demos/{node['demo']}/")
    name = f"{node['demo']}-figure"
    self.body.append(
        self.starttag(
            node, "figure", CLASS=name, **{"data-figure": node["figure"], "data-root": root}
        )
    )
    alt = escape(node["alt"])
    self.body.append(
        f'<div class="{name}-stage" role="group" aria-label="{alt}">'
        f'<noscript><p class="{name}-missing">{alt}</p></noscript></div>\n<figcaption>\n'
    )


def _depart_figure(self: HTML5Translator, node: FigureNode) -> None:  # noqa: ARG001
    self.body.append("</figcaption>\n</figure>\n")


def _keep_caption(self: nodes.NodeVisitor, node: FigureNode) -> None:
    """Outside HTML, a figure is its caption."""


def _figure_role(
    name: str,  # noqa: ARG001
    rawtext: str,
    text: str,
    lineno: int,  # noqa: ARG001
    inliner: object,  # noqa: ARG001
    options: dict[str, Any] | None = None,  # noqa: ARG001
    content: list[str] | None = None,  # noqa: ARG001
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """Refer to a figure or a table of the page by its number: ``:shapiq-figure:`<name>```."""
    return [FigureReference(rawtext, target=text.strip())], []


def _number_captions(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Number the captions of a page with a demo's figures, and link the references to them."""
    labels: dict[str, tuple[str, str]] = {}  # the name of a figure or a table: its number and id
    if next(iter(doctree.findall(FigureNode)), None) is not None:
        counts = {"Figure": 0, "Table": 0}
        for node in list(doctree.findall(lambda n: isinstance(n, _CAPTIONED))):
            kind = "Table" if isinstance(node, nodes.table) else "Figure"
            # the caption is the title of a table, the caption of a figure, or the first
            # paragraph of a demo's figure
            holder = (
                nodes.title
                if kind == "Table"
                else nodes.paragraph
                if isinstance(node, FigureNode)
                else nodes.caption
            )
            caption = next((child for child in node.children if isinstance(child, holder)), None)
            if caption is None:
                continue
            counts[kind] += 1
            label = f"{kind} {counts[kind]}"
            caption.insert(0, nodes.inline("", f"{label}. ", classes=["caption-number"]))
            names = [node["figure"]] if isinstance(node, FigureNode) else node["names"]
            for name in names:
                labels[name] = (label, node["ids"][0] if node["ids"] else "")
    for reference in list(doctree.findall(FigureReference)):
        label, target = labels.get(reference["target"], ("", ""))
        if not label:
            logger.warning(
                "no numbered figure or table %r on this page",
                reference["target"],
                location=(docname, reference.line),
            )
            reference.replace_self(nodes.Text(reference["target"]))
        elif target and app.builder.format == "html":
            reference.replace_self(nodes.reference("", label, refid=target, internal=True))
        else:
            reference.replace_self(nodes.Text(label))


def _add_assets(
    app: Sphinx,
    pagename: str,  # noqa: ARG001
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],  # noqa: ARG001
    doctree: nodes.document | None,
) -> None:
    """Add the stylesheet and the script of every demo of a page to it."""
    if doctree is None:
        return
    demos = {node["demo"] for node in doctree.findall(DemoNode)}
    figures = {node["demo"] for node in doctree.findall(FigureNode)}
    for name in sorted(demos | figures):
        assets = Path(app.srcdir) / "demos" / name / "assets"
        # the stylesheets after html_css_files, whose tokens.css they use
        if name in demos:
            app.add_css_file(f"demos/{name}/demo.css", priority=900)
        if name in figures:
            app.add_css_file(f"demos/{name}/figures.css", priority=900)
        # the data, if any, just before the scripts and loaded alongside them (not by them, once
        # they have run): a script uses the data on the page, and loads it itself where it is missing
        data = [_DATA, *([_FIGURES_DATA] if name in figures else [])]
        for file in data:
            if (assets / file).is_file():
                app.add_js_file(f"demos/{name}/{file}", loading_method="defer", priority=499)
        if name in demos:
            app.add_js_file(f"demos/{name}/demo.js", loading_method="defer")
        if name in figures:
            app.add_js_file(f"demos/{name}/figures.js", loading_method="defer")


def _copy_assets(app: Sphinx, env: object) -> None:  # noqa: ARG001
    """Copy every demo's assets to ``_static/demos/<name>/``, before the pages are written."""
    if app.builder.format != "html":
        return
    for assets in sorted((Path(app.srcdir) / "demos").glob("*/assets")):
        target = Path(app.outdir) / "_static" / "demos" / assets.parent.name
        shutil.copytree(assets, target, dirs_exist_ok=True)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the directive, its node and the assets."""
    skip = (_skip_demo, None)
    app.add_node(
        DemoNode,
        html=(_visit_demo, _depart_demo),
        latex=skip,
        text=skip,
        man=skip,
        texinfo=skip,
    )
    keep = (_keep_caption, _keep_caption)
    app.add_node(
        FigureNode,
        html=(_visit_figure, _depart_figure),
        latex=keep,
        text=keep,
        man=keep,
        texinfo=keep,
    )
    app.add_directive("shapiq-demo", DemoDirective)
    app.add_directive("shapiq-demo-figure", FigureDirective)
    app.add_node(FigureReference)
    app.add_role("shapiq-figure", _figure_role)
    app.connect("doctree-resolved", _number_captions)
    app.connect("html-page-context", _add_assets)
    app.connect("env-updated", _copy_assets)
    return {
        "version": "1.0",
        "env_version": 7,  # pages are read again where what the extension writes to them changed
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
