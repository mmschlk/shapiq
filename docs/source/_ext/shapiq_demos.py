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
       :anchor: interactive-demo
       :fallback: images/hero_siglip2.png
       :alt: What the fallback image shows.

It writes the demo's container, ``<div class="<name>-demo" data-root="...">``, whose ``data-root``
is the URL of the demo's assets relative to the page, and adds ``demo.css`` and ``demo.js`` to the
page. Without JavaScript, the container shows the fallback image, a file of the built ``_static``
folder.
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

_ASSETS = ("demo.js", "demo.css")  # the files that every demo's assets folder has


class DemoNode(nodes.General, nodes.Element):
    """A demo's container; the demo's script fills it."""


class DemoDirective(SphinxDirective):
    """Embed a demo: ``.. shapiq-demo:: <name>``, the name of its folder in ``demos/``."""

    required_arguments = 1
    option_spec: ClassVar[dict[str, Any]] = {
        "anchor": directives.unchanged_required,  # the container's id, to link to it
        "fallback": directives.unchanged_required,  # an image of _static, shown without scripts
        "alt": directives.unchanged_required,  # the fallback image's text
    }

    def run(self) -> list[nodes.Node]:
        """Return the demo's container, after checking that the demo has its assets."""
        name = self.arguments[0]
        assets = Path(self.env.srcdir) / "demos" / name / "assets"
        if missing := [file for file in _ASSETS if not (assets / file).is_file()]:
            msg = f"the demo {name!r} has no {', '.join(missing)} in demos/{name}/assets"
            raise self.error(msg)
        node = DemoNode(demo=name, fallback=self.options.get("fallback"))
        node["alt"] = self.options.get("alt", "")
        if "anchor" in self.options:
            node["ids"].append(self.options["anchor"])
        return [node]


def _visit_demo(self: HTML5Translator, node: DemoNode) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    root = relative_uri(page, f"_static/demos/{node['demo']}/")
    self.body.append(
        self.starttag(node, "div", CLASS=f"{node['demo']}-demo", **{"data-root": root})
    )
    if node["fallback"]:
        src = relative_uri(page, f"_static/{node['fallback']}")
        self.body.append(
            f'<noscript><img class="{node["demo"]}-demo-fallback" src="{escape(src)}"'
            f' alt="{escape(node["alt"])}"></noscript>'
        )


def _depart_demo(self: HTML5Translator, node: DemoNode) -> None:  # noqa: ARG001
    self.body.append("</div>\n")


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
    for name in sorted({node["demo"] for node in doctree.findall(DemoNode)}):
        # the stylesheet after html_css_files, whose tokens.css it uses
        app.add_css_file(f"demos/{name}/demo.css", priority=900)
        app.add_js_file(f"demos/{name}/demo.js", loading_method="defer")


def _copy_assets(app: Sphinx, exception: Exception | None) -> None:
    """Copy every demo's assets to ``_static/demos/<name>/`` of the built docs."""
    if exception is not None or app.builder.format != "html":
        return
    for assets in sorted((Path(app.srcdir) / "demos").glob("*/assets")):
        target = Path(app.outdir) / "_static" / "demos" / assets.parent.name
        shutil.copytree(assets, target, dirs_exist_ok=True)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the directive, its node and the assets."""
    app.add_node(DemoNode, html=(_visit_demo, _depart_demo))
    app.add_directive("shapiq-demo", DemoDirective)
    app.connect("html-page-context", _add_assets)
    app.connect("build-finished", _copy_assets)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
