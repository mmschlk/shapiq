"""Sphinx extension for the illustrations of the user guide (``user_guide/illustrations/``).

An illustration is an interactive figure that explains an idea of the user guide: it shows a
concept, where a demo (``_ext/shapiq_demos.py``) shows a result. Every illustration has a folder
``user_guide/illustrations/<name>/``:

- ``assets/``: what the page loads, ``illustration.js`` and ``illustration.css``, and its data if it
  has any (as a script, ``data.js``, so that docs opened from disk can load it), copied to
  ``_static/illustrations/<name>/`` of the built docs;
- ``compute/``: the scripts that computed the data, if any (not part of the docs).

The ``shapiq-illustration`` directive embeds an illustration in a page; its content, if any, is the
caption::

    ```{shapiq-illustration} vocabulary
    :alt: What the illustration shows, for readers without scripts or eyes.

    The caption, with **markup**, $math$ and {cite:p}`Key`.
    ```

It writes ``<figure class="shapiq-illustration" data-illustration="<name>" data-root="..."
data-version="...">``, whose ``data-root`` is the URL of the illustration's assets relative to the
page and ``data-version`` a checksum of its ``data.js``, with an empty stage that the
illustration's script fills. It adds the shared script and stylesheet of all illustrations
(``_static/js/illustrations.js``, ``_static/css/illustrations.css``) and the illustration's own to
the page, its data (if any) deferred just before its script, so that both load at once; where the
data is missing, the script loads it by ``data-root`` and ``data-version``, past browsers' caches.
Without JavaScript, the stage shows the ``alt`` text; outside HTML, an illustration is left out.

As for the demos (``_ext/shapiq_demos.py``), the assets are copied before the pages are written, so
that Sphinx adds its checksums to their links, and they are dependencies of the pages that embed
them, so that a change to them writes these pages again.
"""

from __future__ import annotations

import shutil
import zlib
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

_FOLDER = ("user_guide", "illustrations")  # the illustrations' folder, relative to the sources
_ASSETS = ("illustration.js", "illustration.css")  # the files that every assets folder has
_DATA = "data.js"  # the illustration's data, if it has any


class IllustrationNode(nodes.General, nodes.Element):
    """An illustration: its figure, its stage for the script, and its caption (the children)."""


class IllustrationDirective(SphinxDirective):
    """Embed an illustration: ``shapiq-illustration`` with the name of its folder."""

    required_arguments = 1
    has_content = True
    option_spec: ClassVar[dict[str, Any]] = {
        "anchor": directives.unchanged_required,  # the figure's id, to link to it
        "alt": directives.unchanged_required,  # what it shows, for readers without scripts
    }

    def run(self) -> list[nodes.Node]:
        """Return the illustration, after checking that it has its assets."""
        name = self.arguments[0]
        assets = Path(self.env.srcdir).joinpath(*_FOLDER, name, "assets")
        if missing := [file for file in _ASSETS if not (assets / file).is_file()]:
            msg = f"the illustration {name!r} has no {', '.join(missing)} in {assets}"
            raise self.error(msg)
        for file in (*_ASSETS, _DATA):
            if (assets / file).is_file():
                self.env.note_dependency(str(assets / file))
        data = assets / _DATA
        version = f"{zlib.crc32(data.read_bytes()):08x}" if data.is_file() else ""
        node = IllustrationNode(illustration=name, alt=self.options.get("alt", ""), version=version)
        if "anchor" in self.options:
            node["ids"].append(self.options["anchor"])
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def _visit(self: HTML5Translator, node: IllustrationNode) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    name = node["illustration"]
    root = relative_uri(page, f"_static/illustrations/{name}/")
    self.body.append(
        self.starttag(
            node,
            "figure",
            CLASS=f"shapiq-illustration shapiq-illustration-{name}",
            **{
                "data-illustration": name,
                "data-root": root,
                "data-version": node.get("version", ""),
            },
        )
    )
    alt = escape(node["alt"])
    self.body.append(
        f'<div class="shapiq-illustration-stage" role="group" aria-label="{alt}">'
        f'<noscript><p class="shapiq-illustration-noscript">{alt}</p></noscript></div>'
    )
    if node.children:
        self.body.append('<figcaption class="shapiq-illustration-caption">')


def _depart(self: HTML5Translator, node: IllustrationNode) -> None:
    if node.children:
        self.body.append("</figcaption>")
    self.body.append("</figure>\n")


def _skip(self: nodes.NodeVisitor, node: IllustrationNode) -> None:  # noqa: ARG001
    """Leave an illustration out of a build that is not HTML (text, LaTeX, ...)."""
    raise nodes.SkipNode


def _add_assets(
    app: Sphinx,
    pagename: str,  # noqa: ARG001
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],  # noqa: ARG001
    doctree: nodes.document | None,
) -> None:
    """Add the shared files and those of every illustration of a page to it."""
    if doctree is None:
        return
    names = sorted({node["illustration"] for node in doctree.findall(IllustrationNode)})
    if not names:
        return
    # the stylesheets after html_css_files, whose tokens.css they use; the shared script first
    app.add_css_file("css/illustrations.css", priority=900)
    app.add_js_file("js/illustrations.js", loading_method="defer", priority=600)
    folder = Path(app.srcdir).joinpath(*_FOLDER)
    for name in names:
        app.add_css_file(f"illustrations/{name}/illustration.css", priority=910)
        # the data, if any, just before the script and loaded alongside it
        if (folder / name / "assets" / _DATA).is_file():
            app.add_js_file(f"illustrations/{name}/{_DATA}", loading_method="defer", priority=605)
        app.add_js_file(
            f"illustrations/{name}/illustration.js", loading_method="defer", priority=610
        )


def _copy_assets(app: Sphinx, env: object) -> None:  # noqa: ARG001
    """Copy every illustration's assets to ``_static/illustrations/<name>/``, before the pages."""
    if app.builder.format != "html":
        return
    for assets in sorted(Path(app.srcdir).joinpath(*_FOLDER).glob("*/assets")):
        target = Path(app.outdir) / "_static" / "illustrations" / assets.parent.name
        shutil.copytree(assets, target, dirs_exist_ok=True)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the directive, its node and the assets."""
    skip = (_skip, None)
    app.add_node(
        IllustrationNode,
        html=(_visit, _depart),
        latex=skip,
        text=skip,
        man=skip,
        texinfo=skip,
    )
    app.add_directive("shapiq-illustration", IllustrationDirective)
    app.connect("html-page-context", _add_assets)
    app.connect("env-updated", _copy_assets)
    return {
        "version": "1.0",
        "env_version": 1,  # the nodes carry a version: environments of earlier builds are read again
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
