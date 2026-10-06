"""Render the canonical About Markdown; run with --check before publishing assets.

Uses markdown-it-py from the existing development environment. The generated HTML
is committed so report exports and Pages need no Markdown runtime dependency.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt

SITE = Path(__file__).resolve().parent / "site"


def render_about(source: str) -> str:
    """Render a static article and a table of contents from its level-two headings."""
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = markdown.parse(source)
    links = []
    identifiers = set()
    for position, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        title = tokens[position + 1].content
        identifier = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if identifier in identifiers:
            message = f"Duplicate About heading: {title}"
            raise ValueError(message)
        identifiers.add(identifier)
        token.attrSet("id", identifier)
        if token.tag == "h2":
            links.append(f'<li><a href="#{identifier}">{html.escape(title)}</a></li>')
    article = markdown.renderer.render(tokens, markdown.options, {})
    contents = (
        '<nav aria-label="On this page"><h2>Contents</h2><ul>' + "".join(links) + "</ul></nav>"
    )
    if article.count("<p>[TOC]</p>") != 1:
        message = "About Markdown requires exactly one [TOC] paragraph."
        raise ValueError(message)
    article = article.replace("<p>[TOC]</p>", contents)
    return f"""<!doctype html>
<!-- Generated from about.md by benchmark/render_about.py; edit the Markdown source. -->
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta name="description" content="The focused Shapley estimator benchmark: games, exact references, query budgets and scores." />
  <title>About the benchmark · shapiq</title>
  <link rel="icon" href="shapiq.svg" type="image/svg+xml" />
  <link rel="stylesheet" href="about.css" />
</head>
<body>
  <a class="skipLink" href="#content">Skip to content</a>
  <header><a href="index.html">← Benchmark results</a><a href="about.md">Markdown source</a></header>
  <main id="content">
{article}  </main>
</body>
</html>
"""


def main() -> None:
    """Write the generated asset or fail when the committed copy is stale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_about((SITE / "about.md").read_text())
    output = SITE / "about.html"
    if args.check:
        if output.read_text() != rendered:
            parser.error("about.html is stale; run python benchmark/render_about.py")
    else:
        output.write_text(rendered)


if __name__ == "__main__":
    main()
