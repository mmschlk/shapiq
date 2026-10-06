"""Keep the exported static guide synchronized with its canonical Markdown."""

from __future__ import annotations

import json
import re

import pytest
from benchmark.render_about import SITE, render_about


def test_about_asset_is_current_and_self_contained() -> None:
    rendered = render_about((SITE / "about.md").read_text())
    assert (SITE / "about.html").read_text() == rendered
    assert '<meta charset="utf-8"' in rendered
    assert "<script" not in rendered
    assert "about.json" not in rendered
    identifiers = re.findall(r'\bid="([^"]+)"', rendered)
    assert len(identifiers) == len(set(identifiers))
    assert set(re.findall(r'href="#([^"]+)"', rendered)) <= set(identifiers)
    assert "scores" in identifiers  # Existing dashboard help link.
    assets = json.loads((SITE / "assets.json").read_text())
    assert {"about.html", "about.css", "about.md"} <= set(assets)
    assert "about.js" not in assets


@pytest.mark.parametrize("source", ["# Title", "# Title\n\n[TOC]\n\n[TOC]"])
def test_about_requires_one_contents_marker(source: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        render_about(source)


def test_about_rejects_duplicate_heading_targets() -> None:
    with pytest.raises(ValueError, match="Duplicate About heading"):
        render_about("# Title\n\n[TOC]\n\n## Repeated\n\n## Repeated")
