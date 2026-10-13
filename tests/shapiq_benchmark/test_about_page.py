"""Keep the exported static guide synchronized with its canonical Markdown."""

from __future__ import annotations

import json
import re

import pytest
from benchmark.render_about import SITE, render_about


def test_about_asset_is_current_and_has_pinned_math_renderer() -> None:
    rendered = render_about((SITE / "about.md").read_text())
    assert (SITE / "about.html").read_text() == rendered
    assert '<meta charset="utf-8"' in rendered
    assert rendered.count("<script") == 1
    assert (
        'defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml-full.js"' in rendered
    )
    assert "polyfill" not in rendered
    assert 'href="style.css"' in rendered
    assert "about.json" not in rendered
    identifiers = re.findall(r'\bid="([^"]+)"', rendered)
    assert len(identifiers) == len(set(identifiers))
    assert set(re.findall(r'href="#([^"]+)"', rendered)) <= set(identifiers)
    dashboard = (SITE / "index.html").read_text()
    assert 'href="about.html"' in dashboard
    dashboard_targets = re.findall(r'href="about\.html#([^"]+)"', dashboard)
    assert set(dashboard_targets) <= set(identifiers)
    assets = json.loads((SITE / "assets.json").read_text())
    assert {"about.html", "about.css", "about.md"} <= set(assets)
    assert "about.js" not in assets


def test_math_preserves_tex_and_escapes_html() -> None:
    rendered = render_about(
        "# Title\n\n[TOC]\n\n"
        r"Inline $\hat{\boldsymbol{\phi}}_i < 2$ and **bold**."
        "\n\n"
        r"$$\mathbf{nMSE} = \frac{\lVert \boldsymbol{\phi} \rVert_2^2}{d}.$$"
    )
    assert r'<span class="math">\(\hat{\boldsymbol{\phi}}_i &lt; 2\)</span>' in rendered
    assert r'<div class="math">\[\mathbf{nMSE}' in rendered
    assert r"\lVert \boldsymbol{\phi} \rVert_2^2" in rendered
    assert "<strong>bold</strong>" in rendered
    assert "SHAPIQMATH" not in rendered
    assert '<p><div class="math">' not in rendered


def test_all_dataset_tables_are_closed_with_prepared_compact_scope() -> None:
    source = (SITE / "about.md").read_text()
    rendered = render_about(source)
    assert rendered.count('<details class="datasetTable">') == 3
    assert "<details open" not in rendered
    assert rendered.count("<tbody>") == 3
    assert rendered.count("<tr>") == 75  # Three headings plus 72 dataset rows.
    assert "Scope version 2" in source
    assert "512 recipes and 2,048" in source
    assert "selected player grids; estimator coverage counts only" in source
    assert "477 core recipes and 35 KNN/TNN recipes" in source
    assert "All 20 required estimators" in source
    assert "**100% successful supported coverage**" in source
    assert "ShaplEIG and SPEX retain partial" in source
    assert "Historical broader results" in source
    assert "97.0% reference coverage" not in source
    assert "Exact / planned instances" not in source
    assert "except ShaplEIG and SPEX. ProxySPEX is included." in source
    assert "0.5, 1, 2, 4, 8, 16 and 32" in source
    assert "64 and 128" not in source
    assert "Neighbor players (KNN / TNN)" in source
    assert "| Adult Census | 4, 8, 12 / 4, 8, 12 | 64 / 64 | 12 |" in source
    assert "| California Housing | — / 4, 8, 12 | — / — | 8 |" in source
    assert "**signed ridge regression** with no intercept" in source
    assert "dataset-grouped cross-validation" in source
    assert "negative weights remain allowed" in source
    assert "Coefficients describe combinations, not estimator importance" in source
    assert "Training-row groups" in source


@pytest.mark.parametrize("source", ["# Title", "# Title\n\n[TOC]\n\n[TOC]"])
def test_about_requires_one_contents_marker(source: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        render_about(source)


def test_about_rejects_duplicate_heading_targets() -> None:
    with pytest.raises(ValueError, match="Duplicate About heading"):
        render_about("# Title\n\n[TOC]\n\n## Repeated\n\n## Repeated")
