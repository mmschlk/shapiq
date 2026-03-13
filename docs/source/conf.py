"""shapiq documentation build configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import commonmark
from sphinx.builders.html import StandaloneHTMLBuilder

root = Path(__file__).resolve().parents[2]  # ../../ from this file
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "shapiq"))
sys.path.insert(0, str(root / "examples"))
sys.path.insert(0, str(root / "src"))  # get the shapiq package


# -- Read the Docs ---------------------------------------------------------------------------------
master_doc = "index"

# -- Project information ---------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "shapiq"
copyright = "2024, Muschalik et al."
author = "Muschalik et al."
release = version("shapiq")
version = version("shapiq")

# -- General configuration -------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.duration",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

nbsphinx_allow_errors = True  # optional, avoids build breaking due to execution errors

# -- Sphinx-Gallery -------------------------------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../../docs/source/examples",  # source scripts
    "gallery_dirs": "auto_examples",  # generated output (relative to source/)
    "filename_pattern": r"/plot_",  # only execute files starting with plot_
    "ignore_pattern": r"__init__\.py",
    "plot_gallery": True,
    "download_all_examples": False,
    "show_signature": False,
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "auto_examples/**.ipynb",
    # sphinx-gallery includes this file internally; exclude to avoid toctree warning
    "examples/GALLERY_HEADER.rst",
]
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = (
    "alpha"  # set to alpha to not confuse references the docs with the footcites in docstrings.
)

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- Options for HTML output -----------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_favicon = "_static/logo/shapiq.ico"
pygments_dark_style = "monokai"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo/logo_shapiq_light.svg",
    "dark_logo": "logo/logo_shapiq_dark.svg",
}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ],
}

# -- Autosectionlabel -----------------------------------------------------------------------------
# Prefix labels with the document path to avoid collisions when multiple documents
# share identical section headings (e.g. "General Use", "Feature Names").
autosectionlabel_prefix_document = True

# -- Napoleon (Google-style docstrings) ------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Autodoc ---------------------------------------------------------------------------------------
autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "member-order": "groupwise",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autoclass_content = "both"
autodoc_inherit_docstrings = True
autodoc_member_order = "groupwise"

# -- Suppress known structural warnings -----------------------------------------------------------
# Duplicate citations/footnotes arise because autodoc renders classes in both the parent package
# page (via automodule + members) and the native module page. Suppressing these is safe because
# the actual content is correct; only the duplicate-label warnings are silenced.
suppress_warnings = [
    "ref.citation",
    "ref.footnote",
    "ref.ref",
    "py.duplicate_object",
    "autodoc.duplicate_object",
    "ref.python",  # suppress "duplicate label" warnings for Python objects (e.g. classes) that are rendered in multiple places
]

# -- Images ----------------------------------------------------------------------------------------
StandaloneHTMLBuilder.supported_image_types = [
    "image/svg+xml",
    "image/gif",
    "image/png",
    "image/jpeg",
]
# -- Copy Paste Button -----------------------------------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Markdown in docstring -----------------------------------------------------------------------------
# https://gist.github.com/dmwyatt/0f555c8f9c129c0ac6fed6fabe49078b#file-docstrings-py
# based on https://stackoverflow.com/a/56428123/23972


def docstring(_app, _what, _name, _obj, _options, lines) -> None:
    """Convert Markdown in docstrings to reStructuredText."""
    if len(lines) > 1 and lines[0] == "@&ismd":
        md = "\n".join(lines[1:])
        ast = commonmark.Parser().parse(md)
        rst = commonmark.ReStructuredTextRenderer().render(ast)
        lines.clear()
        lines += rst.splitlines()


def setup(app) -> None:
    """Setup function for the Sphinx extension to convert Markdown in docstrings to reStructuredText."""
    app.connect("autodoc-process-docstring", docstring)
