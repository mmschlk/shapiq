"""shapiq documentation build configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

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

# -- Sphinx-Gallery -------------------------------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": [str(root / "examples")],
    "gallery_dirs": ["auto_examples"],
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("shapiq",),
    "reference_url": {"shapiq": None},
    "filename_pattern": r"plot_.*\.py",
    "ignore_pattern": r"util_.*\.py",
    "plot_gallery": True,
    "download_all_examples": False,
    "show_signature": False,
    "run_stale_examples": True,
    "abort_on_example_error": True,
    "default_thumb_file": str(
        root / "docs" / "source" / "_static" / "logo" / "logo_shapiq_light.svg"
    ),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_examples/**.ipynb",
    # stale sphinx-gallery artifact at source root (real copy lives in auto_examples/)
    "sg_execution_times.rst",
]
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = (
    "alpha"  # set to alpha to not confuse references the docs with the footcites in docstrings.
)
bibtex_reference_style = "author_year"

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
napoleon_use_ivar = True

# -- Autodoc ---------------------------------------------------------------------------------------
autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "private-members": False,
    "member-order": "groupwise",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autoclass_content = "both"
autodoc_inherit_docstrings = True
autodoc_member_order = "groupwise"
autodoc_typehints = "both"

# -- Suppress warnings -----------------------------------------------------------------------------
suppress_warnings = [
    "misc.highlighting_failure",  # should be fixed in the future
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
