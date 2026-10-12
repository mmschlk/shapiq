"""shapiq documentation build configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, TypeAliasType, override

from pybtex.plugin import register_plugin
from pybtex.style.formatting.alpha import Style as AlphaStyle
from pybtex.style.template import field, href, join, sentence, words
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.errors import PycodeError
from sphinx.pycode import ModuleAnalyzer

if TYPE_CHECKING:
    from pybtex.database import Entry
    from pybtex.style.template import Node

root = Path(__file__).resolve().parents[2]  # ../../ from this file
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "shapiq"))
sys.path.insert(0, str(root / "examples"))
sys.path.insert(0, str(root / "src"))  # get the shapiq package

# The examples download public Hugging Face models without a token; the hub's notice that a token
# raises its rate limits would otherwise show up in an example's rendered output.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


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
    "doc_module": ("shapiq", "shapiq_games", "shapiq_benchmark"),
    "reference_url": {"shapiq": None, "shapiq_games": None, "shapiq_benchmark": None},
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


def _unescape_latex(text: str) -> str:
    r"""Undo the LaTeX escaping of underscores, e.g. in DOIs such as ``10.1007/...-1\_23``."""
    return text.replace("\\_", "_")


class _BibStyle(AlphaStyle):
    r"""The ``alpha`` style, but titles keep the capitalization of ``references.bib``.

    ``references.bib`` is written for LaTeX styles that print titles as written, so its titles are
    not all brace-protected; ``alpha`` would lowercase them (e.g. "Shapley" -> "shapley"). Links
    from the ``doi`` and ``url`` fields are unescaped (``\_`` -> ``_``) so that they resolve.
    """

    @override
    def format_title(self, e: Entry, which_field: str, *, as_sentence: bool = True) -> Node:
        title = field(which_field)
        return sentence[title] if as_sentence else title

    @override
    def format_url(self, e: Entry) -> Node:
        url = field("url", raw=True, apply_func=_unescape_latex)
        return words["URL:", href(url)[url]]

    @override
    def format_doi(self, e: Entry) -> Node:
        doi = field("doi", raw=True, apply_func=_unescape_latex)
        return href(join["https://doi.org/", doi])[join["doi:", doi]]


register_plugin("pybtex.style.formatting", "shapiq_alpha", _BibStyle)
bibtex_default_style = (
    "shapiq_alpha"  # alpha labels so that the references of the docs differ from the footcites
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
# sphinx-autodoc-typehints adds the types to the documented parameters; Sphinx's own type hints
# ("both" or "description") would add a second, types-only parameter list to every class page
autodoc_typehints = "none"

# -- Re-exported constants -------------------------------------------------------------------------
# A package may define a constant in a private module and re-export it, e.g.
# shapiq_benchmark.datasets.TABARENA_DATASETS. Autodoc reads the docstring of a constant from the
# source of the module it documents, finds none there, and shows the docstring of its type (dict,
# int) instead. Take the docstring from the module that defines it.


def _defining_module_docstring(package_name: str, attribute: str, obj: object) -> list[str] | None:
    """Return the docstring of ``attribute`` in the submodule of a package that defines ``obj``."""
    package = sys.modules.get(package_name)
    if package is None or not hasattr(package, "__path__"):
        return None
    for submodule in pkgutil.iter_modules(package.__path__, f"{package_name}."):
        module = importlib.import_module(submodule.name)
        if getattr(module, attribute, None) is not obj:
            continue
        docs = ModuleAnalyzer.for_module(submodule.name).find_attr_docs().get(("", attribute))
        if docs:
            return list(docs)
    return None


def _use_the_defining_module_docstring(app, what, name, obj, options, lines) -> None:  # noqa: ARG001
    if what != "data":
        return
    package_name, _, attribute = name.rpartition(".")
    try:
        if ("", attribute) in ModuleAnalyzer.for_module(package_name).find_attr_docs():
            return  # the documenting module has its docstring
    except PycodeError:  # no source to read
        return
    lines[:] = _defining_module_docstring(package_name, attribute, obj) or []


# -- Type aliases ----------------------------------------------------------------------------------
# The games and the benchmark declare their choices as type aliases (``type Task = Literal[...]``),
# which sphinx-autodoc-typehints shows as "TypeAliasType". Show the alias, linked to its page (an
# alias of a private module is documented under the package that re-exports it).


def _format_type_alias(annotation: object, config: object) -> str | None:  # noqa: ARG001
    if not isinstance(annotation, TypeAliasType):
        return None  # the default formatting
    module = ".".join(part for part in annotation.__module__.split(".") if not part.startswith("_"))
    return f":py:type:`~{module}.{annotation.__name__}`"


typehints_formatter = _format_type_alias


def setup(app) -> None:
    """Connect the docstring handler (before napoleon, which formats the docstrings)."""
    app.connect("autodoc-process-docstring", _use_the_defining_module_docstring, priority=400)


# -- Suppress warnings -----------------------------------------------------------------------------
suppress_warnings = [
    "misc.highlighting_failure",  # should be fixed in the future
    "config.cache",  # typehints_formatter is a function, which Sphinx cannot pickle with the config
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
