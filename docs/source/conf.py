"""shapiq documentation build configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import functools
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
    "sphinx_design",
]

# -- Timeline of Shapley-based methods (timeline.rst) ----------------------------------------------
# The ``shapiq-timeline`` directive of ``_ext/shapiq_timeline.py`` renders ``_data/timeline.toml``
# and adds ``_static/css/timeline.css`` and ``_static/js/timeline.js`` to the pages that use it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))
extensions.append("shapiq_timeline")

# -- Sphinx-Gallery -------------------------------------------------------------------------------
# One gallery per group of examples (``examples/<group>``), each with its own page; the landing
# page ``examples.rst`` links them. Groups may have subsections (subfolders with a header).
EXAMPLE_GROUPS = (
    "basics",
    "exact_computation",
    "approximators",
    "trees",
    "nearest_neighbors",
    "kernel_methods",
    "visualization",
    "applications",
    "benchmarking",
    "technicalities",
)
sphinx_gallery_conf = {
    "examples_dirs": [str(root / "examples" / group) for group in EXAMPLE_GROUPS],
    "gallery_dirs": [f"auto_examples/{group}" for group in EXAMPLE_GROUPS],
    # subsections as paths relative to this file; the other subsections are sorted by name
    "subsection_order": [
        *(
            f"../../examples/approximators/{subsection}"
            for subsection in ("shapley_values", "shapley_interactions", "surrogate_and_sparse")
        ),
        "*",
    ],
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
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -----------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_title = "shapiq documentation"
html_static_path = ["_static"]
html_extra_path = ["_extra"]  # redirects from moved pages
html_css_files = [
    "css/tokens.css",  # colors, radii, shadows and motion shared by the custom pages
    "css/custom.css",
]
html_favicon = "_static/logo/shapiq.ico"
pygments_style = "github-light"
pygments_dark_style = "github-dark"

# The footer icons are the octicons "mark-github" and "package" (as bundled with sphinx-design).
_GITHUB_ICON = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
    '<path fill-rule="evenodd" d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55'
    "-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31"
    "-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2"
    " .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86"
    " 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27"
    ".38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995"
    ' 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path></svg>'
)
_PYPI_ICON = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
    '<path d="m8.878.392 5.25 3.045c.54.314.872.89.872 1.514v6.098a1.75 1.75 0 0 1-.872 1.514l-5.25'
    " 3.045a1.75 1.75 0 0 1-1.756 0l-5.25-3.045A1.75 1.75 0 0 1 1 11.049V4.951c0-.624.332-1.201.872"
    "-1.514L7.122.392a1.75 1.75 0 0 1 1.756 0ZM7.875 1.69l-4.63 2.685L8 7.133l4.755-2.758-4.63-2.685a"
    ".248.248 0 0 0-.25 0ZM2.5 5.677v5.372c0 .09.047.171.125.216l4.625 2.683V8.432Zm6.25 8.271"
    ' 4.625-2.683a.25.25 0 0 0 .125-.216V5.677L8.75 8.432Z"></path></svg>'
)
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo/logo_shapiq_light.svg",
    "dark_logo": "logo/logo_shapiq_dark.svg",
    # "view" and "edit" buttons; _templates/components maps the gallery pages to their scripts
    # and leaves them out on the generated API pages, which have no source file in the repository
    "source_repository": "https://github.com/mmschlk/shapiq/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {"name": "GitHub", "url": "https://github.com/mmschlk/shapiq", "html": _GITHUB_ICON},
        {"name": "PyPI", "url": "https://pypi.org/project/shapiq/", "html": _PYPI_ICON},
    ],
    # brand colors of the logo (see _static/css/tokens.css), darkened or lightened for contrast
    "light_css_variables": {
        "color-brand-primary": "#d0245c",
        "color-brand-content": "#3a57cf",
        "color-brand-visited": "#3a57cf",
        "color-sidebar-link-text--top-level": "var(--color-foreground-primary)",
        "color-highlighted-background": "#fde7ef",
        "color-inline-code-background": "#f3f4f9",
        "color-code-background": "var(--shapiq-surface-raised)",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff6b95",
        "color-brand-content": "#8fb3ff",
        "color-brand-visited": "#8fb3ff",
        "color-sidebar-link-text--top-level": "var(--color-foreground-primary)",
        "color-highlighted-background": "#3b1d2a",
        "color-inline-code-background": "#1d2027",
        "color-code-background": "var(--shapiq-surface-raised)",
    },
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


_PACKAGES = ("shapiq", "shapiq_games", "shapiq_benchmark")


@functools.cache
def _import_path(path: str) -> object | None:
    """Return the object at a dotted path such as ``shapiq.KernelSHAP``, or ``None``."""
    parts = path.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute, None)
        return obj
    return None


def _link_reexported_names(app, env, node, contnode):  # noqa: ANN202
    """Link a re-exported name (``shapiq.KernelSHAP``) to the page that documents its object.

    The API reference documents each object once, under its subpackage (e.g.
    ``shapiq.approximator.KernelSHAP``); references to the shorter public alias, or to a bare name
    outside its module (``MoebiusConverter``), would otherwise render as plain text.
    """
    target = node.get("reftarget", "")
    if node.get("refdomain") != "py":
        return None
    domain = env.domains.python_domain
    name = target.rpartition(".")[2]
    candidates = [
        documented
        for documented in domain.objects
        if documented.split(".")[0] in _PACKAGES and documented.rpartition(".")[2] == name
    ]
    obj = _import_path(target) if target.split(".")[0] in _PACKAGES else None
    if obj is not None:  # a public alias of a documented object
        candidates = [c for c in candidates if _import_path(c) is obj]
    elif len({id(_import_path(c)) for c in candidates}) != 1:
        return None  # otherwise the name links only if it names one documented object
    if not candidates:
        return None
    return domain.resolve_xref(
        env, node["refdoc"], app.builder, node["reftype"], candidates[0], node, contnode
    )


def setup(app) -> None:
    """Connect the docstring handler (before napoleon, which formats the docstrings)."""
    app.connect("autodoc-process-docstring", _use_the_defining_module_docstring, priority=400)
    app.connect("missing-reference", _link_reexported_names)


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
