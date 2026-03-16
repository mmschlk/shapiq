# Contributing to the API Documentation

When you add a new class or function to shapiq, update **2 files** in the docs.

## 1. The overview page: `docs/source/api_reference.rst`

Add a row to the relevant `list-table`. For example, a new explainer:

```rst
   * - :class:`~shapiq.MyNewExplainer`
     - One-line description of what it does.
```

For functions, use `:func:` instead of `:class:`.

## 2. The category page: `docs/source/api/<category>.rst`

Two additions:

### a) Add to the `autosummary` block at the top

This generates the summary table on the category page.

```rst
.. autosummary::
   :nosignatures:

   ...
   MyNewExplainer
```

### b) Add a section header + autodoc directive

This generates the full documentation and makes it appear in the right-side "On this page" TOC.

For classes:

```rst
MyNewExplainer
--------------

.. autoclass:: MyNewExplainer
   :members:
   :no-private-members:
   :show-inheritance:
```

For functions:

```rst
my_new_function
---------------

.. autofunction:: my_new_function
```

## Quick reference by type

| Adding a...     | Overview table in                         | Detail page                                                          |
| --------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| Core class      | `api_reference.rst` > Core                | `api/core.rst` > `autoclass`                                         |
| Explainer       | `api_reference.rst` > Explainers          | `api/explainers.rst` > `autoclass`                                   |
| Approximator    | `api_reference.rst` > relevant sub-table  | `api/approximators.rst` > `autoclass` under the right family heading |
| Imputer         | `api_reference.rst` > Imputers            | `api/imputers.rst` > `autoclass`                                     |
| Plot function   | `api_reference.rst` > Plotting            | `api/plotting.rst` > `autofunction`                                  |
| Dataset loader  | `api_reference.rst` > Datasets            | `api/datasets.rst` > `autofunction`                                  |
| Utility function| `api_reference.rst` > Utilities           | `api/utilities.rst` > `autofunction`                                 |

## Special cases

### Not in top-level `shapiq.__all__`

If the object lives in a submodule (e.g. `shapiq.utils.something`), use the fully qualified name
in the autodoc directive and a prefixed name in autosummary:

```rst
.. autosummary::
   :nosignatures:

   utils.something

...

.. autofunction:: shapiq.utils.something
```

### Approximator proxy classes

Classes in `shapiq.approximator` that are not re-exported at the top level need a
`currentmodule` switch:

```rst
.. currentmodule:: shapiq.approximator

.. autoclass:: ProxySHAP
   :members:
   :no-private-members:
   :show-inheritance:

.. currentmodule:: shapiq
```

## Building the docs

```bash
cd docs && uv run make html
# or
uv run sphinx-build docs/source docs/build -W
```

Open `docs/build/html/index.html` to preview.
