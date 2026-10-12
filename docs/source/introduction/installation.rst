Installation
=============

The latest release version of ``shapiq`` can be installed from
`PyPI <https://pypi.org/project/shapiq>`_ with:

.. code::

   pip install shapiq


The development version can be installed from
`GitHub <https://github.com/mmschlk/shapiq>`_ with:

.. code::

   pip install git+https://github.com/mmschlk/shapiq


Games and Benchmark
~~~~~~~~~~~~~~~~~~~

The games of ``shapiq_games`` and the benchmark ``shapiq_benchmark`` ship with ``shapiq``. Their
optional dependencies (e.g. ``torch`` and ``transformers`` for the image and language games,
``openml`` for the benchmark's datasets) are installed with the extras:

.. code::

   pip install "shapiq[games]"      # shapiq_games
   pip install "shapiq[benchmark]"  # shapiq_benchmark, includes the games extra


Development
~~~~~~~~~~~

Additional packages required for the development of ``shapiq`` (documentation, tests) can be installed with:

.. code::

   pip install shapiq[docs]
   pip install shapiq[dev] # includes docs
