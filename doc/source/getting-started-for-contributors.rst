Getting Started (for Contributors)
==================================

Prerequisites
-------------

- `Python 3.7 <https://docs.python.org/3.7/>`_ or above
- `Poetry 1.0 <https://python-poetry.org/>`_ or above
- (Optional) `pyenv <https://github.com/pyenv/pyenv>`_
- (Optional) `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_

Flower uses :code:`pyproject.toml` to manage dependencies and configure
development tools (the ones which support it). Poetry is a build tool which
supports `PEP 517 <https://www.python.org/dev/peps/pep-0517/>`_.


Developer Machine Setup
-----------------------

First, clone the `Flower repository <https://github.com/adap/flower>`_ from
GitHub::

  $ git clone git@github.com:adap/flower.git
  $ cd flower

Second, create a virtual environment (and activate it). If you chose to use
:code:`pyenv` (with the :code:`pyenv-virtualenv` plugin), you can use the
following convenience script::

  $ ./dev/venv-create.sh

Third, install the Flower package in development mode (think
:code:`pip install -e`) along with all necessary dependencies::

  (flower-3.7.9) $ ./dev/bootstrap.sh


Convenience Scripts
-------------------

The Flower repository contains a number of convenience scripts to make
recurring development tasks easier and less error-prone. See the :code:`/dev`
subdirectory for a full list. The following scripts are amonst the most
important ones:

Create/Delete Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  $ ./dev/venv-create.sh
  $ ./dev/venv-delete.sh

Compile ProtoBuf Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  $ python -m flwr_tool.protoc

Auto-Format Code
~~~~~~~~~~~~~~~~

::

  $ ./dev/format.sh

Run Linters and Tests
~~~~~~~~~~~~~~~~~~~~~

::

  $ ./dev/test.sh


Build Release
-------------

Flower uses Poetry to build releases. The necessary command is wrapped in a
simple script::

  $ ./dev/build.sh

The resulting :code:`.whl` and :code:`.tar.gz` releases will be stored in the
:code:`/dist` subdirectory.


Build Documentation
-------------------

Flower's documentation uses `Sphinx <https://www.sphinx-doc.org/>`_. There's no
convenience script to re-build the documentation yet, but it's pretty easy::

  $ cd doc
  $ make html

This will generate HTML documentation in ``doc/build/html``.
