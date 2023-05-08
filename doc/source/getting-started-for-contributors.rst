Getting Started (for Contributors)
==================================

Prerequisites
-------------

- `Python 3.7 <https://docs.python.org/3.7/>`_ or above
- `Poetry 1.3 <https://python-poetry.org/>`_ or above
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
:code:`pyenv` (with the :code:`pyenv-virtualenv` plugin) and already have it installed
, you can use the following convenience script (by default it will use :code:`Python 3.7.15`,
but you can change it by providing a specific :code:`<version>`)::

  $ ./dev/venv-create.sh <version>

If you don't have :code:`pyenv` installed, 
you can use the following script that will install pyenv, 
set it up and create the virtual environment (with :code:`Python 3.7.15` by default)::

  $ ./dev/setup-defaults.sh <version>

Third, install the Flower package in development mode (think
:code:`pip install -e`) along with all necessary dependencies::

  (flower-<version>) $ ./dev/bootstrap.sh


Convenience Scripts
-------------------

The Flower repository contains a number of convenience scripts to make
recurring development tasks easier and less error-prone. See the :code:`/dev`
subdirectory for a full list. The following scripts are amonst the most
important ones:

Create/Delete Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  $ ./dev/venv-create.sh <version> # Default is 3.7.15
  $ ./dev/venv-delete.sh <version> # Default is 3.7.15

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

Run Github Actions (CI) locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developers could run the full set of Github Actions workflows under their local
environment by using `Act <https://github.com/nektos/act>_`. Please refer to
the installation instructions under the linked repository and run the next
command under Flower main cloned repository folder::

  $ act

The Flower default workflow would run by setting up the required Docker
machines underneath.


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

Note that, in order to build the documentation locally 
(with ``poetry run make html``, like described below), 
`Pandoc <https://pandoc.org/installing.html>_` needs to be installed on the system. 
