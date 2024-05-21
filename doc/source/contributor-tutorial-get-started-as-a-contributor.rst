Get started as a contributor
============================

Prerequisites
-------------

- `Python 3.8 <https://docs.python.org/3.8/>`_ or above
- `Poetry 1.3 <https://python-poetry.org/>`_ or above
- (Optional) `pyenv <https://github.com/pyenv/pyenv>`_
- (Optional) `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_

Flower uses :code:`pyproject.toml` to manage dependencies and configure
development tools (the ones which support it). Poetry is a build tool which
supports `PEP 517 <https://peps.python.org/pep-0517/>`_.


Developer Machine Setup
-----------------------

Preliminarities
~~~~~~~~~~~~~~~
Some system-wide dependencies are needed.

For macOS
^^^^^^^^^

* Install `homebrew <https://brew.sh/>`_. Don't forget the post-installation actions to add `brew` to your PATH.
* Install `xz` (to install different Python versions) and `pandoc` to build the
  docs::

  $ brew install xz pandoc

For Ubuntu
^^^^^^^^^^
Ensure you system (Ubuntu 22.04+) is up-to-date, and you have all necessary
packages::

  $ apt update
  $ apt install build-essential zlib1g-dev libssl-dev libsqlite3-dev \
                libreadline-dev libbz2-dev libffi-dev liblzma-dev pandoc


Create Flower Dev Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the `Flower repository <https://github.com/adap/flower>`_ from
GitHub::

  $ git clone git@github.com:adap/flower.git
  $ cd flower

2. Let's create the Python environment for all-things Flower. If you wish to use :code:`pyenv`, we provide two convenience scripts that you can use. If you prefer using something else than :code:`pyenv`, create a new environment, activate and skip to the last point where all packages are installed.

* If you don't have :code:`pyenv` installed, the following script that will install it, set it up, and create the virtual environment (with :code:`Python 3.8.17` by default)::

  $ ./dev/setup-defaults.sh <version> # once completed, run the bootstrap script

* If you already have :code:`pyenv` installed (along with the :code:`pyenv-virtualenv` plugin), you can use the following convenience script (with :code:`Python 3.8.17` by default)::

  $ ./dev/venv-create.sh <version> # once completed, run the `bootstrap.sh` script

3. Install the Flower package in development mode (think
:code:`pip install -e`) along with all necessary dependencies::

  (flower-<version>) $ ./dev/bootstrap.sh


Convenience Scripts
-------------------

The Flower repository contains a number of convenience scripts to make
recurring development tasks easier and less error-prone. See the :code:`/dev`
subdirectory for a full list. The following scripts are amongst the most
important ones:

Create/Delete Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  $ ./dev/venv-create.sh <version> # Default is 3.8.17
  $ ./dev/venv-delete.sh <version> # Default is 3.8.17

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

Add a pre-commit hook
~~~~~~~~~~~~~~~~~~~~~

Developers may integrate a pre-commit hook into their workflow utilizing the `pre-commit <https://pre-commit.com/#install>`_ library. The pre-commit hook is configured to execute two primary operations: ``./dev/format.sh`` and ``./dev/test.sh`` scripts.

There are multiple ways developers can use this:

1. Install the pre-commit hook to your local git directory by simply running:

   ::
      
      $ pre-commit install

   - Each ``git commit`` will trigger the execution of formatting and linting/test scripts.
   - If in a hurry, bypass the hook using ``--no-verify`` with the ``git commit`` command.
     ::
          
       $ git commit --no-verify -m "Add new feature"
    
2. For developers who prefer not to install the hook permanently, it is possible to execute a one-time check prior to committing changes by using the following command:
   
   ::

      $ pre-commit run --all-files
   
   This executes the formatting and linting checks/tests on all the files without modifying the default behavior of ``git commit``.

Run Github Actions (CI) locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developers could run the full set of Github Actions workflows under their local
environment by using `Act <https://github.com/nektos/act>`_. Please refer to
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
`Pandoc <https://pandoc.org/installing.html>`_ needs to be installed on the system.
