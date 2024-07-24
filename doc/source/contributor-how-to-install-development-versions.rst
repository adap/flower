Install development versions
============================

Install development versions of Flower
--------------------------------------

Using Poetry (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install a ``flwr`` pre-release from PyPI: update the ``flwr`` dependency in ``pyproject.toml`` and then reinstall (don't forget to delete ``poetry.lock`` (``rm poetry.lock``) before running ``poetry install``).

- ``flwr = { version = "1.0.0a0", allow-prereleases = true }`` (without extras)
- ``flwr = { version = "1.0.0a0", allow-prereleases = true, extras = ["simulation"] }`` (with extras)

Install ``flwr`` from a local copy of the Flower source code via ``pyproject.toml``:

- ``flwr = { path = "../../", develop = true }`` (without extras)
- ``flwr = { path = "../../", develop = true, extras = ["simulation"] }`` (with extras)

Install ``flwr`` from a local wheel file via ``pyproject.toml``:

- ``flwr = { path = "../../dist/flwr-1.8.0-py3-none-any.whl" }`` (without extras)
- ``flwr = { path = "../../dist/flwr-1.8.0-py3-none-any.whl", extras = ["simulation"] }`` (with extras)

Please refer to the Poetry documentation for further details: `Poetry Dependency Specification <https://python-poetry.org/docs/dependency-specification/>`_

Using pip (recommended on Colab)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install a ``flwr`` pre-release from PyPI:

- ``pip install -U --pre flwr`` (without extras)
- ``pip install -U --pre 'flwr[simulation]'`` (with extras)

Python packages can be installed from git repositories. Use one of the following commands to install the Flower directly from GitHub.

Install ``flwr`` from the default GitHub branch (``main``):

- ``pip install flwr@git+https://github.com/adap/flower.git`` (without extras)
- ``pip install 'flwr[simulation]@git+https://github.com/adap/flower.git'`` (with extras)

Install ``flwr`` from a specific GitHub branch (``branch-name``):

- ``pip install flwr@git+https://github.com/adap/flower.git@branch-name`` (without extras)
- ``pip install 'flwr[simulation]@git+https://github.com/adap/flower.git@branch-name'`` (with extras)


Open Jupyter Notebooks on Google Colab
--------------------------------------

Open the notebook ``doc/source/tutorial-series-get-started-with-flower-pytorch.ipynb``:

- https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-series-get-started-with-flower-pytorch.ipynb

Open a development version of the same notebook from branch `branch-name` by changing ``main`` to ``branch-name`` (right after ``blob``):

- https://colab.research.google.com/github/adap/flower/blob/branch-name/doc/source/tutorial-series-get-started-with-flower-pytorch.ipynb

Install a `whl` on Google Colab:

1. In the vertical icon grid on the left hand side, select ``Files`` > ``Upload to session storage``
2. Upload the whl (e.g., ``flwr-1.8.0-py3-none-any.whl``)
3. Change ``!pip install -q 'flwr[simulation]' torch torchvision matplotlib`` to ``!pip install -q 'flwr-1.8.0-py3-none-any.whl[simulation]' torch torchvision matplotlib``
