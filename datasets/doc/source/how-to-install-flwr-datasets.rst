Installation
============

Python Version
--------------

Flower Datasets requires `Python 3.8 <https://docs.python.org/3.8/>`_ or above.


Install stable release (pip)
----------------------------

Stable releases are available on `PyPI <https://pypi.org/project/flwr_datasets/>`_

.. code-block:: bash

  python -m pip install flwr-datasets

For vision datasets (e.g. MNIST, CIFAR10) ``flwr-datasets`` should be installed with the ``vision`` extra

.. code-block:: bash

  python -m pip install flwr_datasets[vision]

For audio datasets (e.g. Speech Command) ``flwr-datasets`` should be installed with the ``audio`` extra

.. code-block:: bash

  python -m pip install flwr_datasets[audio]


Verify installation
-------------------

The following command can be used to verify if Flower Datasets was successfully installed:

.. code-block:: bash

  python -c "import flwr_datasets;print(flwr_datasets.__version__)"

If everything worked, it should print the version of Flower Datasets to the command line:

.. code-block:: none

  0.3.0

