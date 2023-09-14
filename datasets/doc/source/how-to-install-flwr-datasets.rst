Installation
============

Python Version
--------------

Flower Datasets requires `Python 3.8 <https://docs.python.org/3.8/>`_ or above.


Install stable release (pip)
----------------------------

Stable releases are available on `PyPI <https://pypi.org/project/flwr_datasets/>`_::

  python -m pip install flwr_datasets

For Image Datasets (e.g. MNIST, CIFAR10) ``flwr_datasets`` should be installed with the ``image`` extra::

  python -m pip install flwr_datasets[vision]

For Audio Datasets (e.g. Speech Command) ``flwr_datasets`` should be installed with the ``audio`` extra::

  python -m pip install flwr_datasets[audio]


Verify installation
-------------------

The following command can be used to verify if Flower Datasets was successfully installed. If everything worked, it should print the version of Flower Datasets to the command line::

  python -c "import flwr_datasets;print(flwr_datasets.__version__)"
  0.0.1

