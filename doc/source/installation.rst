Installing Flower
=================


Python Version
--------------

Flower requires `Python 3.7 <https://docs.python.org/3.7/>`_ or above.


Install Stable Release
----------------------

Stable releases are available on `PyPI <https://pypi.org/>`_::

  pip install flwr

For simulations that use the Virtual Client Engine, ``flwr`` should be installed with the ``simulation`` extra::

  pip install flwr[simulation]


Install Nightly Release
-----------------------

The latest (potentially unstable) changes in Flower are available as nightly releases::

  pip install flwr-nightly

For simulations that use the Virtual Client Engine, ``flwr-nightly`` should be installed with the ``simulation`` extra::

  pip install flwr-nightly[simulation]


Install from GitHub
-------------------

Python packages can be installed from git repositories. Use the following
command to install the latest version of Flower directly from GitHub::

  pip install git+https://github.com/adap/flower.git

One can also install a specific commit::

  pip install git+https://github.com/adap/flower.git@9cc383cddb7dcb0cc41b5a3559106887ba1c34f8
