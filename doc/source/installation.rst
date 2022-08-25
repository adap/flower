Installing Flower
=================


Python Version
--------------

Flower requires `Python 3.7 <https://docs.python.org/3.7/>`_ or above.


Install stable release
----------------------

Stable releases are available on `PyPI <https://pypi.org/project/flwr/>`_::

  python -m pip install flwr

For simulations that use the Virtual Client Engine, ``flwr`` should be installed with the ``simulation`` extra::

  python -m pip install flwr[simulation]


Verify installation
-------------------

The following command can be used to verfiy if Flower was successfully installed. If everything worked, it should print the version of Flower to the command line::

  python -c "import flwr;print(flwr.__version__)"
  1.1.0


Advanced installation options
-----------------------------

Install pre-release
~~~~~~~~~~~~~~~~~~~

New (possibly unstable) versions of Flower are sometimes available as pre-release versions (alpha, beta, release candidate) before the stable release happens::

  python -m pip install -U --pre flwr

For simulations that use the Virtual Client Engine, ``flwr`` pre-releases should be installed with the ``simulation`` extra::

  python -m pip install -U --pre flwr[simulation]

Install nightly release
~~~~~~~~~~~~~~~~~~~~~~~

The latest (potentially unstable) changes in Flower are available as nightly releases::

  python -m pip install -U flwr-nightly

For simulations that use the Virtual Client Engine, ``flwr-nightly`` should be installed with the ``simulation`` extra::

  python -m pip install -U flwr-nightly[simulation]
