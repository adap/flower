Install Flower
==============


Python version
--------------

Flower requires at least `Python 3.8 <https://docs.python.org/3.8/>`_, but `Python 3.10 <https://docs.python.org/3.10/>`_ or above is recommended.


Install stable release
----------------------

Using pip
~~~~~~~~~

Stable releases are available on `PyPI <https://pypi.org/project/flwr/>`_::

  python -m pip install flwr

For simulations that use the Virtual Client Engine, ``flwr`` should be installed with the ``simulation`` extra::

  python -m pip install "flwr[simulation]"


Using conda (or mamba)
~~~~~~~~~~~~~~~~~~~~~~

Flower can also be installed from the ``conda-forge`` channel.

If you have not added ``conda-forge`` to your channels, you will first need to run the following::

  conda config --add channels conda-forge
  conda config --set channel_priority strict

Once the ``conda-forge`` channel has been enabled, ``flwr`` can be installed with ``conda``::

  conda install flwr

or with ``mamba``::

  mamba install flwr


Verify installation
-------------------

The following command can be used to verify if Flower was successfully installed. If everything worked, it should print the version of Flower to the command line::

.. code-block:: bash
   :substitutions:

  python -c "import flwr;print(flwr.__version__)"
  |current_flwr_version|


Advanced installation options
-----------------------------

Install via Docker
~~~~~~~~~~~~~~~~~~

:doc:`How to run Flower using Docker <how-to-run-flower-using-docker>`

Install pre-release
~~~~~~~~~~~~~~~~~~~

New (possibly unstable) versions of Flower are sometimes available as pre-release versions (alpha, beta, release candidate) before the stable release happens::

  python -m pip install -U --pre flwr

For simulations that use the Virtual Client Engine, ``flwr`` pre-releases should be installed with the ``simulation`` extra::

  python -m pip install -U --pre 'flwr[simulation]'

Install nightly release
~~~~~~~~~~~~~~~~~~~~~~~

The latest (potentially unstable) changes in Flower are available as nightly releases::

  python -m pip install -U flwr-nightly

For simulations that use the Virtual Client Engine, ``flwr-nightly`` should be installed with the ``simulation`` extra::

  python -m pip install -U flwr-nightly[simulation]
