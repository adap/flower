Installing Flower
=================


Python Version
--------------

Flower requires `Python 3.7 <https://docs.python.org/3.7/>`_ or above. `Python 3.6 <https://docs.python.org/3.6/>`_ is still supported in Flower 0.18, but Flower 0.19 will drop support for Python 3.6.


Install Stable Release
----------------------

Stable releases are available on `PyPI <https://pypi.org/>`_::

  $ pip install flwr

To use Flower with the Virtual Client Engine, install the necessary dependencies for using `start_simulation`:

  $ pip install flwr[simulation]


Verify Installation
-------------------

The following command can be used to verify the Flower installation:

  $ python3 -c "import flwr;print(flwr.__version__)"
  0.18.0

The installation should work as expected if there are no errors and the current Flower version gets printed on the command line.


Install Nightly Release
-----------------------

The latest (potentially unstable) changes in Flower are available as nightly releases::

  $ pip install flwr-nightly

Flower nightly releases 

  $ pip install flwr-nightly[simulation]


Contributor Install
-------------------

Contributors can install development versions of Flower in multiple ways: `Contributor Installation <contributor-installation.html>`_
