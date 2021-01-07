Changelog
=========

Unreleased
----------

What's new?

* New built-in strategies (`#549 <https://github.com/adap/flower/pull/549>`_)
    * (abstract) FedOpt
    * FedAdagrad
* New example: PyTorch From Centralized To Federated (`#549 <https://github.com/adap/flower/pull/549>`_)
* Improved documetation
    * New documetation theme (`#551 <https://github.com/adap/flower/pull/551>`_)
    * New API reference (`#554 <https://github.com/adap/flower/pull/554>`_)
    * Updated examples documentation (`#549 <https://github.com/adap/flower/pull/549>`_)
    * Removed obsolete documentation (`#548 <https://github.com/adap/flower/pull/548>`_)

Bugfix:

* :code:`Server.fit` does not disconnect clients when finished, disconnecting the clients is now handled in :code:`flwr.server.start_server` (`#553 <https://github.com/adap/flower/pull/553>`_, `#540 <https://github.com/adap/flower/issues/540>`_).


v0.12.0 (2020-12-07)
--------------------

Important changes:

* Added an example for embedded devices (`#507 <https://github.com/adap/flower/pull/507>`_)
* Added a new NumPyClient (in addition to the existing KerasClient) (`#504 <https://github.com/adap/flower/pull/504>`_, `#508 <https://github.com/adap/flower/pull/508>`_)
* Deprecated `flwr_examples` package and started to migrate examples into the top-level `examples` directory (`#494 <https://github.com/adap/flower/pull/494>`_, `#512 <https://github.com/adap/flower/pull/512>`_)


v0.11.0 (2020-11-30)
--------------------

Incompatible changes:

* Renamed strategy methods (`#486 <https://github.com/adap/flower/pull/486>`_) to unify the naming of Flower's public APIs. Other public methods/functions (e.g., every method in :code:`Client`, but also :code:`Strategy.evaluate`) do not use the :code:`on_` prefix, which is why we're removing it from the four methods in Strategy. To migrate rename the following :code:`Strategy` methods accordingly:
    * :code:`on_configure_evaluate` => :code:`configure_evaluate`
    * :code:`on_aggregate_evaluate` => :code:`aggregate_evaluate`
    * :code:`on_configure_fit` => :code:`configure_fit`
    * :code:`on_aggregate_fit` => :code:`aggregate_fit`

Important changes:

* Deprecated :code:`DefaultStrategy` (`#479 <https://github.com/adap/flower/pull/479>`_). To migrate use :code:`FedAvg` instead.
* Simplified examples and baselines (`#484 <https://github.com/adap/flower/pull/484>`_).
* Removed presently unused :code:`on_conclude_round` from strategy interface (`#483 <https://github.com/adap/flower/pull/483>`_).
* Set minimal Python version to 3.6.1 instead of 3.6.9 (`#471 <https://github.com/adap/flower/pull/471>`_).
* Improved :code:`Strategy` docstrings (`#470 <https://github.com/adap/flower/pull/470>`_).
