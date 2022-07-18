Upgrade to Flower 1.0
=====================

Flower 1.0 is finally here! Along with new features, Flower 1.0 provides a stable foundation for future growth. Compared to Flower 0.19 (and other 0.x series releases), there are a few breaking changes that might make it necessary to change the code of existing 0.x-series projects.

Install update
--------------

Installing Flower ``1.0.0a0`` requires you to tell the dependency management tool (pip, Poetry, ...) to install pre-releases. Here's how that works for pip and Poetry:

- pip: add ``-U --pre`` when installing.

  - ``python -m pip install -U --pre flwr`` (when using ``start_server`` and ``start_client``)
  - ``python -m pip install -U --pre flwr[simulation]`` (when using ``start_simulation``)

- Poetry: update the ``flwr`` dependency in ``pyproject.toml`` and then reinstall (don't forget to delete ``poetry.lock`` (``rm poetry.lock``) before running ``poetry install``).

  - ``flwr = { version = "1.0.0a0", allow-prereleases = true }`` (when using ``start_server`` and ``start_client``)
  - ``flwr = { version = "1.0.0a0", allow-prereleases = true, extras = ["simulation"] }`` (when using ``start_simulation``)

Required changes
----------------

A few breaking changes require small manual updates:

- Subclasses of ``NumPyClient``: change ``def get_parameters(self):``` to ``def get_parameters(self, config):``
- Subclasses of ``Client``: change ``def get_parameters(self):``` to ``def get_parameters(self, ins: GetParametersIns):``
- Rename ``parameters_to_weights`` to ``parameters_to_ndarrays`` and ``weights_to_parameters`` to ``ndarrays_to_parameters``
- ``start_simulation``: change ``num_rounds=1`` to ``config={"num_rounds": 1}``
- Strategy initialization: if the strategy relies on the default values for ``fraction_fit`` and ``fraction_eval``, set ``fraction_fit`` and ``fraction_eval`` manually to ``0.1``. Projects that do not manually create a strategy (by calling ``start_server`` or ``start_simulation`` without passing a strategy instance) should now initialize FedAvg with ``fraction_fit`` and ``fraction_eval`` set to ``0.1``.

Optional improvements
---------------------

Along with the necessary changes above, there are a number of potential improvements that just became possible:

- Remove "placeholder" methods from subclasses of ``Client`` or ``NumPyClient``. If you, for example, use server-side evaluation, then empy placeholder implementations of ``evaluate`` are no longer necessary.
- Configure the round timeout via ``start_simulation``: ``start_simulation(..., config={"num_rounds": 3, "round_timeout": 600.0}, ...)``
