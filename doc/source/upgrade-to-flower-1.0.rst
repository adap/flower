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

Client
~~~~~~

- Subclasses of ``NumPyClient``: change ``def get_parameters(self):``` to ``def get_parameters(self, config):``
- Subclasses of ``Client``: change ``def get_parameters(self):``` to ``def get_parameters(self, ins: GetParametersIns):``

Strategies / ``start_server`` / ``start_simulation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Pass ``ServerConfig`` (instead of a dictionary) to ``start_server`` and ``start_simulation``. Here's an example:

  - Flower 0.19: ``start_server(..., config={"num_rounds": 3, "round_timeout": 600.0}, ...)``
  - Flower 1.0: ``start_server(..., config=flwr.server.ServerConfig(num_rounds=3, round_timeout=600.0), ...)``

- Replate ``num_rounds=1`` in ``start_simulation`` with the new ``config=ServerConfig(...)`` (see next item)
- Remove ``force_final_distributed_eval`` parameter from calls to ``start_server``. Distributed evaluation on all clients can be enabled by configuring the strategy to sample all clients for evaluation after the last round of training.
- Rename ``parameters_to_weights`` to ``parameters_to_ndarrays`` and ``weights_to_parameters`` to ``ndarrays_to_parameters``
- Strategy initialization: if the strategy relies on the default values for ``fraction_fit`` and ``fraction_evaluate``, set ``fraction_fit`` and ``fraction_evaluate`` manually to ``0.1``. Projects that do not manually create a strategy (by calling ``start_server`` or ``start_simulation`` without passing a strategy instance) should now initialize FedAvg with ``fraction_fit`` and ``fraction_evaluate`` set to ``0.1``.
- Rename ``rnd`` to ``server_round`` (``evaluate_fn``, ``configure_fit``, ``aggregate_fit``, ``configure_evaluate``, ``aggregate_evaluate``)
- Rename built-in strategy parameters (e.g., ``FedAvg``):

  - ``fraction_eval`` --> ``fraction_evaluate``
  - ``min_eval_clients`` --> ``min_evaluate_clients``
  - ``eval_fn`` --> ``evaluate_fn``

- Add ``server_round`` and ``config`` to ``evaluate_fn``:

  - Flower 0.19: ``def evaluate(parameters: NDArrays) -> Optional[Tuple[float, Dict[str, Scalar]]]:``
  - Flower 1.0: ``def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:``

Custom strategies
~~~~~~~~~~~~~~~~~

- The type of parameter ``failures`` has changed from ``List[BaseException]`` to ``List[Union[Tuple[ClientProxy, FitRes], BaseException]]`` (in ``aggregate_fit``) and ``List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]`` (in ``aggregate_evaluate``)
- The ``Strategy`` method ``evaluate`` now receives the current round of federated learning/evaluation as the first parameter:

  - Flower 0.19: ``def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:``
  - Flower 1.0: ``def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:``

Optional improvements
---------------------

Along with the necessary changes above, there are a number of potential improvements that just became possible:

- Remove "placeholder" methods from subclasses of ``Client`` or ``NumPyClient``. If you, for example, use server-side evaluation, then empy placeholder implementations of ``evaluate`` are no longer necessary.
- Configure the round timeout via ``start_simulation``: ``start_simulation(..., config=flwr.server.ServerConfig(num_rounds=3, round_timeout=600.0), ...)``
