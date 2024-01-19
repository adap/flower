Save and load model checkpoints
===============================

Flower does not automatically save model updates on the server-side. This how-to guide describes the steps to save (and load) model checkpoints in Flower.


Model checkpointing
-------------------

Model updates can be persisted on the server-side by customizing :code:`Strategy` methods.
Implementing custom strategies is always an option, but for many cases it may be more convenient to simply customize an existing strategy.
The following code example defines a new :code:`SaveModelStrategy` which customized the existing built-in :code:`FedAvg` strategy.
In particular, it customizes :code:`aggregate_fit` by calling :code:`aggregate_fit` in the base class (:code:`FedAvg`).
It then continues to save returned (aggregated) weights before it returns those aggregated weights to the caller (i.e., the server):

.. code-block:: python

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
            if aggregated_parameters is not None:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Save aggregated_ndarrays
                print(f"Saving round {server_round} aggregated_ndarrays...")
                np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

            return aggregated_parameters, aggregated_metrics

    # Create strategy and run server
    strategy = SaveModelStrategy(
        # (same arguments as FedAvg here)
    )
    fl.server.start_server(strategy=strategy)


Save and load PyTorch checkpoints
---------------------------------

Similar to the previous example but with a few extra steps, we'll show how to 
store a PyTorch checkpoint we'll use the ``torch.save`` function.
Firstly, ``aggregate_fit`` returns a ``Parameters`` object that has to be transformed into a list of NumPy ``ndarray``'s, 
then those are transformed into the PyTorch ``state_dict`` following the ``OrderedDict`` class structure.

.. code-block:: python

    net = cifar.Net().to(DEVICE)
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using weighted average and store checkpoint"""

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
                
                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                net.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(net.state_dict(), f"model_round_{server_round}.pth")

            return aggregated_parameters, aggregated_metrics

To load your progress, you simply append the following lines to your code. Note that this will iterate over all saved checkpoints and load the latest one:

.. code-block:: python

    list_of_files = [fname for fname in glob.glob("./model_round_*")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    net.load_state_dict(state_dict)
    state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
    parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)

Return/use this object of type ``Parameters`` wherever necessary, such as in the ``initial_parameters`` when defining a ``Strategy``.