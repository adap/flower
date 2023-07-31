Saving and Loading PyTorch Checkpoints
======================================

Similarly to the previous example but with a few extra steps, we'll show how to 
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
