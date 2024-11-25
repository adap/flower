:og:description: Aggregate custom evaluation results from federated clients in Flower using a strategy that applies weighted averaging for metrics like accuracy.
.. meta::
    :description: Aggregate custom evaluation results from federated clients in Flower using a strategy that applies weighted averaging for metrics like accuracy.

Aggregate evaluation results
============================

The Flower server does not prescribe a way to aggregate evaluation results, but it
enables the user to fully customize result aggregation.

Aggregate Custom Evaluation Results
-----------------------------------

The same ``Strategy``-customization approach can be used to aggregate custom evaluation
results coming from individual clients. Clients can return custom metrics to the server
by returning a dictionary:

.. code-block:: python

    from flwr.client import NumPyClient


    class FlowerClient(NumPyClient):

        def fit(self, parameters, config):
            # ...
            pass

        def evaluate(self, parameters, config):
            """Evaluate parameters on the locally held test set."""

            # Update local model with global parameters
            self.model.set_weights(parameters)

            # Evaluate global model parameters on the local test data
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test)

            # Return results, including the custom accuracy metric
            num_examples_test = len(self.x_test)
            return float(loss), num_examples_test, {"accuracy": float(accuracy)}

The server can then use a customized strategy to aggregate the metrics provided in these
dictionaries:

.. code-block:: python

    from flwr.server.strategy import FedAvg


    class AggregateCustomMetricStrategy(FedAvg):
        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Aggregate evaluation accuracy using weighted average."""

            if not results:
                return None, {}

            # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
                server_round, results, failures
            )

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            print(
                f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
            )

            # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}


    def server_fn(context: Context) -> ServerAppComponents:
        # Read federation rounds from config
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        # Define strategy
        strategy = AggregateCustomMetricStrategy(
            # (same arguments as FedAvg here)
        )

        return ServerAppComponents(
            config=config,
            strategy=strategy,  # <-- pass the custom strategy here
        )


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)
