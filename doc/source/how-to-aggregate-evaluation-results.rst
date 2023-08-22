Aggregate evaluation results
============================

The Flower server does not prescribe a way to aggregate evaluation results, but it enables the user to fully customize result aggregation.


Aggregate Custom Evaluation Results
-----------------------------------

The same :code:`Strategy`-customization approach can be used to aggregate custom evaluation results coming from individual clients.
Clients can return custom metrics to the server by returning a dictionary:

.. code-block:: python

    class CifarClient(fl.client.NumPyClient):

        def get_parameters(self, config):
            # ...

        def fit(self, parameters, config):
            # ...

        def evaluate(self, parameters, config):
            """Evaluate parameters on the locally held test set."""

            # Update local model with global parameters
            self.model.set_weights(parameters)

            # Evaluate global model parameters on the local test data
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test)

            # Return results, including the custom accuracy metric
            num_examples_test = len(self.x_test)
            return loss, num_examples_test, {"accuracy": accuracy}

The server can then use a customized strategy to aggregate the metrics provided in these dictionaries:

.. code-block:: python

    class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
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
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

            # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return aggregated_loss, {"accuracy": aggregated_accuracy}

    # Create strategy and run server
    strategy = AggregateCustomMetricStrategy(
        # (same arguments as FedAvg here)
    )
    fl.server.start_server(strategy=strategy)
