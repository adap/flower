Aggregate Evaluation Results
============================

The Flower server does not prescribe a way to aggregate evaluation results, but it
enables the user to fully customize result aggregation.

Aggregate Custom Evaluation Results
-----------------------------------

At the server-side, users can either define a custom ``Callback`` evaluation function as
part of their strategy to aggregate custom evaluation results from individual clients or
use a custom ``Strategy`` approach.

Clients can return custom metrics to the server by returning a dictionary:

.. code-block:: python

    class CifarClient(fl.client.NumPyClient):

        def get_parameters(self, config):
            # ...
            pass

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

Custom Callback
~~~~~~~~~~~~~~~

The server can use a custom evaluate callback function to operate on clients'
dictionaries:

.. code-block:: python

    # Define metric aggregation function
    def custom_evaluation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": float(sum(accuracies) / sum(examples))}


    def server_fn(context: Context):
        # Read federation rounds from config
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        # Define the strategy
        strategy = FedAvg(
            # (same arguments as FedAvg here)
            evaluate_metrics_aggregation_fn=custom_evaluation_fn,  # <-- pass the custom evaluation function here
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(
            config=config,
            strategy=strategy,
        )


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Custom Strategy
~~~~~~~~~~~~~~~

Alternatively, the server can use a customized strategy to aggregate the metrics in the
dictionaries:

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
    app = ServerApp(
        server_fn=server_fn,
    )
