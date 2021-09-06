Saving Progress
===============

The Flower server does not prescribe a way to persist model updates or evaluation results.
Flower does not (yet) automatically save model updates on the server-side.
It's on the roadmap to provide a built-in way of doing this, 

Model Checkpointing
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
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Optional[fl.common.Weights]:
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            if aggregated_weights is not None:
                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
            return aggregated_weights

    # Create strategy and run server
    strategy = SaveModelStrategy(
        # (same arguments as FedAvg here)
    )
    fl.server.start_server(strategy=strategy)

Aggregate Custom Evaluation Results
-----------------------------------

The same :code:`Strategy`-customization approach can be used to aggregate custom evaluation results coming from individual clients.
Clients can return custom metrics to the server by returning a dictionary:

.. code-block:: python

    class CifarClient(fl.client.NumPyClient):

        def get_parameters(self):
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
            rnd: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
        ) -> Optional[float]:
            """Aggregate evaluation losses using weighted average."""
            if not results:
                return None
            
            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
            
            # Aggregate and print custom metric
            accuracy_aggregated = sum(accuracies) / sum(examples)
            print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

            # Call aggregate_evaluate from base class (FedAvg)
            return super().aggregate_evaluate(rnd, results, failures)

    # Create strategy and run server
    strategy = AggregateCustomMetricStrategy(
        # (same arguments as FedAvg here)
    )
    fl.server.start_server(strategy=strategy)
