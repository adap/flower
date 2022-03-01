import flwr as fl
from typing import List, Optional


class BatteryCriterion(fl.server.criterion.Criterion):
    """We will not sample clients which have a low batter level."""

    def select(self, client):
        ins = fl.common.PropertiesIns(config={})
        result = client.get_properties(ins)
        props = result.properties

        print(f"Received props res: {result}")

        if props is not None and "battery_level" in props:
            return props["battery_level"] > 0.5
        return True


class CustomStrategy(fl.server.strategy.FedAvg):
    def configure_fit(self, rnd, parameters, client_manager):
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(rnd)
        fit_ins = fl.common.typing.FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            criterion=BatteryCriterion(),
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    # Not implemented / inherited from FedAvg:
    # def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures:
    #     pass

    # Not implemented / inherited from FedAvg:
    # def aggregate_evaluate(self, rnd: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures):
    #     pass

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List,
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
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}"
        )

        # Call aggregate_evaluate from base class (FedAvg)
        params, _ = super().aggregate_evaluate(rnd, results, failures)
        return params, {"accuracy": accuracy_aggregated}


if __name__ == "__main__":

    # Initialize strategy
    strategy = CustomStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0,
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 3},
        strategy=strategy,
    )
