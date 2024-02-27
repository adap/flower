from typing import Dict, List, Optional, Tuple, Union
from flwr.common import Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import flwr as fl
import logging
from prometheus_client import Gauge

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


class FedCustom(fl.server.strategy.FedAvg):
    def __init__(
        self, accuracy_gauge: Gauge = None, loss_gauge: Gauge = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.accuracy_gauge = accuracy_gauge
        self.loss_gauge = loss_gauge

    def __repr__(self) -> str:
        return "FedCustom"

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracy using weighted average."""

        if not results:
            return None, {}

        # Calculate weighted average for loss using the provided function
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Calculate weighted average for accuracy
        accuracies = [
            evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy_aggregated = (
            sum(accuracies) / sum(examples) if sum(examples) != 0 else 0
        )

        # Update the Prometheus gauges with the latest aggregated accuracy and loss values
        self.accuracy_gauge.set(accuracy_aggregated)
        self.loss_gauge.set(loss_aggregated)

        metrics_aggregated = {"loss": loss_aggregated, "accuracy": accuracy_aggregated}

        return loss_aggregated, metrics_aggregated
