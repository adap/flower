"""fedprox: A Flower Baseline."""

from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgWithStragglerDrop(FedAvg):
    """Custom FedAvg which discards updates from stragglers."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ):
        """Discard all the models sent by the clients that were stragglers."""
        # Record which client was a straggler in this round
        stragglers_mask = [res.metrics["is_straggler"] for _, res in results]

        # keep those results that are not from stragglers
        results = [res for i, res in enumerate(results) if not stragglers_mask[i]]

        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        return super().aggregate_fit(server_round, results, failures)
