from typing import List, Tuple, Union

from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgWithStragglerDrop(FedAvg):
    """My customised FedAvg strategy which discards updates sent by clients that were
    flagged as stragglers."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Here we discard all the models sent by the clients that were stragglers in
        this round."""

        # Record which client was a straggler in this round
        stragglers_mask = [res.metrics["is_straggler"] for _, res in results]

        print(f"Num stragglers in round: {sum(stragglers_mask)}")

        # keep those results that are not from stragglers
        results = [res for i, res in enumerate(results) if not (stragglers_mask[i])]

        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        return super().aggregate_fit(server_round, results, failures)
