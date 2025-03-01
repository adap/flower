"""Federated Hardthresholding (FedHT)."""

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from fedht.aggregate import aggregate_hardthreshold

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class FedHT(FedAvg):
    """Federated Hardthreshold strategy.

    Implementation based on https://arxiv.org/abs/2101.00052

    Parameters
    ----------
    num_keep : int, number of parameters to keep different from 0. Defaults to 5.
    iterht : boolean, if true, utilizes the Fed-IterHT strategy. Defaults to False.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        num_keep: int = 5,
        iterht: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_keep = num_keep
        self.iterht = iterht

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedHT(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # no in-place aggregation for FedHT
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # use hardthresholding
        aggregated_ndarrays = aggregate_hardthreshold(
            weights_results, self.num_keep, self.iterht
        )
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
