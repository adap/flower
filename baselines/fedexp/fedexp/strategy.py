from typing import List, Tuple, Union, Optional, Dict

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedExP(FedAvg):
    pass
    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, FitRes]],
    #     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     """Aggregate fit results using FedProx."""
    #     # Aggregate results
    #     parameters, num_examples = super().aggregate_fit(server_round,
    #                                                      results,
    #                                                      failures)
    #
    #     # Return updated parameters
    #     return parameters, {"num_examples": num_examples}

