from abc import ABC, abstractmethod
import concurrent.futures
from typing import Dict, List, Tuple, Optional
from flwr.common.typing import Scalar, Parameters, FitRes, SAServerMessageCarrier, SAClientMessageCarrier
from flwr.server.client_proxy import ClientProxy
from flwr.server.grpc_server.grpc_client_proxy import GrpcClientProxy


FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]

SecureAggregationResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, SAClientMessageCarrier]], List[BaseException]
]


class SecureAggregationFitRound(ABC):
    @abstractmethod
    def fit_round(self, server, rnd: int) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar],
                                                            FitResultsAndFailures]]:
        """fit round"""

    @staticmethod
    def sa_request(requests: List[Tuple[ClientProxy, SAServerMessageCarrier]]) -> SecureAggregationResultsAndFailures:
        return parallel(_request, requests)


def _request(client: GrpcClientProxy, ins: SAServerMessageCarrier):
    return client, client.sa_request(ins)


def parallel(fn, requests: List[Tuple[ClientProxy, SAServerMessageCarrier]]):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(lambda p: fn(*p), (client, ins))
            for client, ins in requests
        ]
        concurrent.futures.wait(futures)
    results = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


