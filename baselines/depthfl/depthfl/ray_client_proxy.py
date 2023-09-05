from typing import Callable, Dict, Optional, cast
from logging import ERROR
import ray

from flwr import common
from flwr.client import Client, ClientLike
from depthfl.client import to_client
from flwr.client.client import (
    maybe_call_fit,
)
from flwr.simulation.ray_transport.ray_client_proxy import RayClientProxy
from flwr.common.logger import log

ClientFn = Callable[[str], ClientLike]

class RayClientProxy_FedDyn(RayClientProxy):
    
    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
   
        """Train model parameters on the locally held dataset."""
        future_fit_res = launch_and_fit.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
        try:
            res = ray.get(future_fit_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
        return cast(
            common.FitRes,
            res,
        )
    

@ray.remote
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )

def _create_client(client_fn: ClientFn, cid: str) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid)
    return to_client(client_like=client_like)