import torch

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import OrderedDict
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from FedPer.utils.initialization_strategy import ServerInitializationStrategy

class AggregateFullStrategy(ServerInitializationStrategy):
    """Full model aggregation strategy implementation."""

    def __init__(self, save_path: Path = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        if save_path is not None:
            self.save_path = save_path / "models"
            self.save_path.mkdir(parents=True, exist_ok=True)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters, set the global model parameters and save the global model.

        Args:
            rnd: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients. Not
                that not all previously selected clients are necessarily included in
                this list: a client might drop out and not submit a result. For each
                client that did not submit an update, there should be an `Exception`
                in `failures`.
            failures: Exceptions that occurred while the server was waiting for client
                updates.
        Returns:
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        agg_params, agg_metrics = super().aggregate_fit(rnd=rnd, results=results, failures=failures)

        # Update Server Model
        parameters = parameters_to_ndarrays(agg_params)
        model_keys = [k for k in self.model.state_dict().keys()
                    if k.startswith("_body") or k.startswith("_head")]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.set_parameters(state_dict)

        if self.save_path is not None:
            # Save Model
            torch.save(self.model, self.save_path / f"model-ep_{rnd}.pt")


        return agg_params, agg_metrics
