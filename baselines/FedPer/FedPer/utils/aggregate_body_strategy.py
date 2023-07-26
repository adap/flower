import torch

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import OrderedDict
from flwr.common import (
    EvaluateIns, FitIns, FitRes, Parameters,
    Scalar, parameters_to_weights, weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from FedPer.utils.initialization_strategy import ServerInitializationStrategy


class AggregateBodyStrategy(ServerInitializationStrategy):
    """Body Aggregation strategy implementation."""

    def __init__(self, save_path: Path = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        if save_path is not None:
            self.save_path = save_path / "models"
            self.save_path.mkdir(parents=True, exist_ok=True)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training. Adds the global head to the aggregated global body.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        # Same as superclass method but adds the head

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)

        weights = parameters_to_weights(parameters=parameters)

        # Add head parameters to received body parameters
        weights.extend([val.cpu().numpy() for _, val in self.model.head.state_dict().items()])

        parameters = weights_to_parameters(weights=weights)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        # Same as superclass method but adds the head

        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)

        weights = parameters_to_weights(parameters=parameters)

        # Add head parameters to received body parameters
        weights.extend([val.cpu().numpy() for _, val in self.model.head.state_dict().items()])

        parameters = weights_to_parameters(weights=weights)

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

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
        parameters = parameters_to_weights(agg_params)
        model_keys = [k for k in self.model.state_dict().keys() if k.startswith("_body")]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.set_parameters(state_dict)

        if self.save_path is not None:
            # Save Model
            torch.save(self.model, self.save_path / f"model-ep_{rnd}.pt")


        return agg_params, agg_metrics
