"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple, Type, Callable
from pathlib import Path
from collections import OrderedDict, defaultdict
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes,Parameters,
    Scalar, parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.strategy import FedAvg 
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.strategy import Strategy

class ServerInitializationStrategy(Strategy):
    """Server FL Parameter Initialization strategy implementation."""

    def __init__(
        self,
        create_model: Callable[[Dict[str, Any]], nn.Module],
        config: Dict[str, Any] = {},
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        """
            Class to initialize the server model parameters.

            Args:
                create_model: Function to create the model.
                config: Configuration dictionary.
        """
        self.config = config
        self.model = create_model

    def initialize_parameters(
            self, 
            client_manager: ClientManager,
        ) -> Optional[Parameters]:
        """Initialize the (global) model parameters.

        Returns:
            If parameters are returned, server will treat these as the initial global model parameters.
        """
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if initial_parameters is None and self.model is not None:
            initial_parameters = [val.cpu().numpy() for _, val in self.model.body.state_dict().items()]
        else:
            raise ValueError("No initial parameters provided.")

        if isinstance(initial_parameters, list):
            initial_parameters = ndarrays_to_parameters(initial_parameters)
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training. Adds the global head to the aggregated global body.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. 
        """
        # Same as superclass method but adds the head
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        weights = parameters_to_ndarrays(parameters=parameters)

        # Add head parameters to received body parameters
        weights.extend([val.cpu().numpy() for _, val in self.model.head.state_dict().items()])
        parameters = ndarrays_to_parameters(weights)
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
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: Client manager which holds all currently connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. 
        """
        # Same as superclass method but adds the head
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        # if self.eval_fn is not None:
        if self.evaluate_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        weights = parameters_to_ndarrays(parameters=parameters)

        # Add head parameters to received body parameters
        weights.extend([val.cpu().numpy() for _, val in self.model.head.state_dict().items()])
        parameters = ndarrays_to_parameters(weights=weights)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if server_round >= 0:
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
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters, set the global model parameters and save the global model.

        Args:
            rnd: The current round of federated learning.
            results: Successful updates from the previously selected and configured clients. 
            failures: Exceptions that occurred while the server was waiting for client updates.
        Returns:
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        agg_params, agg_metrics = super().aggregate_fit(server_round=server_round, results=results, failures=failures)

        # Update Server Model
        parameters = parameters_to_ndarrays(agg_params)
        model_keys = [k for k in self.model.state_dict().keys() if k.startswith("body")]
        model_keys = [k.replace("body.", "") for k in model_keys]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # self.model.set_parameters(state_dict)
        self.model.body.load_state_dict(state_dict, strict=True)

        return agg_params, agg_metrics

class StoreHistoryStrategy(Strategy):
    """Server FL history storage per training/evaluation round strategy implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hist: Dict[str, Dict[str, Any]] = {
            "trn": defaultdict(dict),
            "tst": defaultdict(dict)
        }
    
class StoreMetricsStrategy(StoreHistoryStrategy):
    """Server FL metrics storage per training/evaluation round strategy implementation."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters and store the train aggregated metrics.

        Args:
            rnd: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients.
            failures: Exceptions that occurred while the server was waiting for client updates.
        Returns:
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        aggregates = super().aggregate_fit(server_round=server_round, results=results, failures=failures)

        self.hist["trn"][server_round] = {k.cid: {"num_examples": v.num_examples, **v.metrics} for k, v in results}

        return aggregates

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters and store the evaluation aggregated metrics.

        Args:
            rnd: The current round of federated learning.
            results: Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: Exceptions that occurred while the server
                was waiting for client updates.
        Returns:
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """

        aggregates = super().aggregate_evaluate(server_round=server_round, results=results, failures=failures)

        self.hist["tst"][server_round] = {
            k.cid: {"num_examples": v.num_examples, "loss": v.loss, **v.metrics} for k, v in results
        }

        return aggregates
    
class StoreSelectedClientsStrategy(StoreHistoryStrategy):
    """Server FL selected client storage per training/evaluation round strategy implementation."""

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training and save the selected clients.

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
        result = super().configure_fit(server_round=server_round, parameters=parameters, client_manager=client_manager)

        if server_round not in self.hist["trn"].keys():
            self.hist["trn"][server_round] = {}

        self.hist["trn"][server_round]["selected_clients"] = [client.cid for client, _ in result]

        # Return client/config pairs
        return result

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation and save the selected clients.

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
        result = super().configure_evaluate(server_round=server_round, parameters=parameters, client_manager=client_manager)

        if server_round not in self.hist["tst"].keys():
            self.hist["tst"][server_round] = {}

        self.hist["tst"][server_round]["selected_clients"] = [client.cid for client, _ in result]

        # Return client/config pairs
        return result

class FederatedServerPipelineStrategy(
    StoreSelectedClientsStrategy,
    StoreMetricsStrategy,
    ServerInitializationStrategy
):
    pass

class AggregateBodyStrategyPipeline(
    FederatedServerPipelineStrategy,
    # AggregateBodyStrategy,
    FedAvg
):
    pass