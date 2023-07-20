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
    Scalar, parameters_to_weights, weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.strategy import Strategy
from flwr.server.strategy.aggregate import FedAvg

#from fedpfl.federated_learning.constants import Algorithms
#from fedpfl.model.model_split import ModelSplit


class ServerInitializationStrategy(Strategy):
    """Server FL Parameter Initialization strategy implementation."""

    def __init__(
        self,
        create_model: Callable[[Dict[str, Any]], nn.Module],
        config: Dict[str, Any] = {},
        algorithm: str = 'fedavg',
        has_fixed_head: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        assert algorithm.lower() in ['fedavg', 'fedper'], f"Algorithm {algorithm} not supported."
        self.config = config
        self.algorithm = algorithm
        self.model = model_split_class(model=create_model(config), has_fixed_head=has_fixed_head)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters.

        Args:
            client_manager: ClientManager. The client manager which holds all currently
                connected clients.

        Returns:
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if initial_parameters is None and self.model is not None:
            if self.algorithm.lower() == 'fedper' :
                initial_parameters = [val.cpu().numpy() for _, val in self.model.body.state_dict().items()]
            else:  # FedAvg
                initial_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        if isinstance(initial_parameters, list):
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters


class AggregateBodyStrategy(ServerInitializationStrategy):
    """Body Aggregation strategy implementation."""

    def __init__(
            self, 
            create_model: Callable[[Dict[str, Any]], nn.Module],
            save_path: Path = None, 
            config: Dict[str, Any] = {},
            algorithm : str = 'fedavg',
            *args: Any, 
            **kwargs: Any
        ) -> None:
        super().__init__(*args, **kwargs)
        """ 
            Class to aggregate the body of the model. 
        
            Args:
                save_path: Path to save the model.
                config: Configuration dictionary.
        """
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
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters and store the train aggregated metrics.

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
        aggregates = super().aggregate_fit(rnd=rnd, results=results, failures=failures)

        self.hist["trn"][rnd] = {k.cid: {"num_examples": v.num_examples, **v.metrics} for k, v in results}

        return aggregates

    def aggregate_evaluate(
        self,
        rnd: int,
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

        aggregates = super().aggregate_evaluate(rnd=rnd, results=results, failures=failures)

        self.hist["tst"][rnd] = {
            k.cid: {"num_examples": v.num_examples, "loss": v.loss, **v.metrics} for k, v in results
        }

        return aggregates
    
class StoreSelectedClientsStrategy(StoreHistoryStrategy):
    """Server FL selected client storage per training/evaluation round strategy implementation."""

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
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
        result = super().configure_fit(rnd=rnd, parameters=parameters, client_manager=client_manager)

        if rnd not in self.hist["trn"].keys():
            self.hist["trn"][rnd] = {}

        self.hist["trn"][rnd]["selected_clients"] = [client.cid for client, _ in result]

        # Return client/config pairs
        return result

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
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
        result = super().configure_evaluate(rnd=rnd, parameters=parameters, client_manager=client_manager)

        if rnd not in self.hist["tst"].keys():
            self.hist["tst"][rnd] = {}

        self.hist["tst"][rnd]["selected_clients"] = [client.cid for client, _ in result]

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
    AggregateBodyStrategy,
    FedAvg
):
    pass