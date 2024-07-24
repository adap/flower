"""FL server strategies."""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg
from torch import nn

from fedrep.implemented_models.cnn_cifar10 import CNNCifar10ModelSplit
from fedrep.implemented_models.cnn_cifar100 import CNNCifar100ModelSplit


class ServerInitializationStrategy(FedAvg):
    """Server FL Parameter Initialization strategy implementation."""

    # pylint: disable=R0913
    def __init__(
        self,
        *args: Any,
        model_split_class: Union[
            Type[CNNCifar10ModelSplit], Type[CNNCifar100ModelSplit]
        ],
        create_model_fn: Callable[[], nn.Module],
        algorithm: str = "fedrep",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm
        self.model = model_split_class(model=create_model_fn())


class AggregateFullStrategy(ServerInitializationStrategy):
    """Full model aggregation strategy implementation."""

    def __init__(self, *args, save_path: Path = Path(""), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = save_path if save_path != "" else None
        if save_path is not None:
            self.save_path = save_path / "models"
            self.save_path.mkdir(parents=True, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate received local parameters, set global model parameters and save.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients. Not
                that not all previously selected clients are necessarily included in
                this list: a client might drop out and not submit a result. For each
                client that did not submit an update, there should be an `Exception`
                in `failures`.
            failures: Exceptions that occurred while the server was waiting for client
                updates.

        Returns
        -------
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        agg_params, agg_metrics = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if agg_params is not None:
            # Update Server Model
            parameters = parameters_to_ndarrays(agg_params)
            model_keys = list(self.model.state_dict().keys())
            params_dict = zip(model_keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.set_parameters(state_dict)

        if self.save_path is not None:
            # Save Model
            torch.save(self.model, self.save_path / f"model-ep_{server_round}.pt")

        return agg_params, agg_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the received local parameters and store the test aggregated.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: Exceptions that occurred while the server
                was waiting for client updates.

        Returns
        -------
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round=server_round, results=results, failures=failures
        )
        _ = aggregated_metrics  # Avoid unused variable warning

        # Weigh accuracy of each client by number of examples used
        accuracies: List[float] = []
        for _, res in results:
            accuracy: float = float(res.metrics["accuracy"])
            accuracies.append(accuracy)
        print(f"Round {server_round} accuracies: {accuracies}")

        # Aggregate and print custom metric
        averaged_accuracy = sum(accuracies) / len(accuracies)
        print(f"Round {server_round} accuracy averaged: {averaged_accuracy}")
        return aggregated_loss, {"accuracy": averaged_accuracy}


class AggregateBodyStrategy(ServerInitializationStrategy):
    """Body Aggregation strategy implementation."""

    def __init__(self, *args, save_path: Path = Path(""), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = save_path if save_path != "" else None
        if save_path is not None:
            self.save_path = save_path / "models"
            self.save_path.mkdir(parents=True, exist_ok=True)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all
                currently connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        # Same as superclass method but adds the head

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        weights = parameters_to_ndarrays(parameters)

        # Add head parameters to received body parameters
        weights.extend(
            [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]
        )

        parameters = ndarrays_to_parameters(weights)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.min_available_clients, min_num_clients=self.min_fit_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        # Same as superclass method but adds the head

        # Parameters and config
        config: Dict[Any, Any] = {}

        weights = parameters_to_ndarrays(parameters)

        # Add head parameters to received body parameters
        weights.extend(
            [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]
        )

        parameters = ndarrays_to_parameters(weights)

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if server_round >= 0:
            # Sample clients
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
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:
        """Aggregate received local parameters, set global model parameters and save.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients. Note
                that not all previously selected clients are necessarily included in
                this list: a client might drop out and not submit a result. For each
                client that did not submit an update, there should be an `Exception`
                in `failures`.
            failures: Exceptions that occurred while the server was waiting for client
                updates.

        Returns
        -------
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        agg_params, agg_metrics = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if agg_params is not None:
            parameters = parameters_to_ndarrays(agg_params)
            model_keys = list(self.model.body.state_dict().keys())
            params_dict = zip(model_keys, parameters)
            state_dict = OrderedDict(
                {key: torch.tensor(param) for key, param in params_dict}
            )
            self.model.set_parameters(state_dict)

        if self.save_path is not None:
            # Save Model
            torch.save(self.model, self.save_path / f"model-ep_{server_round}.pt")

        return agg_params, agg_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the received local parameters and store the test aggregated.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: Exceptions that occurred while the server
                was waiting for client updates.

        Returns
        -------
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """
        aggregated_loss, _ = super().aggregate_evaluate(
            server_round=server_round, results=results, failures=failures
        )

        # Weigh accuracy of each client by number of examples used
        accuracies: List[float] = []
        for _, res in results:
            accuracy: float = float(res.metrics["accuracy"])
            accuracies.append(accuracy)
        print(f"Round {server_round} accuracies: {accuracies}")

        # Aggregate and print custom metric
        averaged_accuracy = sum(accuracies) / len(accuracies)
        print(f"Round {server_round} accuracy averaged: {averaged_accuracy}")
        return aggregated_loss, {"accuracy": averaged_accuracy}
