"""floco: A Flower Baseline."""
"""floco: A Flower Baseline."""
"""Optionally define a custom strategy.
Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

import json
import copy

import torch
import wandb
import numpy as np
from sklearn import decomposition

from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarray_to_bytes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,

)

from flwr.common import logger
from flwr.common.typing import UserConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace

from floco.model import SimplexModel, set_weights


PROJECT_NAME = "Floco_WandB"


class Floco(FedAvg):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        tau: int = 0,
        rho: float = 1.0,
        endpoints: int = 1,
        num_clients: int = 10,
        pers_epoch: int = 0,
        run_config: UserConfig, 
        use_wandb: bool
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.tau = tau
        self.rho = rho
        self.endpoints = endpoints
        self.pers_epoch = pers_epoch
        self.num_clients = num_clients
        self.client_gradients = {}

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()
        # Keep track of best acc
        self.best_acc_so_far = 0.0
        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.
        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            model = SimplexModel(endpoints=self.endpoints, seed=0)
            set_weights(model, ndarrays)
            # Save the PyTorch model
            file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )
        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(
            server_round, 
            parameters_ndarrays, 
            {
                "center": tuple([1 / self.endpoints for _ in range(self.endpoints)]), 
                "radius": self.rho
                }
            )
        if eval_res is None:
            return None
        loss, metrics = eval_res
        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training.
        """
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config["server_round"] = server_round
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self.selected_client_cids = [client.cid for client in clients]
        if (server_round + 1) == self.tau: # Round before projection
            clients = client_manager.sample( # Sample all clients to get most up to date gradients  
                num_clients=self.num_clients,
                min_num_clients=self.num_clients
                )
            self.all_client_cids = [client.cid for client in clients]
        elif server_round == self.tau: # Round of projection
            self.projected_clients = project_clients(   
                    self.client_gradients, self.endpoints
                )
            self.client_subregion_parameters = {
                client_id: subregion_parameters
                for client_id, subregion_parameters in zip(self.all_client_cids, self.projected_clients)
            }
        if server_round >= self.tau:
            fit_ins_all_clients = []
            for client in clients:
                tmp_client_config = copy.deepcopy(config)
                tmp_client_config["center"] = ndarray_to_bytes(self.client_subregion_parameters[client.cid])
                tmp_client_config["radius"] = ndarray_to_bytes(self.rho)
                tmp_fit_ins = FitIns(parameters, tmp_client_config)
                fit_ins_all_clients.append(
                    (client, tmp_fit_ins)
                    )
            return fit_ins_all_clients
        # Return client/config pairs
        return [(client, fit_ins) for i, client in enumerate(clients)]

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
        if self.tau == (server_round + 1): # All clients results are collected
            # Save client gradients for later projection
            for cl, fit_res in results:
                client_id = cl.cid
                w = parameters_to_ndarrays(fit_res.parameters)
                client_grads = [w[-i].flatten() for i in range(1,self.endpoints+1)] # Get gradients for simplex layer(s) only
                client_grads = np.concatenate(client_grads)
                self.client_gradients[client_id] = client_grads
                self.client_gradients = dict(sorted(self.client_gradients.items())) # Sort gradients by client id for better readability
            results = [(cl, fit_res) for cl, fit_res in results if cl.cid in self.selected_client_cids] # Only select the clients that were sampled
        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No fit_metrics_aggregation_fn provided")
        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics


def project_clients(client_gradients, endpoints):
    client_stats = np.array(list(client_gradients.values()))
    kappas = decomposition.PCA(n_components=endpoints).fit_transform(client_stats)
    # Find optimal projection
    lowest_log_energy = np.inf
    best_beta = None
    for i, z in enumerate(np.linspace(1e-4, 1, 1000)):
        betas = _project_client_onto_simplex(kappas, z=z)
        betas /= betas.sum(axis=1, keepdims=True)
        log_energy = _riesz_s_energy(betas)
        if log_energy not in [-np.inf, np.inf] and log_energy < lowest_log_energy:
            lowest_log_energy = log_energy
            best_beta = betas
    return best_beta


def _project_client_onto_simplex(kappas, z):
    sorted_kappas = np.sort(kappas, axis=1)[:, ::-1]
    z = np.ones(len(kappas)) * z
    cssv = np.cumsum(sorted_kappas, axis=1) - z[:, np.newaxis]
    ind = np.arange(kappas.shape[1]) + 1
    cond = sorted_kappas - cssv / ind > 0
    nonzero = np.count_nonzero(cond, axis=1)
    normalized_kappas = cssv[np.arange(len(kappas)), nonzero - 1] / nonzero
    betas = np.maximum(kappas - normalized_kappas[:, np.newaxis], 0)
    return betas


def _riesz_s_energy(simplex_points):
    diff = simplex_points[:, None] - simplex_points[None, :]
    distance = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    epsilon = 1e-4  # epsilon is the smallest distance possible to avoid overflow during gradient calculation
    distance[distance < epsilon] = epsilon
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = distance[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = epsilon
    energies = 1 / mutual_dist**2
    energy = energies[~np.isnan(energies)].sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy)
    return log_energy


def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir