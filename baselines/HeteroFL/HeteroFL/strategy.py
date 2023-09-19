"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from models import (
    get_parameters,
    get_state_dict_from_param,
    param_idx_to_local_params,
    param_model_rate_mapping,
)
import copy
from utils import make_optimizer, make_scheduler


class HeteroFL(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        net=None,
        optim_scheduler_settings = None,
        evaluate_fn = None
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        # # created client_to_model_mapping
        # self.client_to_model_rate_mapping: Dict[str, ClientProxy] = {}

        self.net = net
        self.optim_scheduler_settings = optim_scheduler_settings
        self.local_param_model_rate = None
        self.active_cl_mr = None
        self.active_cl_labels = None
        # required for scheduling the lr
        self.optimizer = None,
        self.scheduler = None

    def __repr__(self) -> str:
        return "HeteroFL"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # self.make_client_to_model_rate_mapping(client_manager)
        # net = conv(model_rate = 1)
        ndarrays = get_parameters(self.net)
        # print(self.net.state_dict())
        self.local_param_model_rate = param_model_rate_mapping(self.net.state_dict() , client_manager.get_all_clients_to_model_mapping())

        self.active_cl_labels = client_manager.client_label_split.copy()
        self.optimizer = make_optimizer( self.optim_scheduler_settings["optimizer"], self.net.parameters() , lr=self.optim_scheduler_settings["lr"], momentum=self.optim_scheduler_settings["momentum"], weight_decay=self.optim_scheduler_settings["weight_decay"])
        self.scheduler = make_scheduler(self.optim_scheduler_settings["scheduler"], self.optimizer , milestones=self.optim_scheduler_settings["milestones"])
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        print("in configure fit , server round no. = {}".format(server_round))
        # Sample clients
        # no need to change this
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # for sampling we pass the criterion to select the required clients
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        # update client model rate mapping
        client_manager.update(server_round)

        global_parameters = get_state_dict_from_param(self.net, parameters)

        self.active_cl_mr = OrderedDict()

        # Create custom configs
        fit_configurations = []
        lr = self.optimizer.param_groups[0]["lr"]
        print(f'lr = {lr}')
        for idx, client in enumerate(clients):
            model_rate = client_manager.get_client_to_model_mapping(client.cid)
            client_param_idx = self.local_param_model_rate[model_rate]
            local_param = param_idx_to_local_params(global_parameters, client_param_idx)
            self.active_cl_mr[client.cid] = model_rate
            # local param are in the form of state_dict, so converting them only to values of tensors
            local_param_fitres = [v.cpu() for v in local_param.values()]
            fit_configurations.append(
                (client, FitIns(ndarrays_to_parameters(local_param_fitres), {"lr": lr}))
            )

        self.scheduler.step()
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        print('in aggregate fit')
        gl_model = self.net.state_dict()

        param_idx = []
        for i in range(len(results)):
            param_idx.append(
                copy.deepcopy(self.local_param_model_rate[self.active_cl_mr[results[i][0].cid]])
            )

        local_parameters = [fit_res.parameters for _, fit_res in results]
        for i in range(len(results)):
            local_parameters[i] = parameters_to_ndarrays(local_parameters[i])
            j = 0
            temp_od = OrderedDict()
            for k, _ in gl_model.items():
                temp_od[k] = local_parameters[i][j]
                j += 1
            local_parameters[i] = temp_od

        count = OrderedDict()
        output_weight_name = [k for k in gl_model.keys() if "weight" in k][-1]
        output_bias_name = [k for k in gl_model.keys() if "bias" in k][-1]
        for k, v in gl_model.items():
            parameter_type = k.split(".")[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if "weight" in parameter_type or "bias" in parameter_type:
                    if parameter_type == "weight":
                        if v.dim() > 1:
                            if k == output_weight_name:
                                label_split = self.active_cl_labels[
                                    int(results[m][0].cid)
                                ]
                                label_split = label_split.type(torch.int)
                                param_idx[m][k] = list(param_idx[m][k])
                                # print(f'Oohalu gusugusalaade {label_split}')
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[
                                    torch.meshgrid(param_idx[m][k])
                                ] += local_parameters[m][k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[
                                    torch.meshgrid(param_idx[m][k])
                                ] += local_parameters[m][k]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                    else:
                        if k == output_bias_name:
                            label_split = self.active_cl_labels[int(results[m][0].cid)]
                            label_split = label_split.type(torch.int)
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += local_parameters[m][k][
                                label_split
                            ]
                            count[k][param_idx[m][k]] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                else:
                    tmp_v += local_parameters[m][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)

        return ndarrays_to_parameters([v for k, v in gl_model.items()]), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        global_parameters = get_state_dict_from_param(self.net, parameters)

        self.active_cl_mr = OrderedDict()

        # Create custom configs
        evaluate_configurations = []
        for idx, client in enumerate(clients):
            model_rate = client_manager.get_client_to_model_mapping(client.cid)
            client_param_idx = self.local_param_model_rate[model_rate]
            local_param = param_idx_to_local_params(global_parameters, client_param_idx)
            self.active_cl_mr[client.cid] = model_rate
            # local param are in the form of state_dict, so converting them only to values of tensors
            local_param_fitres = [v.cpu() for v in local_param.values()]
            evaluate_configurations.append(
                (client, EvaluateIns(ndarrays_to_parameters(local_param_fitres), {}))
            )
        return evaluate_configurations

        # return self.configure_fit(server_round , parameters , client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        accuracy_aggregated = 0
        for cp, y in results:
            print(f"{cp.cid}-->{y.metrics['accuracy']}", end=" ")
            accuracy_aggregated += y.metrics["accuracy"]
        accuracy_aggregated /= len(results)

        metrics_aggregated = {"accuracy": accuracy_aggregated}
        print(f"\npaneer lababdar {metrics_aggregated}")
        return loss_aggregated, metrics_aggregated

    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
