"""Flower strategy for HeteroFL."""

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
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
from torch import nn

from heterofl.client_manager_heterofl import ClientManagerHeteroFL
from heterofl.models import (
    get_parameters,
    get_state_dict_from_param,
    param_idx_to_local_params,
    param_model_rate_mapping,
)
from heterofl.utils import make_optimizer, make_scheduler


# pylint: disable=too-many-instance-attributes
class HeteroFL(fl.server.strategy.Strategy):
    """HeteroFL strategy.

    Distribute subsets of a global model to clients according to their

    computational complexity and aggregate received models from clients.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model_name: str,
        net: nn.Module,
        optim_scheduler_settings: Dict,
        global_model_rate: float = 1.0,
        evaluate_fn=None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
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

        self.model_name = model_name
        self.net = net
        self.global_model_rate = global_model_rate
        # info required for configure and aggregate
        # to be filled in initialize
        self.local_param_model_rate: OrderedDict = OrderedDict()
        # to be filled in initialize
        self.active_cl_labels: List[torch.tensor] = []
        # to be filled in configure
        self.active_cl_mr: OrderedDict = OrderedDict()
        # required for scheduling the lr
        self.optimizer = make_optimizer(
            optim_scheduler_settings["optimizer"],
            self.net.parameters(),
            learning_rate=optim_scheduler_settings["lr"],
            momentum=optim_scheduler_settings["momentum"],
            weight_decay=optim_scheduler_settings["weight_decay"],
        )
        self.scheduler = make_scheduler(
            optim_scheduler_settings["scheduler"],
            self.optimizer,
            milestones=optim_scheduler_settings["milestones"],
        )

    def __repr__(self) -> str:
        """Return a string representation of the HeteroFL object."""
        return "HeteroFL"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # self.make_client_to_model_rate_mapping(client_manager)
        # net = conv(model_rate = 1)
        if not isinstance(client_manager, ClientManagerHeteroFL):
            raise ValueError(
                "Not valid client manager, use ClientManagerHeterFL instead"
            )
        clnt_mngr_heterofl: ClientManagerHeteroFL = client_manager

        ndarrays = get_parameters(self.net)
        self.local_param_model_rate = param_model_rate_mapping(
            self.model_name,
            self.net.state_dict(),
            clnt_mngr_heterofl.get_all_clients_to_model_mapping(),
            self.global_model_rate,
        )

        if clnt_mngr_heterofl.client_label_split is not None:
            self.active_cl_labels = clnt_mngr_heterofl.client_label_split.copy()

        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        print(f"in configure fit , server round no. = {server_round}")
        if not isinstance(client_manager, ClientManagerHeteroFL):
            raise ValueError(
                "Not valid client manager, use ClientManagerHeterFL instead"
            )
        clnt_mngr_heterofl: ClientManagerHeteroFL = client_manager
        # Sample clients
        # no need to change this
        clientts_selection_config = {}
        (
            clientts_selection_config["sample_size"],
            clientts_selection_config["min_num_clients"],
        ) = self.num_fit_clients(clnt_mngr_heterofl.num_available())

        # for sampling we pass the criterion to select the required clients
        clients = clnt_mngr_heterofl.sample(
            num_clients=clientts_selection_config["sample_size"],
            min_num_clients=clientts_selection_config["min_num_clients"],
        )

        # update client model rate mapping
        clnt_mngr_heterofl.update(server_round)

        global_parameters = get_state_dict_from_param(self.net, parameters)

        self.active_cl_mr = OrderedDict()

        # Create custom configs
        fit_configurations = []
        learning_rate = self.optimizer.param_groups[0]["lr"]
        print(f"lr = {learning_rate}")
        for client in clients:
            model_rate = clnt_mngr_heterofl.get_client_to_model_mapping(client.cid)
            client_param_idx = self.local_param_model_rate[model_rate]
            local_param = param_idx_to_local_params(
                global_parameters=global_parameters, client_param_idx=client_param_idx
            )
            self.active_cl_mr[client.cid] = model_rate
            # local param are in the form of state_dict,
            #  so converting them only to values of tensors
            local_param_fitres = [val.cpu() for val in local_param.values()]
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        ndarrays_to_parameters(local_param_fitres),
                        {"lr": learning_rate},
                    ),
                )
            )

        self.scheduler.step()
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average.

        Adopted from authors implementation.
        """
        print("in aggregate fit")
        gl_model = self.net.state_dict()

        param_idx = []
        for res in results:
            param_idx.append(
                copy.deepcopy(
                    self.local_param_model_rate[self.active_cl_mr[res[0].cid]]
                )
            )

        local_param_as_parameters = [fit_res.parameters for _, fit_res in results]
        local_parameters_as_ndarrays = [
            parameters_to_ndarrays(local_param_as_parameters[i])
            for i in range(len(local_param_as_parameters))
        ]
        local_parameters: List[OrderedDict] = [
            OrderedDict() for _ in range(len(local_param_as_parameters))
        ]
        for i in range(len(results)):
            j = 0
            for k, _ in gl_model.items():
                local_parameters[i][k] = local_parameters_as_ndarrays[i][j]
                j += 1

        if "conv" in self.model_name:
            self._aggregate_conv(param_idx, local_parameters, results)

        elif "resnet" in self.model_name:
            self._aggregate_resnet18(param_idx, local_parameters, results)
        else:
            raise ValueError("Not valid model name")

        return ndarrays_to_parameters([v for k, v in gl_model.items()]), {}

    def _aggregate_conv(self, param_idx, local_parameters, results):
        gl_model = self.net.state_dict()
        count = OrderedDict()
        output_bias_name = [k for k in gl_model.keys() if "bias" in k][-1]
        output_weight_name = [k for k in gl_model.keys() if "weight" in k][-1]
        for k, val in gl_model.items():
            parameter_type = k.split(".")[-1]
            count[k] = val.new_zeros(val.size(), dtype=torch.float32)
            tmp_v = val.new_zeros(val.size(), dtype=torch.float32)
            for clnt, _ in enumerate(local_parameters):
                if "weight" in parameter_type or "bias" in parameter_type:
                    self._agg_layer_conv(
                        {
                            "cid": int(results[clnt][0].cid),
                            "param_idx": param_idx,
                            "local_parameters": local_parameters,
                        },
                        {
                            "tmp_v": tmp_v,
                            "count": count,
                        },
                        {
                            "clnt": clnt,
                            "parameter_type": parameter_type,
                            "k": k,
                            "val": val,
                        },
                        {
                            "output_weight_name": output_weight_name,
                            "output_bias_name": output_bias_name,
                        },
                    )
                else:
                    tmp_v += local_parameters[clnt][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            val[count[k] > 0] = tmp_v[count[k] > 0].to(val.dtype)

    def _agg_layer_conv(
        self,
        clnt_params,
        tmp_v_count,
        param_info,
        output_names,
    ):
        # pi = param_info
        param_idx = clnt_params["param_idx"]
        clnt = param_info["clnt"]
        k = param_info["k"]
        tmp_v = tmp_v_count["tmp_v"]
        count = tmp_v_count["count"]

        if param_info["parameter_type"] == "weight":
            if param_info["val"].dim() > 1:
                if k == output_names["output_weight_name"]:
                    label_split = self.active_cl_labels[clnt_params["cid"]]
                    label_split = label_split.type(torch.int)
                    param_idx[clnt][k] = list(param_idx[clnt][k])
                    param_idx[clnt][k][0] = param_idx[clnt][k][0][label_split]
                    tmp_v[torch.meshgrid(param_idx[clnt][k])] += clnt_params[
                        "local_parameters"
                    ][clnt][k][label_split]
                    count[k][torch.meshgrid(param_idx[clnt][k])] += 1
                else:
                    tmp_v[torch.meshgrid(param_idx[clnt][k])] += clnt_params[
                        "local_parameters"
                    ][clnt][k]
                    count[k][torch.meshgrid(param_idx[clnt][k])] += 1
            else:
                tmp_v[param_idx[clnt][k]] += clnt_params["local_parameters"][clnt][k]
                count[k][param_idx[clnt][k]] += 1
        else:
            if k == output_names["output_bias_name"]:
                label_split = self.active_cl_labels[clnt_params["cid"]]
                label_split = label_split.type(torch.int)
                param_idx[clnt][k] = param_idx[clnt][k][label_split]
                tmp_v[param_idx[clnt][k]] += clnt_params["local_parameters"][clnt][k][
                    label_split
                ]
                count[k][param_idx[clnt][k]] += 1
            else:
                tmp_v[param_idx[clnt][k]] += clnt_params["local_parameters"][clnt][k]
                count[k][param_idx[clnt][k]] += 1

    def _aggregate_resnet18(self, param_idx, local_parameters, results):
        gl_model = self.net.state_dict()
        count = OrderedDict()
        for k, val in gl_model.items():
            parameter_type = k.split(".")[-1]
            count[k] = val.new_zeros(val.size(), dtype=torch.float32)
            tmp_v = val.new_zeros(val.size(), dtype=torch.float32)
            for clnt, _ in enumerate(local_parameters):
                if "weight" in parameter_type or "bias" in parameter_type:
                    self._agg_layer_resnet18(
                        {
                            "cid": int(results[clnt][0].cid),
                            "param_idx": param_idx,
                            "local_parameters": local_parameters,
                        },
                        tmp_v,
                        count,
                        {
                            "clnt": clnt,
                            "parameter_type": parameter_type,
                            "k": k,
                            "val": val,
                        },
                    )
                else:
                    tmp_v += local_parameters[clnt][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            val[count[k] > 0] = tmp_v[count[k] > 0].to(val.dtype)

    def _agg_layer_resnet18(self, clnt_params, tmp_v, count, param_info):
        param_idx = clnt_params["param_idx"]
        k = param_info["k"]
        clnt = param_info["clnt"]

        if param_info["parameter_type"] == "weight":
            if param_info["val"].dim() > 1:
                if "linear" in k:
                    label_split = self.active_cl_labels[clnt_params["cid"]]
                    label_split = label_split.type(torch.int)
                    param_idx[clnt][k] = list(param_idx[clnt][k])
                    param_idx[clnt][k][0] = param_idx[clnt][k][0][label_split]
                    tmp_v[torch.meshgrid(param_idx[clnt][k])] += clnt_params[
                        "local_parameters"
                    ][clnt][k][label_split]
                    count[k][torch.meshgrid(param_idx[clnt][k])] += 1
                else:
                    tmp_v[torch.meshgrid(param_idx[clnt][k])] += clnt_params[
                        "local_parameters"
                    ][clnt][k]
                    count[k][torch.meshgrid(param_idx[clnt][k])] += 1
            else:
                tmp_v[param_idx[clnt][k]] += clnt_params["local_parameters"][clnt][k]
                count[k][param_idx[clnt][k]] += 1
        else:
            if "linear" in k:
                label_split = self.active_cl_labels[clnt_params["cid"]]
                label_split = label_split.type(torch.int)
                param_idx[clnt][k] = param_idx[clnt][k][label_split]
                tmp_v[param_idx[clnt][k]] += clnt_params["local_parameters"][clnt][k][
                    label_split
                ]
                count[k][param_idx[clnt][k]] += 1
            else:
                tmp_v[param_idx[clnt][k]] += clnt_params["local_parameters"][clnt][k]
                count[k][param_idx[clnt][k]] += 1

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # if self.fraction_evaluate == 0.0:
        #     return []
        # config = {}
        # evaluate_ins = EvaluateIns(parameters, config)

        # # Sample clients
        # sample_size, min_num_clients = self.num_evaluation_clients(
        #     client_manager.num_available()
        # )
        # clients = client_manager.sample(
        #     num_clients=sample_size, min_num_clients=min_num_clients
        # )

        # global_parameters = get_state_dict_from_param(self.net, parameters)

        # self.active_cl_mr = OrderedDict()

        # # Create custom configs
        # evaluate_configurations = []
        # for idx, client in enumerate(clients):
        #     model_rate = client_manager.get_client_to_model_mapping(client.cid)
        #     client_param_idx = self.local_param_model_rate[model_rate]
        #     local_param =
        #     param_idx_to_local_params(global_parameters, client_param_idx)
        #     self.active_cl_mr[client.cid] = model_rate
        #     # local param are in the form of state_dict,
        #     # so converting them only to values of tensors
        #     local_param_fitres = [v.cpu() for v in local_param.values()]
        #     evaluate_configurations.append(
        #         (client, EvaluateIns(ndarrays_to_parameters(local_param_fitres), {}))
        #     )
        # return evaluate_configurations

        return []

        # return self.configure_fit(server_round , parameters , client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # if not results:
        #     return None, {}

        # loss_aggregated = weighted_loss_avg(
        #     [
        #         (evaluate_res.num_examples, evaluate_res.loss)
        #         for _, evaluate_res in results
        #     ]
        # )

        # accuracy_aggregated = 0
        # for cp, y in results:
        #     print(f"{cp.cid}-->{y.metrics['accuracy']}", end=" ")
        #     accuracy_aggregated += y.metrics["accuracy"]
        # accuracy_aggregated /= len(results)

        # metrics_aggregated = {"accuracy": accuracy_aggregated}
        # print(f"\npaneer lababdar {metrics_aggregated}")
        # return loss_aggregated, metrics_aggregated

        return None, {}

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
