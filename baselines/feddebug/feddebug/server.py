"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""


import random
from functools import partial
from logging import INFO

import torch
from flwr.common import ndarrays_to_parameters
from flwr.common.logger import log

from feddebug.models import test, initialize_model
from feddebug.strategy import FedAvgWithFedDebug
from feddebug.utils import get_parameters, set_parameters, calculate_localization_accuracy


def _fit_metrics_aggregation_fn(metrics):
    """Aggregate metrics recieved from client."""

    log(INFO, ">>   ------------------- Clients Metrics ------------- ")
    all_logs = {}
    for nk_points, metric_d in metrics:
        cid = int(metric_d["cid"])
        temp_s = (
            f' Client {metric_d["cid"]}, Loss Train {metric_d["train_loss"]}, '
            f'Accuracy Train {metric_d["train_accuracy"]}, data_points = {nk_points}'
        )
        all_logs[cid] = temp_s

    # sorted by client id from lowest to highest
    for k in sorted(all_logs.keys()):
        log(INFO, all_logs[k])
    return {"loss": 0.0, "accuracy": 0.0}


def _get_evaluate_gm_func(cfg, server_testdata):
    def _eval_gm(server_round, parameters, config):
        gm_dict = initialize_model(cfg.model.name, cfg.dataset)
        set_parameters(gm_dict["model"], parameters)
        gm_dict["model"].eval()  # type: ignore
        d_res = test(gm_dict['model'], server_testdata, device=cfg.device)
        return d_res["loss"], {"accuracy": d_res["accuracy"], "loss": d_res["loss"], "round": server_round}

    return _eval_gm


def _get_fit_config(cfg, server_round):
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": cfg.client.epochs,  #
        "batch_size": cfg.client.batch_size,
        "lr": cfg.client.lr,
    }
    random.seed(server_round)
    torch.manual_seed(server_round)
    return config


def get_strategy(cfg, server_testdata):
    """Return the strategy based on the configuration."""
    def _create_model():
        return initialize_model(cfg.model.name, cfg.dataset)["model"]

    def _callback_fed_debug_evaluate_fn(server_round, predicted_malicious_clients):
        """Callback function to evaluate the global model."""
        true_malicious_clients = cfg.malicious_clients_ids

        log(INFO, f"***FedDebug Output Round {server_round} ***")
        log(INFO, f"Total Random Inputs = {cfg.feddebug.r_inputs}")
        log(INFO, f"True Malicious Clients (Ground Truth) = {true_malicious_clients}")
        log(INFO, f"Predicted Malicious Clients = {predicted_malicious_clients}")
        localization_accuracy = calculate_localization_accuracy(true_malicious_clients, predicted_malicious_clients)
        log(INFO, f"FedDebug Localization Accuracy = {localization_accuracy}")

    num_bugs = len(cfg.malicious_clients_ids)
    eval_gm_func = _get_evaluate_gm_func(cfg, server_testdata)
    initial_net = _create_model()

    strategy = FedAvgWithFedDebug(num_bugs=num_bugs, num_inputs=cfg.feddebug.r_inputs, input_shape=server_testdata.dataset[0]["image"].clone().detach().shape, na_t=cfg.feddebug.na_t, device=cfg.device, fast=cfg.feddebug.fast, callback_create_model_fn=_create_model, callback_fed_debug_evaluate_fn=_callback_fed_debug_evaluate_fn, accept_failures=False,
                                  fraction_fit=0, fraction_evaluate=0.0, min_fit_clients=cfg.clients_per_round, min_evaluate_clients=0, min_available_clients=cfg.num_clients, initial_parameters=ndarrays_to_parameters(ndarrays=get_parameters(initial_net)), evaluate_fn=eval_gm_func, on_fit_config_fn=partial(_get_fit_config, cfg), fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn)

    return strategy
