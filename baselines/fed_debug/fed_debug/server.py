"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from flwr.common.logger import log
from flwr.common import ndarrays_to_parameters

from logging import INFO
import torch
import gc
import random
from functools import partial
from fed_debug.models import initialize_model, global_model_eval
from fed_debug.utils import set_parameters, get_parameters
from fed_debug.strategy import FedAvgSave


def _fit_metrics_aggregation_fn(metrics):
    """Aggregate metrics recieved from client."""
    log(INFO, ">>   ------------------- Clients Metrics ------------- ")
    all_logs = {}

    # if len(metrics) != cfg.strategy.clients_per_round:

    for t in metrics:
        nk, m = t
        cid = int(m["cid"])
        s = (f' Client {m["cid"]}, Loss Train {m["train_loss"]},'
                f'/ Accuracy Train {m["train_accuracy"]}, data_points = {nk}')
        all_logs[cid] = s

    # sorted by client id from lowest to highest
    for k in sorted(all_logs.keys()):
        log(INFO, all_logs[k])

    return {"loss": 0.1, "accuracy": 0.2}


def _get_evaluate_gm_func(cfg, server_testdata):
    def _eval_gm(server_round, parameters, config):
        gm_dict = initialize_model(cfg.model.name, cfg.dataset)
        set_parameters(gm_dict["model"], parameters)
        gm_dict["model"].eval()  # type: ignore
        d = global_model_eval(cfg.model.arch, gm_dict, server_testdata)
        loss = d["loss"]
        accuracy = d["accuracy"]
        del gm_dict["model"]
        torch.cuda.empty_cache()
        gc.collect()
        return loss, {"accuracy": accuracy, "loss": loss, "round": server_round}
    return _eval_gm
    

def _get_fit_config(cfg, server_round):
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": cfg.client.epochs,  #
        "batch_size": cfg.data_dist.batch_size,
        "lr": cfg.client.lr,
    }
    random.seed(server_round)
    torch.manual_seed(server_round)
    gc.collect()
    return config


def get_strategy(cfg, server_testdata, cache):
    eval_gm_func = _get_evaluate_gm_func(cfg, server_testdata)
    initial_net = initialize_model(cfg.model.name, cfg.dataset)["model"]
    if cfg.strategy.name in ["fedavg"]:
        strategy = FedAvgSave(
            cfg=cfg,
            cache=cache,
            accept_failures=False,
            fraction_fit=0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.strategy.clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=cfg.data_dist.num_clients,
            initial_parameters=ndarrays_to_parameters(
                ndarrays=get_parameters(initial_net)
            ),
            evaluate_fn=eval_gm_func,  # ignore
            on_fit_config_fn=partial(_get_fit_config, cfg),  # Pass the fit_config function
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn,
        )
        return strategy
    else:
        raise ValueError(f"Strategy {cfg.strategy.name} not supported")

    
