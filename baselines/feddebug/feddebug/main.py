"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""


from logging import DEBUG, INFO
import torch
import hydra

import flwr as fl
from flwr.common.logger import log
from flwr.common import ndarrays_to_parameters

from feddebug import utils
from feddebug.client import CNNFlowerClient
from feddebug.dataset import ClientsAndServerDatasets
from feddebug.models import test, initialize_model
from feddebug.strategy import FedAvgWithFedDebug

utils.seed_everything(786)


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


def run_simulation(cfg):
    """Run the simulation."""
    if cfg.total_malicious_clients:
        cfg.malicious_clients_ids = list(range(cfg.total_malicious_clients))

    cfg.malicious_clients_ids = [f"{c}" for c in cfg.malicious_clients_ids]

    exp_key = utils.set_exp_key(cfg)

    log(INFO, f" ***********  Starting Experiment: {exp_key} ***************")

    log(DEBUG, f"Simulation Configuration: {cfg}")

    num_bugs = len(cfg.malicious_clients_ids)
    ds_prep = ClientsAndServerDatasets(cfg)
    ds_dict = ds_prep.get_data()
    server_testdata = ds_dict["server_testdata"]

    round2gm_accs = []
    round2feddebug_accs = []

    def _create_model():
        return initialize_model(cfg.model_name, cfg.dataset)

    def _get_fit_config(server_round):
        return {"server_round": server_round, "local_epochs": cfg.client.epochs, 
                "batch_size": cfg.client.batch_size, "lr": cfg.client.lr}

    def _get_client(cid):
        """Give the new client."""
        client2data = ds_dict["client2data"]

        args = {"cid": cid, "model": _create_model(),
                "client_data_train": client2data[cid], "device": torch.device(cfg.device)}
        client = CNNFlowerClient(args).to_client()
        return client

    def _eval_gm(server_round, parameters, config):
        gm_model = _create_model()
        utils.set_parameters(gm_model, parameters)
        d_res = test(gm_model, server_testdata, device=cfg.device)
        round2gm_accs.append(d_res["accuracy"])
        return d_res["loss"], {"accuracy": d_res["accuracy"], "loss": d_res["loss"], 
                               "round": server_round}

    def _callback_fed_debug_evaluate_fn(server_round, predicted_malicious_clients):
        """Callback function to evaluate the global model."""
        true_malicious_clients = cfg.malicious_clients_ids

        log(INFO, f"***FedDebug Output Round {server_round} ***")
        log(INFO, f"Total Random Inputs = {cfg.feddebug.r_inputs}")
        log(INFO, f"True Malicious Clients (Ground Truth) = {true_malicious_clients}")
        log(INFO, f"Predicted Malicious Clients = {predicted_malicious_clients}")
        localization_accuracy = utils.calculate_localization_accuracy(
            true_malicious_clients, predicted_malicious_clients)
        log(INFO, f"FedDebug Localization Accuracy = {localization_accuracy}")
        round2feddebug_accs.append(localization_accuracy)

    initial_net = _create_model()
    strategy = FedAvgWithFedDebug(num_bugs=num_bugs, num_inputs=cfg.feddebug.r_inputs,
            input_shape=server_testdata.dataset[0]["image"].clone().detach().shape, na_t=cfg.feddebug.na_t, 
            device=cfg.device, fast=cfg.feddebug.fast, callback_create_model_fn=_create_model, 
            callback_fed_debug_evaluate_fn=_callback_fed_debug_evaluate_fn, accept_failures=False,
            fraction_fit=0, fraction_evaluate=0.0, 
            min_fit_clients=cfg.clients_per_round, 
            min_evaluate_clients=0, min_available_clients=cfg.num_clients, 
            initial_parameters=ndarrays_to_parameters(ndarrays=utils.get_parameters(initial_net)), 
            evaluate_fn=_eval_gm, on_fit_config_fn=_get_fit_config, 
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn)

    server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)

    client_app = fl.client.ClientApp(client_fn=_get_client)
    server_app = fl.server.ServerApp(config=server_config, strategy=strategy)

    fl.simulation.run_simulation(server_app=server_app, client_app=client_app,
                                 num_supernodes=cfg.num_clients, backend_config=utils.config_sim_resources(cfg))

    utils.plot_metrics(round2gm_accs, round2feddebug_accs, cfg)

    log(INFO, f"Training Complete for Experiment: {exp_key} ")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg) -> None:
    """Run the baseline."""
    run_simulation(cfg)


if __name__ == "__main__":
    main()
