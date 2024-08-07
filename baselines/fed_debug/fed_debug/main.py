"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import gc
import logging
import random
import time

import flwr as fl
import hydra
import ray
import torch
from diskcache import Index
from flwr.common import ndarrays_to_parameters
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from fed_debug.client import CNNFlowerClient, get_parameters, set_parameters
from fed_debug.dataset import ClientsAndServerDatasetsPrep
from fed_debug.differential_testing import (
    eval_na_threshold,
    run_fed_debug_differential_testing,
)
from fed_debug.models import global_model_eval, initialize_model
from fed_debug.strategy import FedAvgSave
from fed_debug.utils import gen_thresholds_exp_graph, generate_table2_csv

seed_everything(786)


class FLSimulation:
    """Main class to run the simulation."""

    def __init__(self, cfg, cache):
        self.all_rounds_results = []
        self.cache = cache
        self.cfg = cfg
        self.strategy = None
        self.device = torch.device(self.cfg.device)

        self.client_resources = {"num_cpus": cfg.client_cpus}
        if self.cfg.device == "cuda":
            self.client_resources = {
                "num_gpus": cfg.client_gpu,
                "num_cpus": cfg.client_cpus,
            }

        init_args = {"num_cpus": self.cfg.total_cpus, "num_gpus": self.cfg.total_gpus}
        self.backend_config = {
            "client_resources": self.client_resources,
            "init_args": init_args,
        }
        self._set_strategy()

    def set_server_data(self, ds):
        """Set the server data."""
        self.server_testdata = ds

    def set_clients_data(self, c2d):
        """Set the clients data."""
        self.client2data = c2d

    def _set_strategy(self):
        initial_net = initialize_model(self.cfg.model.name, self.cfg.dataset)["model"]
        if self.cfg.strategy.name in ["fedavg"]:
            strategy = FedAvgSave(
                cfg=self.cfg,
                cache=self.cache,
                accept_failures=False,
                fraction_fit=0,
                fraction_evaluate=0.0,
                min_fit_clients=self.cfg.strategy.clients_per_round,
                min_evaluate_clients=0,
                min_available_clients=self.cfg.data_dist.num_clients,
                initial_parameters=ndarrays_to_parameters(
                    ndarrays=get_parameters(initial_net)
                ),
                evaluate_fn=self._evaluate_global_model,  # ignore
                on_fit_config_fn=self._get_fit_config,  # Pass the fit_config function
                fit_metrics_aggregation_fn=self._fit_metrics_aggregation_fn,
            )
            self.strategy = strategy

    def _fit_metrics_aggregation_fn(self, metrics):
        """Aggregate metrics recieved from client."""
        logging.info(">>   ------------------- Clients Metrics ------------- ")
        all_logs = {}

        # if len(metrics) != self.cfg.strategy.clients_per_round:

        for t in metrics:
            nk, m = t
            cid = int(m["cid"])
            s = f' Client {m["cid"]}, Loss Train {m["train_loss"]},\
                    / Accuracy Train {m["train_accuracy"]}, data_points = {nk}'
            all_logs[cid] = s

        # sorted by client id from lowest to highest
        for k in sorted(all_logs.keys()):
            logging.info(all_logs[k])

        return {"loss": 0.1, "accuracy": 0.2}

    def _get_fit_config(self, server_round: int):
        random.seed(server_round)
        torch.manual_seed(server_round)
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": self.cfg.client.epochs,  #
            "batch_size": self.cfg.data_dist.batch_size,
            "lr": self.cfg.client.lr,
        }
        gc.collect()
        return config

    def _evaluate_global_model(self, server_round, parameters, config):
        gm_dict = initialize_model(self.cfg.model.name, self.cfg.dataset)
        set_parameters(gm_dict["model"], parameters)
        gm_dict["model"].eval()  # type: ignore
        d = global_model_eval(self.cfg.model.arch, gm_dict, self.server_testdata)
        loss = d["loss"]
        accuracy = d["accuracy"]
        self.all_rounds_results.append({"loss": loss, "accuracy": accuracy})

        del gm_dict["model"]
        torch.cuda.empty_cache()
        gc.collect()
        return loss, {"accuracy": accuracy, "loss": loss, "round": server_round}

    def _get_client(self, cid):
        model_dict = initialize_model(self.cfg.model.name, self.cfg.dataset)
        client = None
        args = {
            "cid": cid,
            "model_dict": model_dict,
            "client_data_train": self.client2data[cid],
            "valloader": None,
            "device": self.device,
            "mode": self.cfg.strategy.name,
        }

        client = CNNFlowerClient(args).to_client()
        return client

    def _run2(self):
        """Run the simulation."""
        client_app = fl.client.ClientApp(client_fn=self._get_client)

        server_config = fl.server.ServerConfig(num_rounds=self.cfg.strategy.num_rounds)
        server_app = fl.server.ServerApp(config=server_config, strategy=self.strategy)

        fl.simulation.run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=self.cfg.num_clients,
            backend_config=self.backend_config,  # type: ignore
        )
        return self.all_rounds_results

    def run(self):
        """Run the simulation."""
        fl.simulation.start_simulation(
            ray_init_args=self.backend_config["init_args"],
            client_fn=self._get_client,
            num_clients=self.cfg.num_clients,
            config=fl.server.ServerConfig(
                num_rounds=self.cfg.strategy.num_rounds
            ),  # Just three rounds
            strategy=self.strategy,
            client_resources=self.backend_config["client_resources"],
        )

        return self.all_rounds_results


def run_simulation(cfg):
    """Run the simulation."""
    # print(cfg.multirun_faulty_clients)

    if cfg.multirun_faulty_clients != -1:
        cfg.faulty_clients_ids = list(range(cfg.multirun_faulty_clients))

    cfg.faulty_clients_ids = [f"{c}" for c in cfg.faulty_clients_ids]

    def set_exp_key(cfg):
        key = (
            f"{cfg.model.name}-{cfg.dataset.name}-"
            f"faulty_clients[{cfg.faulty_clients_ids}]-"
            f"noise_rate{cfg.noise_rate}-"
            f"TClients{cfg.data_dist.num_clients}-"
            f"{cfg.strategy.name}-(R{cfg.strategy.num_rounds}"
            f"-clientsPerR{cfg.strategy.clients_per_round})"
            f"-{cfg.data_dist.dist_type}{cfg.data_dist.dirichlet_alpha}"
            f"-batch{cfg.data_dist.batch_size}-epochs{cfg.client.epochs}-"
            f"lr{cfg.client.lr}"
        )
        return key

    train_cache = Index(cfg.storage.dir + cfg.storage.train_cache_name)

    exp_key = set_exp_key(cfg)
    cfg.exp_key = exp_key

    logging.info(f" ***********  Starting Experiment: {cfg.exp_key} ***************")


    if cfg.check_cache:
        if cfg.exp_key in train_cache:
            temp_dict = train_cache[cfg.exp_key]
            # type: ignore
            if "complete" in temp_dict and temp_dict["complete"]:  # type: ignore
                logging.info(f"Experiment already completed: {cfg.exp_key}")
                return

    logging.info(f"Simulation Configuration: {cfg}")

    ds_prep = ClientsAndServerDatasetsPrep(cfg)
    ds_dict = ds_prep.get_clients_server_data()
    client2class = ds_dict["client2class"]

    sim = FLSimulation(cfg, train_cache)
    sim.set_server_data(ds_dict["server_testdata"])
    sim.set_clients_data(ds_dict["client2data"])
    round2results = sim.run()

    temp_input = None

    if "pixel_values" in sim.server_testdata:
        temp_input = sim.server_testdata[0]["pixel_values"].clone().detach()
    else:
        temp_input = sim.server_testdata[0][0].clone().detach()

    train_cache[cfg.exp_key] = {
        "client2class": client2class,
        "train_cfg": cfg,
        "complete": True,
        "input_shape": temp_input.shape,
        "all_ronuds_gm_results": round2results,
    }

    logging.info(f"Results of gm evaluations each round: {round2results}")
    logging.info(f"Simulation Complete for: {cfg.exp_key} ")

    ray.shutdown()
    if ray.is_initialized():
        ray.cluster().shutdown()

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline."""
    if not cfg.generate_graphs and not cfg.vary_thresholds:
        run_simulation(cfg)
        time.sleep(1)
        run_fed_debug_differential_testing(cfg)

    if cfg.vary_thresholds:
        eval_na_threshold(cfg)

    if cfg.generate_graphs:
        gen_thresholds_exp_graph(
            cache_path=cfg.storage.dir + cfg.storage.results_cache_name,
            threshold_exp_key=cfg.threshold_variation_exp_key,
        )
        generate_table2_csv(
            cfg.storage.dir + cfg.storage.results_cache_name,
            igonre_keys=[cfg.threshold_variation_exp_key],
        )


if __name__ == "__main__":
    main()
