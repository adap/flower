"""fls.py."""

import gc
import json
import logging

import torch

import flwr as fl
from flwr.common import ndarrays_to_parameters
from tracefl.fl_provenance import round_lambda_prov
from tracefl.models import (
    get_parameters,
    global_model_eval,
    initialize_model,
    set_parameters,
)
from tracefl.strategy import FedAvgSave
from tracefl.utils import get_backend_config


class FLSimulation:
    """Main class to run the simulation."""

    def __init__(self, cfg, ff, nsr, le, min_ev):
        self.all_rounds_results = []
        self.cfg = cfg
        self.strategy = None
        self.device = torch.device(self.cfg.tool.tracefl.device.device)
        self.backend_config = get_backend_config(cfg)
        self.fraction_fit = ff
        self.num_server_rounds = nsr
        self.local_epochs = le
        self.min_eval = min_ev

    def set_server_data(self, ds):
        self.server_testdata = ds

    def set_clients_data(self, c2d):
        self.client2data = c2d
        logging.info(f"Number of clients: {len(self.client2data)}")
        logging.info(f"Participating clients ids: {list(self.client2data.keys())}")
        if len(self.client2data) != self.cfg.tool.tracefl.data_dist.num_clients:
            self.cfg.tool.tracefl.num_clients = len(self.client2data)
            logging.warning(
                f"Adjusting number of clients to: {self.cfg.tool.tracefl.num_clients}"
            )

    def set_strategy(self):
        model_dict = initialize_model(
            self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
        )
        initial_parameters = ndarrays_to_parameters(get_parameters(model_dict["model"]))

        strategy = FedAvgSave(
            initial_parameters=initial_parameters,
            cfg=self.cfg,
            accept_failures=False,
            fraction_fit=self.fraction_fit,
            fraction_evaluate=0,
            min_fit_clients=self.cfg.tool.tracefl.strategy.clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=self.cfg.tool.tracefl.data_dist.num_clients,
            evaluate_fn=self._evaluate_global_model,
            evaluate_metrics_aggregation_fn=lambda metrics: {},
            on_fit_config_fn=self._get_fit_config,
            fit_metrics_aggregation_fn=self._fit_metrics_aggregation_fn,
        )

        self.strategy = strategy

    def _fit_metrics_aggregation_fn(self, metrics):
        logging.info(">> Client Metrics Summary:")
        for nk, m in metrics:
            cid = int(m["cid"])
            logging.info(
                f"Client {cid} | Loss: {m['train_loss']} | Accuracy: {m['train_accuracy']} | Samples: {nk}"
            )
        return {"loss": 0.1, "accuracy": 0.2}

    def _get_fit_config(self, server_round: int):
        
        torch.Generator().manual_seed(server_round)
        config = {
            "server_round": server_round,
            "local_epochs": self.cfg.tool.tracefl.client.epochs,
            "batch_size": self.cfg.tool.tracefl.data_dist.batch_size,
            "lr": self.cfg.tool.tracefl.client.lr
        }
        return config

    def _evaluate_global_model(self, server_round, parameters, config):
        logging.info("Evaluating global model...")
        try:
            model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            set_parameters(model_dict["model"], parameters)
            model_dict["model"].eval()

            metrics = global_model_eval(
                self.cfg.tool.tracefl.model.arch, model_dict, self.server_testdata
            )
            loss = metrics["loss"]
            acc = metrics["accuracy"]
            self.all_rounds_results.append({"loss": loss, "accuracy": acc})

            if server_round == 0:
                logging.info(
                    ">> Round 0 — skipping provenance analysis (no trained global model yet)"
                )
                return loss, {"accuracy": acc, "loss": loss, "round": server_round}

            logging.info(
                f">> Round {server_round}: Running TraceFL Provenance analysis..."
            )

            fedavg = (
                self.strategy
                if isinstance(self.strategy, FedAvgSave)
                else self.strategy.inner_strategy
            )

            client2model = {}
            for cid, weights in fedavg.client2ws.items():
                m_dict = initialize_model(
                    self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
                )
                model = m_dict["model"]
                if weights is not None:
                    model.load_state_dict(weights)
                model.eval()
                client2model[cid] = model

            if not hasattr(fedavg, "gm_ws"):
                logging.warning(
                    f"gm_ws not set — skipping provenance analysis for round {server_round}"
                )
                return loss, {"accuracy": acc, "loss": loss, "round": server_round}

            prov_global_model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            prov_global_model = prov_global_model_dict["model"]
            prov_global_model.load_state_dict(fedavg.gm_ws)
            prov_global_model.eval()

            provenance_input = {
                "train_cfg": self.cfg.tool.tracefl,
                "prov_cfg": self.cfg,
                "round_key": f"{self.cfg.tool.tracefl.exp_key}",
                "central_test_data": self.server_testdata,
                "client2model": client2model,
                "client2num_examples": fedavg.client2num_examples,
                "prov_global_model": prov_global_model,
                "ALLROUNDSCLIENTS2CLASS": fedavg.client2class,
            }

            logging.info(">> Running provenance analysis...")
            prov_result = round_lambda_prov(**provenance_input)
            logging.info(f">> Provenance analysis completed. Results:\n{prov_result}")

            gc.collect()
            return loss, {
                "accuracy": acc,
                "loss": loss,
                "round": server_round,
                "prov_result": prov_result,
            }

        except Exception as e:
            logging.error(
                f"Evaluation failed during round {server_round}: {str(e)}",
                exc_info=True,
            )
            return loss, {
                "accuracy": 0.0,
                "loss": loss,
                "round": server_round,
                "error": str(e),
            }
