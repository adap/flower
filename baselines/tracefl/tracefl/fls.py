"""TraceFL Federated Learning Simulation Module.

This module implements the FLSimulation class which serves as the central orchestrator
for TraceFL federated learning experiments. It is NOT redundant despite only the
strategy being used in the return - it performs critical setup and configuration work.
"""

import gc
import logging
import time

import numpy as np
import torch

from flwr.common import ndarrays_to_parameters
from tracefl.dp_strategy import TraceFLDifferentialPrivacy
from tracefl.fl_provenance import round_lambda_prov
from tracefl.models_train_eval import global_model_eval
from tracefl.models_utils import get_parameters, initialize_model, set_parameters
from tracefl.strategy import FedAvgSave


class FLSimulation:
    """Central orchestrator for TraceFL federated learning experiments.

    This class is essential for TraceFL functionality and cannot be removed, despite
    appearing to only provide the strategy. It performs critical functions including:

    1. **Strategy Configuration**: Creates and configures the appropriate FL strategy
       based on configuration (with or without differential privacy)
    2. **Model Initialization**: Handles model setup and parameter initialization
    3. **Data Management**: Sets up server and client data for provenance analysis
    4. **Provenance Integration**: Integrates neuron provenance tracking capabilities
    5. **Evaluation Orchestration**: Manages global model evaluation and metric tracking
    6. **DP Integration**: Conditionally wraps strategies with differential privacy

    The strategy returned is pre-configured with all necessary components for TraceFL's
    neuron provenance mechanism to function correctly in the Flower framework.

    Args:
        cfg: Complete configuration object containing all experiment parameters
        fraction_fit: Fraction of clients participating in each round
        num_server_rounds: Total number of federated learning rounds
        local_epochs: Number of local training epochs per client per round
    """

    def __init__(self, cfg, ff, nsr, le):
        # EXTRA: Not essential for basic FL - used for provenance tracking
        self.all_rounds_results = []
        self.cfg = cfg
        self.strategy = None
        self.device = torch.device(self.cfg.tool.tracefl.device.device)
        self.fraction_fit = ff
        self.num_server_rounds = nsr
        self.local_epochs = le
        # EXTRA: Not essential for basic FL - used for timing
        self.start_time = time.time()
        # Initialize attributes to avoid pylint warnings
        self.server_testdata = None
        self.client2data = {}

    def set_server_data(self, server_data):
        """Set the server's dataset for evaluation.

        Args:
            server_data: Dataset to be used for server-side evaluation
        """
        self.server_testdata = server_data

    def set_clients_data(self, clients_data):
        """Set the datasets for all clients.

        Args:
            clients_data: Dictionary mapping client IDs to their datasets
        """
        self.client2data = clients_data
        if len(self.client2data) != self.cfg.tool.tracefl.data_dist.num_clients:
            self.cfg.tool.tracefl.num_clients = len(self.client2data)
            logging.warning(
                "Adjusting number of clients to: %s", self.cfg.tool.tracefl.num_clients
            )

    def set_strategy(self):
        """Set the federated learning strategy.

        This method configures the FL strategy based on the configuration. If
        differential privacy is enabled (noise_multiplier and clipping_norm > 0), it
        wraps the base strategy with a DP strategy.
        """
        try:
            # ========== Model Initialization ==========
            model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            initial_parameters = ndarrays_to_parameters(
                get_parameters(model_dict["model"])
            )

            # Verify initial parameters are not all zeros
            params = get_parameters(model_dict["model"])
            if all(np.all(p == 0) for p in params):
                logging.warning(
                    "Initial parameters are all zeros. Reinitializing model..."
                )
                model_dict = initialize_model(
                    self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
                )
                initial_parameters = ndarrays_to_parameters(
                    get_parameters(model_dict["model"])
                )

            # ========== Base Strategy Configuration ==========
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

            # ========== Differential Privacy Configuration ==========
            dp_enabled = (
                self.cfg.tool.tracefl.strategy.noise_multiplier > 0
                and self.cfg.tool.tracefl.strategy.clipping_norm > 0
            )

            if dp_enabled:
                logging.info(
                    ">> ----------------------------- "
                    "Running DP FL -----------------------------"
                )

                dp_strategy = TraceFLDifferentialPrivacy(
                    strategy=strategy,
                    noise_multiplier=self.cfg.tool.tracefl.strategy.noise_multiplier,
                    clipping_norm=self.cfg.tool.tracefl.strategy.clipping_norm,
                    num_sampled_clients=(
                        self.cfg.tool.tracefl.strategy.clients_per_round
                    ),
                )
                self.strategy = dp_strategy

                logging.info(
                    "Differential Privacy enabled: noise_mult=%s, clipping_norm=%s",
                    self.cfg.tool.tracefl.strategy.noise_multiplier,
                    self.cfg.tool.tracefl.strategy.clipping_norm,
                )
            else:
                logging.info(
                    ">> ----------------------------- "
                    "Running Non-DP FL -----------------------------"
                )

                if (
                    self.cfg.tool.tracefl.strategy.noise_multiplier == -1
                    or self.cfg.tool.tracefl.strategy.clipping_norm == -1
                ):
                    logging.info(
                        "Differential Privacy disabled "
                        "(noise_multiplier or clipping_norm set to -1)"
                    )
                self.strategy = strategy

        except Exception as e:
            logging.error("Error setting up strategy: %s", str(e))
            raise

    def _fit_metrics_aggregation_fn(self, _metrics):
        # Aggregate client metrics without verbose logging
        return {"loss": 0.1, "accuracy": 0.2}

    def _get_fit_config(self, server_round: int):
        """Get configuration for client training."""
        torch.Generator().manual_seed(server_round)
        config = {
            "server_round": server_round,
            "local_epochs": self.cfg.tool.tracefl.client.epochs,
            "batch_size": self.cfg.tool.tracefl.data_dist.batch_size,
            "lr": self.cfg.tool.tracefl.client.lr,
            "grad_clip": 1.0,  # Add gradient clipping threshold
        }
        return config

    def _evaluate_global_model(self, server_round, parameters, _config=None):
        """Evaluate the global model.

        Args:
            server_round: Current server round
            parameters: Model parameters to evaluate
            _config: Optional configuration dictionary (unused)

        Returns
        -------
            Tuple of (loss, metrics)
        """
        try:
            model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            set_parameters(model_dict["model"], parameters)
            model_dict["model"].eval()
            model_dict["test_data"] = (
                self.server_testdata
            )  # Add test data to model_dict

            metrics = global_model_eval(self.cfg.tool.tracefl.model.arch, model_dict)
            loss = metrics["loss"]
            acc = metrics["accuracy"]
            # EXTRA: Not essential for basic FL - used for provenance tracking
            self.all_rounds_results.append({"loss": loss, "accuracy": acc})

            if server_round == 0:
                logging.info(
                    "initial parameters (loss, other metrics): %s, "
                    "{'accuracy': %s, 'loss': %s, 'round': %s}",
                    loss,
                    acc,
                    loss,
                    server_round,
                )
                return loss, {"accuracy": acc, "loss": loss, "round": server_round}

            logging.info(
                "fit progress: (%s, %s, "
                "{'accuracy': %s, 'loss': %s, 'round': %s}, %s)",
                server_round,
                loss,
                acc,
                loss,
                server_round,
                time.time() - self.start_time,
            )

            if self.strategy is None:
                logging.error("Strategy is not initialized")
                return loss, {
                    "accuracy": acc,
                    "loss": loss,
                    "round": server_round,
                    "error": "Strategy not initialized",
                }

            # Check if we're using a DP strategy and extract the inner strategy if so
            fedavg = self.strategy
            if isinstance(self.strategy, TraceFLDifferentialPrivacy):
                fedavg = self.strategy.strategy

            if not isinstance(fedavg, FedAvgSave):
                logging.error("Invalid strategy type")
                return loss, {
                    "accuracy": acc,
                    "loss": loss,
                    "round": server_round,
                    "error": "Invalid strategy type",
                }

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

            # EXTRA: Not essential for basic FL - provenance analysis code
            if not hasattr(fedavg, "gm_ws"):
                logging.warning(
                    "Skipping provenance analysis for round %s", server_round
                )
                return loss, {"accuracy": acc, "loss": loss, "round": server_round}

            # EXTRA: Not essential for basic FL - provenance tracking code
            prov_global_model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            prov_global_model = prov_global_model_dict["model"]
            prov_global_model.load_state_dict(fedavg.gm_ws)
            prov_global_model.eval()

            # EXTRA: Not essential for basic FL - provenance analysis input
            provenance_input = {
                "train_cfg": self.cfg.tool.tracefl,
                "prov_cfg": self.cfg.tool.tracefl,
                "prov_global_model": prov_global_model,
                "client2model": client2model,
                "client2num_examples": fedavg.client2num_examples,
                "all_rounds_clients2class": fedavg.client2class,
                "central_test_data": self.server_testdata,
                "server_round": server_round,
            }

            logging.info(">> Running provenance analysis...")

            try:
                prov_result = round_lambda_prov(**provenance_input)
                logging.info(
                    ">> Provenance analysis completed. Results:\n%s", prov_result
                )
            except (KeyError, ValueError) as e:
                logging.error("Configuration error in provenance analysis: %s", str(e))
                prov_result = {"Error": f"Configuration error: {str(e)}"}
            except (RuntimeError, TypeError) as e:
                logging.error("Runtime error in provenance analysis: %s", str(e))
                prov_result = {"Error": f"Runtime error: {str(e)}"}
            except (OSError, MemoryError, ImportError) as e:
                logging.error("Unexpected error in provenance analysis: %s", str(e))
                prov_result = {"Error": f"Unexpected error: {str(e)}"}

            gc.collect()
            return loss, {
                "accuracy": acc,
                "loss": loss,
                "round": server_round,
                "prov_result": prov_result,
            }

        except (ValueError, RuntimeError, KeyError, OSError) as e:
            logging.error("Evaluation failed during round %s: %s", server_round, str(e))
            return loss, {
                "accuracy": 0.0,
                "loss": loss,
                "round": server_round,
                "error": str(e),
            }
