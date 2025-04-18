# strategy.py

import gc
import logging


import flwr as fl
from tracefl.global_data import update_round_data
from tracefl.models import initialize_model, set_parameters
from tracefl.utils import get_backend_config


class FedAvgSave(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that extends Flower's FedAvg.

    It aggregates client updates (where each client returns a tuple of
    the form: (num_examples, metrics_dict)), then updates an in-memory
    global dictionary with key data (global model state, client weights,
    number of examples, and client-to-class mapping) for use in later
    provenance analysis.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.backend_config = get_backend_config(cfg)
        # These dictionaries are updated each round.
        self.client2ws = {}  # Mapping: client ID -> client model weights (if available)
        self.client2num_examples = (
            {}
        )  # Mapping: client ID -> number of training examples
        self.client2class = {}  # Mapping: client ID -> client-to-class mapping
        self.initial_parameters = None  # Will store the initial global model parameters

    def set_initial_parameters(self, initial_parameters):
        """Store the initial global model parameters."""
        self.initial_parameters = initial_parameters

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client updates using FedAvg and update the in-memory
        global dictionary.
        """
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert aggregated parameters to a PyTorch state_dict.
            gm_ws = self.get_state_dict_from_parameters(aggregated_parameters)
            self.gm_ws = gm_ws  # Save for provenance
            logging.info("Aggregated global model for round %s", server_round)

            # Process each client result.
            for client_proxy, fit_res in results:
                cid = client_proxy.cid  # ✅ Real client ID
                weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
                self.client2ws[cid] = self.get_state_dict_from_parameters(
                    fit_res.parameters
                )
                self.client2num_examples[cid] = fit_res.num_examples

                # ✅ Collect class distribution if present
                if "class_distribution" in fit_res.metrics:
                    self.client2class[cid] = fit_res.metrics["class_distribution"]
                else:
                    self.client2class[cid] = {}

            # Store initial global model parameters on the first round
            if self.initial_parameters is None:
                self.initial_parameters = aggregated_parameters

            # Update round-level provenance info
            update_round_data(
                server_round,
                self.initial_parameters,
                self.client2ws,
                self.client2num_examples,
                self.client2class,
            )

        gc.collect()
        return aggregated_parameters, aggregated_metrics

    def get_state_dict_from_parameters(self, parameters):
        """Convert Flower parameters (from ndarrays) into a PyTorch
        state_dict.
        """
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model_dict = initialize_model(
            self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
        )
        temp_net = model_dict["model"]
        set_parameters(temp_net, ndarr)
        return temp_net.state_dict()
