"""TraceFL Federated Learning Strategy Module.

This module implements custom federated learning strategies for TraceFL, including the
FedAvgSave strategy which extends the standard FedAvg algorithm with provenance tracking
capabilities.
"""

# strategy.py

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import flwr as fl
from tracefl.global_data import update_round_data
from tracefl.models_utils import initialize_model, set_parameters


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
        self.client2ws = {}  # Mapping: client ID -> client model weights (if available)
        self.client2num_examples = (
            {}
        )  # Mapping: client ID -> number of training examples
        self.client2class = {}  # Mapping: client ID -> client-to-class mapping
        self.initial_parameters = None  # Will store the initial global model parameters
        # Initialize attribute to avoid pylint warnings
        self.gm_ws = None

    def set_initial_parameters(self, initial_parameters):
        """Store the initial global model parameters."""
        # EXTRA: Not essential for basic FL - used for provenance tracking
        self.initial_parameters = initial_parameters

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client updates using FedAvg.

        Update the in-memory global dictionary with key data for provenance
        analysis.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            List of tuples containing client proxy and fit results.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            List of tuples containing client proxy and fit results that failed.

        Returns
        -------
        Optional[Parameters]
            Aggregated model parameters if successful, None otherwise.
        """
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert aggregated parameters to a PyTorch state_dict.
            gm_ws = self.get_state_dict_from_parameters(aggregated_parameters)
            # EXTRA: Not essential for basic FL - used for provenance tracking
            self.gm_ws = gm_ws  # Save for provenance
            logging.info("Aggregated global model for round %s", server_round)

            # Process each client result.
            for client_proxy, fit_res in results:
                cid = client_proxy.cid
                # EXTRA: Not essential for basic FL - used for provenance tracking
                self.client2ws[cid] = self.get_state_dict_from_parameters(
                    fit_res.parameters
                )
                self.client2num_examples[cid] = fit_res.num_examples

                # EXTRA: Not essential for basic FL - used for provenance tracking
                if "class_distribution" in fit_res.metrics:
                    self.client2class[cid] = fit_res.metrics["class_distribution"]
                else:
                    self.client2class[cid] = {}

            # EXTRA: Not essential for basic FL - used for provenance tracking
            if self.initial_parameters is None:
                self.initial_parameters = aggregated_parameters

            # EXTRA: Not essential for basic FL - used for provenance tracking
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
        """Convert Flower parameters (from ndarrays) into a PyTorch state_dict."""
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model_dict = initialize_model(
            self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
        )
        temp_net = model_dict["model"]
        set_parameters(temp_net, ndarr)
        return temp_net.state_dict()


class TraceFLStrategy(FedAvgSave):
    """Federated Averaging strategy with provenance tracking."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the strategy.

        Parameters
        ----------
        *args : Any
            Variable length argument list.
        **kwargs : Any
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.client2ws: Dict[str, Any] = {}
        self.client2num_examples: Dict[str, int] = {}
        self.client2class: Dict[str, Any] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[
                Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes],
                BaseException,
            ]
        ],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        """Aggregate fit results and update global model parameters.

        This method aggregates the model parameters from all clients that participated
        in the current round of training. It uses weighted averaging based on the
        number of examples each client used for training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            List of tuples containing client proxy and fit results.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            List of tuples containing client proxy and fit results that failed.

        Returns
        -------
        Tuple[Optional[Parameters], Dict[str, Any]]
            A tuple containing:
                - Aggregated model parameters if successful, None otherwise
                - Dictionary of aggregated metrics
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
        """Convert Flower parameters (from ndarrays) into a PyTorch state_dict."""
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        model_dict = initialize_model(
            self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
        )
        temp_net = model_dict["model"]
        set_parameters(temp_net, ndarr)
        return temp_net.state_dict()
