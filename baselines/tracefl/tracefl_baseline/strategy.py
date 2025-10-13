"""TraceFL Strategy extending FedAvg for provenance analysis."""
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable

from flwr.common import MetricRecord
from flwr.serverapp.strategy import FedAvg
from flwr.common.logger import log

from .fl_provenance import FlowerProvenance


class TraceFL_Strategy(FedAvg):
    """Custom Flower strategy that extends FedAvg with TraceFL provenance analysis."""

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: Optional[Callable[[list, str], MetricRecord]] = None,
        evaluate_metrics_aggr_fn: Optional[Callable[[list, str], MetricRecord]] = None,
        # TraceFL-specific parameters
        provenance_rounds: Optional[List[int]] = None,
        enable_beta: bool = True,
        client_weights_normalization: bool = True,
        cfg: Optional[object] = None,
    ) -> None:
        """Initialize TraceFL strategy.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during evaluation. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during evaluation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[callable], optional
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Optional[callable], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Optional[callable], optional
            Function used to configure evaluation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Optional[bytes], optional
            Initial global model parameters. Defaults to None.
        fit_metrics_aggregation_fn : Optional[callable], optional
            Metrics aggregation function for fit. Defaults to None.
        evaluate_metrics_aggregation_fn : Optional[callable], optional
            Metrics aggregation function for evaluate. Defaults to None.
        provenance_rounds : Optional[List[int]], optional
            List of rounds to run provenance analysis. Defaults to None.
        enable_beta : bool, optional
            Enable layer importance weighting. Defaults to True.
        client_weights_normalization : bool, optional
            Normalize client contributions. Defaults to True.
        cfg : Optional[object], optional
            Configuration object. Defaults to None.
        """
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        
        # TraceFL-specific attributes
        self.provenance_rounds = provenance_rounds or []
        self.enable_beta = enable_beta
        self.client_weights_normalization = client_weights_normalization
        self.cfg = cfg

        # Storage for client models and metadata
        self.client_models = {}  # round_id -> {client_id -> model_state_dict}
        self.client_num_examples = {}  # round_id -> {client_id -> num_examples}
        self.server_test_data = None
        self.client2class = {}  # Mapping from client ID to label counts
        self._result_logger = None

        log(logging.INFO, f"TraceFL Strategy initialized with provenance rounds: {self.provenance_rounds}")

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ):
        """Aggregate train results and store client models for provenance analysis."""
        
        # Call parent aggregate_train
        arrays, metrics = super().aggregate_train(server_round, replies)
        
        # Store client models and metadata for provenance analysis
        replies_list = list(replies)
        if replies_list:
            log(logging.INFO, f"Processing {len(replies_list)} replies for round {server_round}")
            
            self.client_models[server_round] = {}
            self.client_num_examples[server_round] = {}
            
            # Extract client models from Message objects
            for msg in replies_list:
                if not msg.has_error():
                    # Extract client ID from Flower's node_id (matching TraceFL-Flower-Baseline)
                    flower_node_id = msg.metadata.src_node_id
                    
                    # Create sequential mapping if not exists
                    if not hasattr(self, '_node_id_to_client_id'):
                        self._node_id_to_client_id = {}
                        self._next_client_id = 0
                    
                    if flower_node_id not in self._node_id_to_client_id:
                        self._node_id_to_client_id[flower_node_id] = self._next_client_id
                        self._next_client_id += 1
                    
                    client_id = self._node_id_to_client_id[flower_node_id]
                    
                    # Extract num_examples from metrics
                    metric_content = msg.content.get("metrics")
                    if metric_content is not None:
                        num_examples = metric_content.get(self.weighted_by_key, 0)
                        
                        # Extract ArrayRecord (model weights)
                        arrayrecord = msg.content.get(self.arrayrecord_key)
                        if arrayrecord:
                            # Convert ArrayRecord to PyTorch state_dict
                            state_dict = arrayrecord.to_torch_state_dict()
                            self.client_models[server_round][client_id] = state_dict
                        
                        self.client_num_examples[server_round][client_id] = num_examples
            
            log(logging.INFO, f"Stored models for {len(self.client_models[server_round])} clients in round {server_round}")
            
            # Trigger provenance analysis if this round is in provenance_rounds
            if server_round in self.provenance_rounds:
                self._run_provenance_analysis(server_round, arrays)
        
        return arrays, metrics

    def _run_provenance_analysis(self, server_round: int, global_arrays):
        """Run provenance analysis for the current round."""
        
        if not self.cfg or not self.server_test_data:
            log(logging.WARNING, "Cannot run provenance analysis: missing config or test data")
            return
        
        try:
            
            # Create global model from aggregated ArrayRecord
            from .model import initialize_model
            import torch
            
            model_dict = initialize_model(self.cfg.data_dist.model_name, self.cfg.data_dist)
            global_model = model_dict["model"]
            
            # Load global parameters from ArrayRecord
            if global_arrays:
                state_dict = global_arrays.to_torch_state_dict()
                # Convert numpy arrays to tensors
                state_dict = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                              for k, v in state_dict.items()}
                global_model.load_state_dict(state_dict)
            
            # Create client models
            client2model = {}
            for client_id, state_dict in self.client_models[server_round].items():
                client_model_dict = initialize_model(self.cfg.data_dist.model_name, self.cfg.data_dist)
                client_model = client_model_dict["model"]
                # Convert numpy arrays to tensors
                state_dict_tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                                      for k, v in state_dict.items()}
                client_model.load_state_dict(state_dict_tensors)
                client2model[client_id] = client_model
            
            # Initialize provenance analysis
            provenance = FlowerProvenance(
                cfg=self.cfg,
                round_id=server_round,
                server_test_data=self.server_test_data,
                global_model=global_model,
                client2model=client2model,
                client2num_examples=self.client_num_examples[server_round],
                client2class=self.client2class,
            )
            
            # Calculate client contributions
            results = provenance.calculate_client_contributions()

            if "error" not in results:
                log(logging.INFO, f"✅ Provenance analysis completed for round {server_round}")
                log(logging.INFO, f"📊 Samples analyzed: {results['samples_analyzed']}")
                log(logging.INFO, f"🎯 Top contributor: Client {results['top_contributor']}")

                # Persist results for downstream plotting utilities
                try:
                    result_logger = self._get_result_logger()
                    result_logger.record_round(server_round, results)
                    log(
                        logging.INFO,
                        "💾 Stored provenance metrics in %s",
                        result_logger.file_path,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    log(logging.WARNING, f"⚠️ Failed to persist provenance metrics: {exc}")
            else:
                log(logging.ERROR, f"❌ Provenance analysis failed: {results['error']}")
                
        except Exception as e:
            import traceback
            log(logging.ERROR, f"❌ Error in provenance analysis: {e}")
            log(logging.ERROR, f"📋 Traceback:\n{traceback.format_exc()}")
            log(logging.ERROR, f"📋 Error type: {type(e)}")
            log(logging.ERROR, f"📋 Error args: {e.args}")

    def set_server_test_data(self, test_data):
        """Set server test data for provenance analysis."""
        self.server_test_data = test_data
        log(logging.INFO, f"Server test data set: {len(test_data)} samples")

    def set_client2class(self, client2class):
        """Set client-to-class mappings for provenance analysis."""
        normalized = {}
        for cid, label_counts in client2class.items():
            try:
                client_id = int(cid)
            except (TypeError, ValueError):
                client_id = cid

            normalized_counts = {}
            for label, count in (label_counts or {}).items():
                if isinstance(count, torch.Tensor):
                    count = int(count.item())
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                label_str = str(label.item() if hasattr(label, "item") else label)
                normalized_counts[label_str] = count_int

            normalized[client_id] = normalized_counts

        self.client2class = normalized
        log(logging.INFO, f"Client2class mapping set for {len(self.client2class)} clients")

    def configure_fit(
        self, server_round: int, parameters: bytes, client_manager
    ) -> List[Tuple[int, Dict[str, Union[bool, bytes, float, int, str]]]]:
        """Configure the next round of training."""
        config = super().configure_fit(server_round, parameters, client_manager)
        
        # Add TraceFL-specific configuration
        if self.cfg:
            for i, (client_id, client_config) in enumerate(config):
                config[i] = (client_id, {
                    **client_config,
                    "tracefl": {
                        "model_name": self.cfg.data_dist.model_name,
                        "model_architecture": self.cfg.data_dist.model_architecture,
                        "num_classes": self.cfg.data_dist.num_classes,
                        "channels": self.cfg.data_dist.channels,
                    }
                })

        return config

    def _get_result_logger(self):
        if self._result_logger is None and self.cfg is not None:
            from .results_logging import ExperimentResultLogger

            self._result_logger = ExperimentResultLogger(self.cfg)
        return self._result_logger
