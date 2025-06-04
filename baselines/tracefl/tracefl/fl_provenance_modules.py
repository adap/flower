"""Federated Learning Provenance Core Components.

This module contains the core components for tracking and analyzing provenance in
federated learning systems. It includes classes and functions for computing client
contributions, analyzing model updates, and tracking the evolution of model parameters
across training rounds.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

import torch

from tracefl.models_train_eval import test_neural_network
from tracefl.models_utils import initialize_model
from tracefl.neuron_provenance import NeuronProvenance, get_all_layers
from tracefl.utils import get_prov_eval_metrics, safe_len


class FederatedProvTrue:
    """A class for tracking and analyzing federated learning provenance.

    This class implements the core functionality for tracking and analyzing the
    provenance of model updates during federated learning rounds.
    """

    def __init__(
        self,
        train_cfg: Any,
        prov_cfg: Any,
        round_key: str,
        *,
        server_test_data: Any,
        client2model: Dict[str, Any],
        client2num_examples: Dict[str, int],
        prov_global_model: Any,
        all_rounds_clients2class: Dict[str, Dict[int, int]],
        t: Optional[Any] = None,
    ) -> None:
        """Initialize the FederatedProvTrue instance."""
        self.train_cfg = train_cfg
        self.prov_cfg = prov_cfg
        self.round_id = round_key
        self.server_test_data = server_test_data
        self.client2model = client2model
        self.client2num_examples = client2num_examples
        self.prov_global_model = prov_global_model
        self.all_rounds_clients2class = all_rounds_clients2class
        self.t = t
        self.neuron_provenance: Optional[NeuronProvenance] = None
        self.subset_test_data: Optional[Any] = None
        self.loss: float = 0.0
        self.acc: float = 0.0
        self.participating_clients_labels: List[str] = []

        self._set_participating_clients_labels()
        self._select_provenance_data(server_test_data)

    def _model_initialize_wrapper(self):
        """Model initialize wrapper."""
        m = initialize_model(self.train_cfg.model.name, self.train_cfg.dataset)
        return m["model"]

    def _set_participating_clients_labels(self) -> None:
        """Set the labels for participating clients."""
        labels: Set[str] = set()
        for c in self.client2model.keys():
            client_label_data = self.all_rounds_clients2class[c]
            if isinstance(client_label_data, str):
                client_label_data = json.loads(client_label_data)
            labels = labels.union({str(k) for k in client_label_data.keys()})
        self.participating_clients_labels = list(labels)
        logging.debug(
            "participating_clients_labels: %s", self.participating_clients_labels
        )

    def _eval_and_extract_correct_preds_transformer(self, test_data):
        """Evaluate and extract correct preds."""
        d = test_neural_network(
            self.train_cfg.model.arch, {"model": self.prov_global_model}, test_data
        )
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug("Accuracy on test data: %s", self.acc)
        return d["eval_correct_indices"], d["eval_actual_labels"]

    def _balance_dataset_by_label(
        self,
        correct_indices: torch.Tensor,
        dataset_labels: torch.Tensor,
        min_per_label: int,
    ) -> List:
        """Balance Dataset by Label."""
        balanced_indices = []
        logging.debug(
            "participating_clients_labels: %s", self.participating_clients_labels
        )
        selected_labels = dataset_labels[correct_indices].tolist()

        for label_str in self.participating_clients_labels:
            temp_bools = [str(label) == label_str for label in selected_labels]
            temp_correct_indxs = [
                correct_indices[i] for i, flag in enumerate(temp_bools) if flag
            ]
            if len(temp_correct_indxs) >= min_per_label:
                balanced_indices.extend(temp_correct_indxs[:min_per_label])

        logging.debug("Balanced indices (list): %s", balanced_indices)
        return balanced_indices

    def _select_provenance_data(
        self, central_test_data: Any, min_per_label: int = 2
    ) -> None:
        """Select data for provenance analysis."""
        all_correct_i, dataset_labels = (
            self._eval_and_extract_correct_preds_transformer(central_test_data)
        )
        balanced_indices = self._balance_dataset_by_label(
            all_correct_i, dataset_labels, min_per_label
        )
        self.subset_test_data = central_test_data.select(balanced_indices)
        if self.subset_test_data is not None and safe_len(self.subset_test_data) == 0:
            logging.warning("No correct predictions found")

    def _sanity_check(self) -> float:
        """Perform sanity check on the selected data."""
        if self.subset_test_data is None or safe_len(self.subset_test_data) == 0:
            raise ValueError("No correct predictions found")

        acc = test_neural_network(
            self.train_cfg.model.arch,
            {"model": self.prov_global_model},
            self.subset_test_data,
        )["accuracy"]

        logging.info("Sanity check: %s", acc)
        assert int(acc) == 1, "Sanity check failed"
        return acc

    def _compute_eval_metrics(self, input2prov: List[Dict]) -> Dict[str, float]:
        """Compute evaluation metrics for provenance analysis.

        Parameters
        ----------
        input2prov : List[Dict]
            List of dictionaries containing provenance data for each input

        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics including accuracy
        """
        # ========== Collect Ground-Truth Labels ==========
        data_loader = torch.utils.data.DataLoader(
            self.subset_test_data,
            batch_size=1,
        )
        target_labels = [row["label"].item() for row in data_loader]

        # ========== Normalize Client Class Mappings ==========
        client2class: Dict[str, Dict[str, int]] = {}
        for cid in self.client2model:
            raw = self.all_rounds_clients2class[cid]
            if isinstance(raw, str):
                raw = json.loads(raw)
            client2class[cid] = {str(k): v for k, v in raw.items()}

        true_labels: List[int] = []
        predicted_labels: List[int] = []

        # ========== Evaluate Each Provenance Record ==========
        for idx, prov_rec in enumerate(input2prov):
            traced_client = prov_rec["traced_client"]
            client2prov = prov_rec["client2prov"]
            target_l = target_labels[idx]
            target_l_str = str(target_l)

            # Find responsible clients for this label
            responsible_clients = [
                cid
                for cid, c_labels in client2class.items()
                if target_l_str in c_labels
            ]

            logging.info(
                "*********** Input Label: %s, Responsible Client(s): %s *************",
                target_l,
                ",".join(f"c{cid}" for cid in responsible_clients),
            )

            # ========== TraceFL Correctness Check ==========
            if target_l_str in client2class[traced_client]:
                logging.info(
                    "     Traced Client: c%s || Tracing = Correct",
                    traced_client,
                )
                predicted_labels.append(1)
            else:
                logging.info(
                    "     Traced Client: c%s || Tracing = Wrong",
                    traced_client,
                )
                predicted_labels.append(0)
            true_labels.append(1)

            # ========== TraceFL Contribution Scores ==========
            contrib_pretty = {f"c{cid}": round(p, 2) for cid, p in client2prov.items()}
            logging.info("TraceFL Clients Contributions Rank: %s\n", contrib_pretty)

        return get_prov_eval_metrics(true_labels, predicted_labels)

    def run(self) -> Dict[str, Any]:
        """Execute the provenance analysis process and return the results.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - clients: List of client IDs
            - data_points: Number of data points used
            - eval_metrics: Dictionary of evaluation metrics
            - test_data_acc: Test accuracy
            - test_data_loss: Test loss
            - prov_time: Time taken for provenance analysis
            - round_id: Current round ID
            - prov_layers: Set of layer types used
            - Error: Error message if analysis failed
        """
        try:
            # ========== Sanity Check ==========
            r = self._sanity_check()
            if r is None:
                return {
                    "clients": list(self.client2model.keys()),
                    "data_points": safe_len(self.subset_test_data),
                    "eval_metrics": {},
                    "test_data_acc": self.acc,
                    "test_data_loss": self.loss,
                    "prov_time": -1,
                    "round_id": self.round_id,
                    "prov_layers": {
                        type(layer) for layer in get_all_layers(self.prov_global_model)
                    },
                }

            # ========== TraceFL Neuron Provenance Analysis ==========
            start_time = time.time()

            # Get num_classes from the correct path in configuration
            num_classes = self.train_cfg.dataset.num_classes

            nprov = NeuronProvenance(
                cfg=self.prov_cfg,
                arch=self.train_cfg.model.arch,
                test_data=self.subset_test_data,
                gmodel=self.prov_global_model,
                c2model=self.client2model,
                num_classes=num_classes,
                c2nk=self.client2num_examples,
            )

            logging.info("client ids: %s", list(self.client2model.keys()))

            # ========== Compute Input Provenance ==========
            input2prov = nprov.compute_input_provenance()
            eval_metrics = self._compute_eval_metrics(input2prov)
            end_time = time.time()

            # ========== TraceFL Results Logging ==========
            logging.info(
                "[Round %s] TraceFL Accuracy = %.2f%%",
                self.round_id,
                eval_metrics["Accuracy"] * 100,
            )
            logging.info(
                "Total Inputs: %d | GM_loss: %.4f | GM_acc: %.4f",
                safe_len(self.subset_test_data),
                self.loss,
                self.acc,
            )

            # ========== Return Results ==========
            return {
                "clients": list(self.client2model.keys()),
                "data_points": safe_len(self.subset_test_data),
                "eval_metrics": eval_metrics,
                "test_data_acc": self.acc,
                "test_data_loss": self.loss,
                "prov_time": end_time - start_time,
                "round_id": self.round_id,
                "prov_layers": {
                    type(layer) for layer in get_all_layers(self.prov_global_model)
                },
            }

        # ========== Error Handling ==========
        except (ValueError, RuntimeError, KeyError, OSError) as e:
            logging.error("Unexpected error in provenance analysis: %s", str(e))
            return {"Error": f"Unexpected error: {str(e)}"}
