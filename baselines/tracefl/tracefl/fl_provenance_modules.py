import json
import logging
import time
from typing import Dict, List

import torch

from tracefl.models import initialize_model, test_neural_network
from tracefl.neuron_provenance import NeuronProvenance, getAllLayers
from tracefl.utils import get_prov_eval_metrics


class FederatedProvTrue:
    def __init__(
        self,
        train_cfg,
        prov_cfg,
        round_key: str,
        server_test_data,
        client2model,
        client2num_examples,
        prov_global_model,
        ALLROUNDSCLIENTS2CLASS,
        t=None,
    ) -> None:
        self.t = t
        self.prov_cfg = prov_cfg
        self.train_cfg = train_cfg
        self.round_key = round_key
        self.round_id = self.round_key.split(":")[-1]
        self.client2model = client2model
        self.client2num_examples = client2num_examples
        self.central_test_data = server_test_data
        self.prov_global_model = prov_global_model
        self.ALLROUNDSCLIENTS2CLASS = ALLROUNDSCLIENTS2CLASS
        self._setParticipatingClientsLabels()
        self._selectProvenanceData(server_test_data)

    def _modelInitializeWrapper(self):
        m = initialize_model(self.train_cfg.model.name, self.train_cfg.dataset)
        return m["model"]

    def _setParticipatingClientsLabels(self) -> None:
        labels = set()
        for c in self.client2model.keys():
            client_label_data = self.ALLROUNDSCLIENTS2CLASS[c]
            # If the data is a string, parse it to a dictionary
            if isinstance(client_label_data, str):
                client_label_data = json.loads(client_label_data)
            # Use the keys as the labels
            labels = labels.union(set(client_label_data.keys()))
        self.participating_clients_labels = list(labels)
        logging.debug(
            f"participating_clients_labels: {self.participating_clients_labels}"
        )

    def _evalAndExtractCorrectPredsTransformer(self, test_data):
        d = test_neural_network(
            self.train_cfg.model.arch, {"model": self.prov_global_model}, test_data
        )
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug(f"Accuracy on test data: {self.acc}")
        return d["eval_correct_indices"], d["eval_actual_labels"]

    def _balanceDatasetByLabel(
        self,
        correct_indices: torch.Tensor,
        dataset_labels: torch.Tensor,
        min_per_label: int,
    ) -> List:
        balanced_indices = []
        logging.debug(
            f"participating_clients_labels: {self.participating_clients_labels}"
        )
        # Convert dataset labels (at the given indices) to a list of strings
        selected_labels = dataset_labels[correct_indices].tolist()
        for l in self.participating_clients_labels:
            temp_bools = [str(label) == l for label in selected_labels]
            # Collect indices where the flag is True
            temp_correct_indxs = [
                correct_indices[i] for i, flag in enumerate(temp_bools) if flag
            ]
            if len(temp_correct_indxs) >= min_per_label:
                balanced_indices.extend(temp_correct_indxs[:min_per_label])
        logging.debug(f"Balanced indices (list): {balanced_indices}")
        return balanced_indices

    def _selectProvenanceData(self, central_test_data, min_per_label: int = 2) -> None:
        all_correct_i, dataset_labels = self._evalAndExtractCorrectPredsTransformer(
            central_test_data
        )
        balanced_indices = self._balanceDatasetByLabel(
            all_correct_i, dataset_labels, min_per_label
        )
        self.subset_test_data = central_test_data.select(balanced_indices)
        if len(self.subset_test_data) == 0:
            logging.info("No correct predictions found")

    def _sanityCheck(self):
        if len(self.subset_test_data) == 0:
            raise ValueError("No correct predictions found")
        acc = test_neural_network(
            self.train_cfg.model.arch,
            {"model": self.prov_global_model},
            self.subset_test_data,
        )["accuracy"]
        logging.info(f"Sanity check: {acc}")
        assert int(acc) == 1, "Sanity check failed"
        return acc

    def _computeEvalMetrics(self, input2prov: List[Dict]) -> Dict[str, float]:

        data_loader = torch.utils.data.DataLoader(self.subset_test_data, batch_size=1)
        target_labels = [data["label"].item() for data in data_loader]

        client2class = {
            c: json.loads(self.ALLROUNDSCLIENTS2CLASS[c]) for c in self.client2model
        }

        logging.debug(f"client2class: {client2class}")
    

        correct_tracing = 0
        true_labels = []
        predicted_labels = []

        for idx, prov_r in enumerate(input2prov):
            traced_client = prov_r["traced_client"]
            client2prov = prov_r["client2prov"]
            target_l = target_labels[idx]
            target_l_str = str(
                target_l
            ) 
            responsible_clients = [
                cid
                for cid, c_labels in client2class.items()
                if target_l_str in c_labels
            ]

            res_c_string = ",".join(map(str, [f"c{c}" for c in responsible_clients]))
            logging.info(
                f"*********** Input Label: {target_l}, Responsible Client(s): {res_c_string} *************"
            )

            # Check if traced client is among responsible ones
            if target_l_str in client2class[traced_client]:
                logging.info(f"Traced Client: c{traced_client} || Tracing = Correct")
                correct_tracing += 1
                predicted_labels.append(1)
                true_labels.append(1)
            else:
                logging.info(f"Traced Client: c{traced_client} || Tracing = Wrong")
                predicted_labels.append(0)
                true_labels.append(1)

            client2prov_score = {f"c{c}": round(p, 2) for c, p in client2prov.items()}
            logging.info(f"TraceFL Clients Contributions Rank: {client2prov_score}\n")

        eval_metrics = get_prov_eval_metrics(true_labels, predicted_labels)
        return eval_metrics

    def run(self) -> Dict[str, any]:
        # Run sanity check first
        self._sanityCheck()

        start_time = time.time()
        nprov = NeuronProvenance(
            cfg=self.prov_cfg,
            arch=self.train_cfg.model.arch,
            test_data=self.subset_test_data,
            gmodel=self.prov_global_model,
            c2model=self.client2model,
            num_classes=self.train_cfg.dataset.num_classes,
            c2nk=self.client2num_examples,
        )

        input2prov = nprov.computeInputProvenance()
        eval_metrics = self._computeEvalMetrics(input2prov)
        end_time = time.time()

        logging.info(
            f"[Round {self.round_id}] TraceFL Localization Accuracy = {eval_metrics['Accuracy']*100} || "
            f"Total Inputs Used In Prov: {len(self.subset_test_data)} || GM_(loss, acc) ({self.loss},{self.acc})"
        )

        prov_result = {
            "clients": list(self.client2model.keys()),
            "data_points": len(self.subset_test_data),
            "eval_metrics": eval_metrics,
            "test_data_acc": self.acc,
            "test_data_loss": self.loss,
            "prov_time": end_time - start_time,
            "round_id": self.round_id,
            "prov_layers": set([type(l) for l in getAllLayers(self.prov_global_model)]),
        }

        return prov_result
