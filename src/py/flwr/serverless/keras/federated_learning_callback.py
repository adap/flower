import os
from io import BytesIO
import json
import logging
from typing import Union
from tensorflow import keras
from flwr.serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr.serverless.federated_node.sync_federated_node import SyncFederatedNode
from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


LOGGER = logging.getLogger(__name__)


class FlwrFederatedCallback(keras.callbacks.Callback):
    def __init__(
        self,
        node: Union[AsyncFederatedNode, SyncFederatedNode],
        num_examples_per_epoch: int,
        x_test=None,
        y_test=None,
        test_batch_size=32,
        test_steps=10,
        postfix_for_federated_metrics="_fed",
        override_metrics_with_aggregated_metrics: bool = False,
        save_model_before_aggregation: bool = False,
        save_model_after_aggregation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node = node
        self.num_examples_per_epoch = num_examples_per_epoch
        self.override_metrics_with_aggregated_metrics = (
            override_metrics_with_aggregated_metrics
        )
        self.save_model_before_aggregation = save_model_before_aggregation
        self.save_model_after_aggregation = save_model_after_aggregation
        self.postfix_for_federated_metrics = postfix_for_federated_metrics
        self.x_test = x_test
        self.y_test = y_test
        self.test_batch_size = test_batch_size
        self.test_steps = test_steps
        self.model_before_aggregation_filename_pattern = (
            "keras/{node_id}/model_before_aggregation_{epoch:05d}.h5"
        )
        self.metrics_before_aggregation_filename_pattern = (
            "keras/{node_id}/metrics_before_aggregation_{epoch:05d}.json"
        )
        self.model_after_aggregation_filename_pattern = (
            "keras/{node_id}/model_after_aggregation_{epoch:05d}.h5"
        )
        self.metrics_after_aggregation_filename_pattern = (
            "keras/{node_id}/metrics_after_aggregation_{epoch:05d}.json"
        )
        self._federated_metrics = {}

    def _save_model_to_shared_folder(self, filename: str):
        folder = self.node.model_store.get_raw_folder()
        key = filename
        # convert model into bytes
        tmp_path = f"tmp_model_{self.node.node_id}.h5"
        self.model.save(tmp_path)
        with open(tmp_path, "rb") as f:
            model_bytes = f.read()
        folder[key] = model_bytes
        # delete
        os.remove(tmp_path)

    def _save_metrics_to_shared_folder(self, filename: str, metrics: dict):
        folder = self.node.model_store.get_raw_folder()
        key = filename
        metrics_bytes = BytesIO()
        simple_metrics = {}
        for k, v in metrics.items():
            try:
                simple_metrics[k] = float(v)
            except:
                pass
        json_str = json.dumps(simple_metrics, indent=2)
        metrics_bytes.write(json_str.encode("utf-8"))
        folder[key] = metrics_bytes.getvalue()

    @property
    def federated_metrics(self):
        """Return the metrics from the federated aggreation process."""
        return self._federated_metrics

    def _save_metrics_before_aggregation(self, logs, node_id, epoch):
        if logs:
            # Save metrics.
            filename = self.metrics_before_aggregation_filename_pattern.format(
                node_id=node_id, epoch=epoch
            )
            self._save_metrics_to_shared_folder(filename, logs)

    def _save_metrics_after_aggregation(self, logs, node_id, epoch):
        if logs:
            # Save metrics.
            filename = self.metrics_after_aggregation_filename_pattern.format(
                node_id=node_id, epoch=epoch
            )
            self._save_metrics_to_shared_folder(filename, logs)

    def _save_model_before_aggregation(self, node_id, epoch):
        if self.save_model_before_aggregation:
            filename = self.model_before_aggregation_filename_pattern.format(
                node_id=node_id, epoch=epoch
            )
            self._save_model_to_shared_folder(filename)

    def _save_model_after_aggregation(self, node_id, epoch):
        if self.save_model_after_aggregation:
            filename = self.model_after_aggregation_filename_pattern.format(
                node_id=node_id, epoch=epoch
            )
            self._save_model_to_shared_folder(filename)
            
    def on_epoch_end(self, epoch: int, logs=None):
        # use the P2PStrategy to update the model.
        node_id = self.node.node_id
        LOGGER.info(f"[flwr_serverless] on_epoch_end, logs={logs}")

        self._save_metrics_before_aggregation(logs, node_id, epoch)
        self._save_model_before_aggregation(node_id, epoch)

        params: Parameters = ndarrays_to_parameters(self.model.get_weights())
        if self.override_metrics_with_aggregated_metrics:
            metrics = logs
        else:
            metrics = {
                k: v
                for k, v in logs.items()
                if not k.endswith(self.postfix_for_federated_metrics)
            }

        updated_params, updated_metrics = self.node.update_parameters(
            params,
            num_examples=self.num_examples_per_epoch,
            epoch=epoch,
            metrics=metrics,
        )
        self._federated_metrics = updated_metrics

        self._save_metrics_after_aggregation(updated_metrics, node_id, epoch)

        # Update the keras model and keras logs.
        if updated_params is not None:
            self.model.set_weights(parameters_to_ndarrays(updated_params))
            self._save_model_after_aggregation(node_id, epoch)
            if updated_metrics is not None:
                if self.override_metrics_with_aggregated_metrics:
                    logs.update(updated_metrics)
                    LOGGER.info(
                        "[flwr_serverless] Metrics in Keras logs object are overriden."
                    )
                else:
                    for key, value in updated_metrics.items():
                        logs[f"{key}{self.postfix_for_federated_metrics}"] = value
                    msg = f"[flwr_serverless] Federated metrics are added to Keras logs object with postfix {self.postfix_for_federated_metrics}."
                    LOGGER.info(msg)

            if self.x_test is not None:
                print("\n=========================== eval inside callback")
                self.model.evaluate(
                    self.x_test,
                    self.y_test,
                    batch_size=self.test_batch_size,
                    steps=self.test_steps,
                    verbose=2,
                )
                print("Done evaluating inside callback =====================\n")
        else:
            print("waiting for other nodes to send their parameters")

        # Keep track of keras logs
        self.logs = logs
        if not self.override_metrics_with_aggregated_metrics:
            assert any(
                key.endswith(self.postfix_for_federated_metrics)
                for key in self.logs.keys()
            ), f"No federated metrics found in Keras logs object. {logs}"
