import logging
from typing import List, Tuple
from uuid import uuid4
import time
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.serverless.shared_folder.base_folder import SharedFolder
from .aggregatable import Aggregatable


LOGGER = logging.getLogger(__name__)


class AsyncFederatedNode:
    """
    Synchronous version:

    8 am:
    client 1 (faster client) sends params1_1
    server has no params yet, so client 1 is told to wait
    server keeps params1_1

    9 am:
    client 2 (slower) sends params2_1 (client 1 is waiting from 8 am to 9 am)
    server aggregated params1_1 and params2_1, and sends back to client 1 and 2
    both client 1 and client 2 updates their local models, and resume training

    10 am:
    client 1: sends params1_2
    ...

    Asynchronous version (client does not wait for the server to get new aggregated weights):

    8 am:
    client 1 sends params1_1
    server returns params1_1, and sets params_federated_0 = params1_1
    client 1 keeps training with params1_1 for 2 hours

    9 am:
    client 2 sends params2_1
    server aggregates params1_1 and params2_1 into params_federated_1
    server returns aggregated params_federated_1
    client 2 updates its params to params_federated_1 and keeps training
    (but client 1 is busy doing its own training now, so it is not updated)

    10 am:
    client 1 sends params1_2
    server aggregates params_federated_1 and params1_2 into params_federated_2
    server returns aggregated params_federated_2
    client 1 updates its params to params_federated_2 and keeps training

    References:
    - [Semi-Synchronous Federated Learning for Energy-Efficient
    Training and Accelerated Convergence in Cross-Silo Settings](https://arxiv.org/pdf/2102.02849.pdf)
    """

    def __init__(
        self,
        shared_folder: SharedFolder,
        strategy: Strategy,
        ignore_seen_models: bool = False,
        node_id: str = None,
    ):
        self.node_id = node_id or str(uuid4())
        self.counter = 0
        self.strategy = strategy
        self.model_store = shared_folder
        self.sample_sizes_from_other_nodes = {}  # node_id -> num_examples
        self.ignore_seen_models = ignore_seen_models
        self.seen_models = set()

    def _aggregate(
        self,
        aggregatables: List[Aggregatable],
    ) -> Aggregatable:
        # Aggregation using the flwr strategy.
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                None,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=param_holder.parameters,
                    num_examples=param_holder.num_examples,
                    metrics=param_holder.metrics,
                ),
            )
            for param_holder in aggregatables
        ]

        aggregated_parameters, aggregated_metrics = self.strategy.aggregate_fit(
            server_round=self.counter + 1, results=results, failures=[]
        )
        aggregated_metrics = self._update_aggregated_metrics_in_case_flwr_did_not_do_it(
            aggregatables, aggregated_metrics
        )

        self.counter += 1
        return Aggregatable(
            parameters=aggregated_parameters,
            num_examples=sum(
                [param_holder.num_examples for param_holder in aggregatables]
            ),
            metrics=aggregated_metrics,
        )

    def _update_aggregated_metrics_in_case_flwr_did_not_do_it(
        self, aggregatables, aggregated_metrics: dict
    ) -> dict:
        if len(aggregated_metrics) == 0:
            aggregated_metrics = {}
            aggregated_metrics["num_examples"] = sum(
                [param_holder.num_examples for param_holder in aggregatables]
            )
            aggregated_metrics["num_nodes"] = len(aggregatables)
            first_metric = aggregatables[0].metrics
            if first_metric is None:
                LOGGER.warning(f"No metrics found in {aggregatables[0]}")
                return aggregated_metrics
            for k, _ in first_metric.items():
                if k in ["num_nodes", "num_examples"]:
                    continue
                aggregated_metrics[k] = (
                    sum(
                        [
                            param_holder.metrics[k] * param_holder.num_examples
                            for param_holder in aggregatables
                        ]
                    )
                    / aggregated_metrics["num_examples"]
                )
        LOGGER.info(f"Aggregated metrics: {aggregated_metrics}")
        return aggregated_metrics

    def _get_aggregatables_from_other_nodes(self) -> List[Aggregatable]:
        unseen_parameters_from_other_nodes = []
        for key, value in self.model_store.items():
            if key.startswith("accum_num_examples_"):
                continue
            if isinstance(value, dict) and "model_hash" in value:
                if key != self.node_id:
                    model_hash = value["model_hash"]
                    if (
                        not self.ignore_seen_models
                    ) or model_hash not in self.seen_models:
                        self.seen_models.add(model_hash)
                        unseen_parameters_from_other_nodes.append(value["aggregatable"])
        return unseen_parameters_from_other_nodes

    def update_parameters(
        self,
        local_parameters: Parameters,
        num_examples: int = None,
        metrics: dict = None,
        epoch: int = None,
        upload_only=False,
    ) -> Tuple[Parameters, dict]:
        LOGGER.info(f"node {self.node_id} @ epoch {epoch}: updating model weights using federated learning.")
        if num_examples is not None:
            assert isinstance(num_examples, int)
            assert num_examples >= 1
            LOGGER.info(f"node {self.node_id} @ epoch {epoch}: {num_examples} local examples contributed to the last model update.")
        self_aggregatable = Aggregatable(
            parameters=local_parameters,
            num_examples=num_examples,
            metrics=metrics,
        )
        self.model_store[self.node_id] = dict(
            aggregatable=self_aggregatable,
            model_hash=self.node_id + "_" + str(time.time()),
            epoch=epoch,
            node_id=self.node_id,
        )
        if upload_only:
            LOGGER.info(f"node {self.node_id} @ epoch {epoch}: uploading local parameters without aggregation, because upload_only is True")
            return local_parameters, metrics
        (aggregatables_from_other_nodes) = self._get_aggregatables_from_other_nodes()
        LOGGER.info(
            f"node {self.node_id} @ epoch {epoch}: found {len(aggregatables_from_other_nodes or [])} models from other nodes"
        )
        if len(aggregatables_from_other_nodes) == 0:
            # No other nodes, so just return the local parameters
            return local_parameters, metrics
        else:
            # Aggregate the parameters from other nodes
            parameters_from_all_nodes = [
                self_aggregatable
            ] + aggregatables_from_other_nodes
            updated_aggregatable = self._aggregate(parameters_from_all_nodes)

            # It is counter-productive to set self.model_store[node_id] to the aggregated parameters.
            # It makes the accuracy worse.
            # self.model_store[self.node_id] = dict(
            #     parameters=aggregated_parameters,
            #     model_hash=self.node_id + str(time.time()),
            #     num_examples=num_examples,
            # )
            aggregated_parameters = updated_aggregatable.parameters
            aggregated_metrics = updated_aggregatable.metrics

            # print the weight delta
            LOGGER.info(
                f"Finished weight aggregation for epoch {epoch} at node {self.node_id}. "
                f"The weight delta is {self._print_weight_delta(previous_weights=local_parameters, new_weights=aggregated_parameters)}"
            )

            return aggregated_parameters, aggregated_metrics

    def _print_weight_delta(
        self, previous_weights: Parameters, new_weights: Parameters
    ) -> float:
        if previous_weights is None:
            return
        # convert to numpy
        previous_weights_np = parameters_to_ndarrays(previous_weights)
        new_weights_np = parameters_to_ndarrays(new_weights)
        delta = 0
        count = 0
        for w1, w2 in zip(previous_weights_np, new_weights_np):
            delta += float(abs(w1 - w2).sum())
            count += w1.size
        avg_l1_diff = delta / float(count)
        LOGGER.info(f"    Weight delta (average absolute difference): {avg_l1_diff}")
        return avg_l1_diff
