# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fair Resource Allocation in Federated Learning [Li et al., 2020] strategy.

Paper: openreview.net/pdf?id=ByexElSYDr
"""


from collections.abc import Iterable
from logging import INFO
from typing import Callable, Optional, cast

import numpy as np
from flwr.server import Grid
from flwr.common import Array, ArrayRecord, Message, MetricRecord, NDArray, RecordDict, ConfigRecord
from flwr.common.logger import log

from ..exception import AggregationError
from .fedavg import FedAvg


class QFedAvg(FedAvg):
    """q-FedAvg strategy.

    Implementation based on openreview.net/pdf?id=ByexElSYDr

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation. In case `min_evaluate_nodes`
        is larger than `fraction_evaluate * total_connected_nodes`,
        `min_evaluate_nodes` will still be sampled.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for MetricRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    train_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    q : float (default: 0.2)
        The parameter q that controls the degree of fairness of the algorithm.
        When set to 0, q-FedAvg is equivalent to FedAvg.
    client_learning_rate : float (default: 0.01)
        Local learning rate used by clients during training. This value is used by 
        the strategy to approximate the base Lipschitz constant L, via 
        L = 1 / client_learning_rate.
    train_loss_key : str (default: "train_loss")
        The key within the MetricRecord whose value is used as the training loss when
        aggregating ArrayRecords following q-FedAvg.
    """

    def __init__(  # pylint: disable=R0913, R0917
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: Optional[
            Callable[[list[RecordDict], str], MetricRecord]
        ] = None,
        evaluate_metrics_aggr_fn: Optional[
            Callable[[list[RecordDict], str], MetricRecord]
        ] = None,
        q: float = 0.2,
        client_learning_rate: float = 0.1,
        train_loss_key: str = "train_loss",
    ) -> None:
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
        self.q = q
        self.client_learning_rate = client_learning_rate
        self.train_loss_key = train_loss_key
        self.current_arrays: Optional[ArrayRecord] = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> q-FedAvg settings:")
        log(INFO, "\t|\t├── q: %s", self.q)
        log(INFO, "\t|\t├── client_learning_rate: %s", self.client_learning_rate)
        log(INFO, "\t|\t└── train_loss_key: %s", self.train_loss_key)
        super().summary()


    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        if self.current_arrays is None:
            self.current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)


    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Call FedAvg aggregate_train to perform validation and aggregation
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        # Compute estimate of Lipschitz constant L
        L = 1.0 / self.client_learning_rate

        # Aggregate ArrayRecords using trimmed mean
        # Get the key for the only ArrayRecord and MetricRecord from the first Message
        array_record_key = list(valid_replies[0].content.array_records.keys())[0]
        metric_record_key = list(valid_replies[0].content.metric_records.keys())[0]
        # Preserve keys for arrays in ArrayRecord
        array_keys = list(valid_replies[0].content[array_record_key].keys())

        # q-FedAvg aggregation
        global_weights = 
        sum_delta, sum_h = 

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        return arrays, metrics


def l2_norm(ndarrays: list[NDArray]) -> float:
    """Compute the squared L2 norm of a list of numpy.ndarray."""
    return float(sum(np.sum(np.square(g)) for g in ndarrays))


def compute_delta_and_h(
    global_weights: list[NDArray], 
    local_weights: list[NDArray], q: float, L: float, loss: float
) -> tuple[float, float]:
    """Compute delta and h used in q-FedAvg aggregation."""
    # Compute gradient_k
    grad = [L * (gw - lw) for gw, lw in zip(global_weights, local_weights)]
    # Compute ||w_k - w||^2
    norm = l2_norm(grad)
    # Compute delta_k
    loss_pow_q = np.float_power(loss + 1e-10, q)
    grad = [loss_pow_q * g for g in grad]  # This is delta_k already
    delta = grad  # Avoid duplicate memory usage
    # Compute h_k
    h = q * np.float_power(loss + 1e-10, q - 1) * norm + L * loss_pow_q
    return delta, h
