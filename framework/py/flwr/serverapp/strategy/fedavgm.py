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
"""Federated Averaging with Momentum (FedAvgM) [Hsu et al., 2019] strategy.

Paper: arxiv.org/pdf/1909.06335.pdf
"""


from collections.abc import Iterable
from logging import INFO
from typing import Callable, Optional

from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    NDArrays,
    RecordDict,
    log,
)
from flwr.server import Grid

from .fedavg import FedAvg
from .result import Result


class FedAvgM(FedAvg):
    """Federated Averaging with Momentum strategy.

    Implementation based on https://arxiv.org/abs/1909.06335

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
        computing weighted averages for both ArrayRecords and MetricRecords.
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
    server_learning_rate: float (default: 1.0)
        Server-side learning rate used in server-side optimization.
    server_momentum: float (default: 0.0)
        Server-side momentum factor used for FedAvgM.
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
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
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
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )
        self.current_arrays: Optional[ArrayRecord] = None
        self.momentum_vector: Optional[NDArrays] = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> Sampling:")
        log(
            INFO,
            "\t│\t├──Fraction: train (%.2f) | evaluate ( %.2f)",
            self.fraction_train,
            self.fraction_evaluate,
        )
        log(
            INFO,
            "\t│\t├──Minimum nodes: train (%d) | evaluate (%d)",
            self.min_train_nodes,
            self.min_evaluate_nodes,
        )
        log(INFO, "\t│\t└──Minimum available nodes: %d", self.min_available_nodes)
        log(INFO, "\t├──> Keys in records:")
        log(INFO, "\t│\t├── Weighted by: '%s'", self.weighted_by_key)
        log(INFO, "\t│\t├── ArrayRecord key: '%s'", self.arrayrecord_key)
        log(INFO, "\t│\t└── ConfigRecord key: '%s'", self.configrecord_key)
        if self.server_opt:
            log(INFO, "\t└──> Server optimizer:")
            log(INFO, "\t\t├── Learning rate: %.4f", self.server_learning_rate)
            log(INFO, "\t\t└── Momentum: %.4f", self.server_momentum)
        else:
            log(INFO, "\t└──> Server optimizer: None")

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Call FedAvg aggregate_train to perform validation and aggregation
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        # following convention described in
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        if self.server_opt and aggregated_arrays is not None:
            # The initial parameters should be set in `start()` method already
            if self.current_arrays is None:
                raise RuntimeError("No initial parameters set for FedAvgM")
            ndarrays = self.current_arrays.to_numpy_ndarrays()
            aggregated_ndarrays = aggregated_arrays.to_numpy_ndarrays()

            # Remember that updates are the opposite of gradients
            pseudo_gradient = [
                old - new for new, old in zip(aggregated_ndarrays, ndarrays)
            ]
            if self.server_momentum > 0.0:
                if server_round == 1:
                    self.momentum_vector = pseudo_gradient
                else:
                    if self.momentum_vector is None:
                        raise RuntimeError("Momentum vector not initialized")
                    self.momentum_vector = [
                        self.server_momentum * mv + pg
                        for mv, pg in zip(self.momentum_vector, pseudo_gradient)
                    ]

                # No nesterov for now
                pseudo_gradient = self.momentum_vector

            # SGD
            aggregated_ndarrays = [
                old - self.server_learning_rate * pg
                for old, pg in zip(ndarrays, pseudo_gradient)
            ]
            del ndarrays, pseudo_gradient  # save memory
            # Ensure aggregated_arrays has the same keys as before
            for key, ndarray in zip(aggregated_arrays.keys(), aggregated_ndarrays):
                aggregated_arrays[key] = Array(ndarray=ndarray)
            del aggregated_ndarrays  # save memory

            # Update current weights
            self.current_arrays = aggregated_arrays

        return aggregated_arrays, aggregated_metrics

    def start(  # pylint: disable=R0913, R0917
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        """Execute the federated learning strategy."""
        # Save initial parameters
        self.current_arrays = initial_arrays
        return super().start(
            grid,
            initial_arrays,
            num_rounds,
            timeout,
            train_config,
            evaluate_config,
            evaluate_fn,
        )
