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
"""Adaptive Federated Optimization (FedOpt) [Reddi et al., 2020] abstract strategy.

Paper: arxiv.org/abs/2003.00295
"""

from collections.abc import Iterable
from logging import INFO
from typing import Callable, Optional

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    NDArray,
    RecordDict,
    log,
)
from flwr.server import Grid

from .fedavg import FedAvg


# pylint: disable=line-too-long
class FedOpt(FedAvg):
    """Federated Optim strategy.

    Implementation based on https://arxiv.org/abs/2003.00295v5

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
    eta : float, optional
        Server-side learning rate. Defaults to 1e-1.
    eta_l : float, optional
        Client-side learning rate. Defaults to 1e-1.
    beta_1 : float, optional
        Momentum parameter. Defaults to 0.0.
    beta_2 : float, optional
        Second moment parameter. Defaults to 0.0.
    tau : float, optional
        Controls the algorithm's degree of adaptability. Defaults to 1e-3.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals, line-too-long
    def __init__(
        self,
        *,
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
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.0,
        beta_2: float = 0.0,
        # TODO: changed from 1e-9 to 1e-3
        # TODO: As per paper (see paragraph just before 5.3)
        tau: float = 1e-3,
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
        # TODO: dict[str, NDArray]
        # TODO: i.e. {k: array.numpy() for k,array in <arrayrecord>.items()})
        # TODO: should we make it a new type?
        self.current_arrays: Optional[dict[str, NDArray]] = None
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_t: Optional[dict[str, NDArray]] = None
        self.v_t: Optional[dict[str, NDArray]] = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        super().summary()
        log(INFO, "\t├──> FedOpt settings:")
        log(
            INFO,
            "\t│\t├──eta (%.2f) | eta_l ( %.2f)",
            self.eta,
            self.eta_l,
        )
        log(
            INFO,
            "\t│\t├──beta_1 (%.2f) | beta_2 ( %.2f)",
            self.beta_1,
            self.beta_2,
        )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Keep track of array record being communicated
        # TODO: a point of diff w.r.t previous strategies.
        # the current weights was always set (externally by the Server class fit FL loop)
        # (the original FedOpt didn't need of this step in configure_fit)
        # I think it's clearer to keep track of the current arrays explicitly like this
        self.current_arrays = {k: array.numpy() for k, array in arrays.items()}
        return super().configure_train(server_round, arrays, config, grid)
