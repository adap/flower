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
"""FedAdagrad [Reddi et al., 2020] strategy.

Adaptive Federated Optimization using Adagrad.

Paper: arxiv.org/abs/2003.00295
"""

from collections import OrderedDict
from collections.abc import Iterable
from typing import Optional

import numpy as np

from flwr.common import Array, ArrayRecord, Message, MetricRecord

from .fedopt import FedOpt


# pylint: disable=line-too-long
class FedAdagrad(FedOpt):
    """FedAdagrad strategy - Adaptive Federated Optimization using Adagrad.

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
    tau : float, optional
        Controls the algorithm's degree of adaptability. Defaults to 1e-3.
    """

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        aggregated_arrayrecord, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrayrecord is None:
            return aggregated_arrayrecord, aggregated_metrics

        aggregated_ndarrays = {
            k: array.numpy() for k, array in aggregated_arrayrecord.items()
        }

        # Adagrad
        delta_t = {
            k: x - y
            for (k, x), (_, y) in zip(
                aggregated_ndarrays.items(), self.current_arrays.items()
            )
        }

        # m_t
        if not self.m_t:
            self.m_t = {k: np.zeros_like(v) for k, v in aggregated_ndarrays.items()}
        self.m_t = {
            k: self.beta_1 * v + (1 - self.beta_1) * delta_t[k]
            for k, v in self.m_t.items()
        }

        # v_t
        if not self.v_t:
            self.v_t = {k: np.zeros_like(v) for k, v in aggregated_ndarrays.items()}
        self.v_t = {k: v + (delta_t[k] ** 2) for k, v in self.v_t.items()}

        new_arrays = {
            k: x + self.eta * self.m_t[k] / (np.sqrt(self.v_t[k]) + self.tau)
            for k, x in aggregated_ndarrays.items()
        }

        # Update current arrays
        self.current_arrays = new_arrays

        return (
            ArrayRecord(OrderedDict({k: Array(v) for k, v in new_arrays.items()})),
            aggregated_metrics,
        )
