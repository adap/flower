# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Adaptive Federated Optimization (FedOpt) [Reddi et al., 2020] abstract
strategy.

Paper: https://arxiv.org/abs/2003.00295
"""


from typing import Callable, Dict, Optional, Tuple

from flwr.common import (
    MetricsAggregationFn,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
)

from .fedavg import FedAvg


class FedOpt(FedAvg):
    """Configurable FedAdagrad strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.0,
        beta_2: float = 0.0,
        tau: float = 1e-9,
    ) -> None:
        """Federated Optim strategy interface.

        Implementation based on https://arxiv.org/abs/2003.00295v5

        Parameters
        ----------
        fraction_fit (float, optional): Fraction of clients used during
            training. Defaults to 0.1.
        fraction_eval (float, optional): Fraction of clients used during
            validation. Defaults to 0.1.
        min_fit_clients (int, optional): Minimum number of clients used
            during training. Defaults to 2.
        min_eval_clients (int, optional): Minimum number of clients used
            during validation. Defaults to 2.
        min_available_clients (int, optional): Minimum number of total
            clients in the system. Defaults to 2.
        eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
            Function used for validation. Defaults to None.
        on_fit_config_fn (Callable[[int], Dict[str, str]], optional):
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn (Callable[[int], Dict[str, str]], optional):
            Function used to configure validation. Defaults to None.
        accept_failures (bool, optional): Whether or not accept rounds
            containing failures. Defaults to True.
        initial_parameters (Parameters): Initial set of parameters from the server.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        eta (float, optional): Server-side learning rate. Defaults to 1e-1.
        eta_l (float, optional): Client-side learning rate. Defaults to 1e-1.
        beta_1 (float, optional): Momentum parameter. Defaults to 0.0.
        beta_2 (float, optional): Second moment parameter. Defaults to 0.0.
        tau (float, optional): Controls the algorithm's degree of adaptability.
            Defaults to 1e-9.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.current_weights = parameters_to_weights(initial_parameters)
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_t: Optional[Weights] = None
        self.v_t: Optional[Weights] = None

    def __repr__(self) -> str:
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep
