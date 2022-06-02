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
"""Secure Aggregation (SecAgg) Bonawitz et al.

Paper: https://eprint.iacr.org/2017/281.pdf
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.strategy.sec_agg_strategy import SecAggStrategy
from flwr.common.sec_agg.sec_agg_server_logic import sec_agg_fit_round
from flwr.common.secure_aggregation import SecureAggregationFitRound

from .aggregate import aggregate, weighted_loss_avg
from .strategy import Strategy

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""


class SecAggFedAvg(FedAvg, SecureAggregationFitRound):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
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
        initial_parameters: Optional[Parameters] = None,
        sec_agg_param_dict: Dict[str, Scalar] = {}
    ) -> None:

        FedAvg.__init__(self, fraction_fit=fraction_fit,
                        fraction_eval=fraction_eval,
                        min_fit_clients=min_fit_clients,
                        min_eval_clients=min_eval_clients,
                        min_available_clients=min_available_clients,
                        eval_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn,
                        on_evaluate_config_fn=on_evaluate_config_fn,
                        accept_failures=accept_failures,
                        initial_parameters=initial_parameters)
        self.config = sec_agg_param_dict

    def fit_round(self, server, rnd: int):
        return sec_agg_fit_round(self, server, rnd=rnd)


import math
def num_shares_suggest(collusion_fraction, dropout_fraction,
                       num_clients, beta=0.5, sigma=40, eta=30):
    gamma = collusion_fraction
    delta = dropout_fraction
    n = num_clients
    assert 0 < delta < 1 and 0 < gamma < 1
    assert gamma * n / (n - 1) + delta < 1
    assert gamma * n / (n - 1) < beta < 1 - delta
    # c1 = 2 * (beta - (n * gamma / (n - 1))) ** 2
    # c2 = 2 * (n * (1 - delta) / (n - 1) - beta) ** 2
    c1 = -math.log(gamma + delta)
    c2 = 2 * (beta - (n * gamma / (n - 1))) ** 2
    # c = 1. / min(c1, c2)
    c = min(c1, c2)
    k1 = ((sigma + 1) * math.log(2) + math.log(n)) / c1 + 1
    k2 = ((sigma + 1) * math.log(2) + math.log(n)) / c2
    k2_prime = ((sigma + 1) * math.log(2) + math.log(n)) / c1
    k3 = (eta * math.log(2) + math.log(n)) / (2 * (n * (1 - delta) / (n - 1) - beta) ** 2)
    # k = max(
    #     ((sigma + 1) * math.log(2) + math.log(n)) / c + 1,
    #     (eta * math.log(2) + math.log(n)) / (2 * (n * (1 - delta) / (n - 1) - beta) ** 2)
    # )
    return math.ceil(max(k1, k2, k3)), k1, k2, k2_prime, k3


import numpy as np
def my_range(start, stop, step):
    ret = [start]
    while start < stop:
        start += step
        ret.append(start)
    return ret


lst = [(b, num_shares_suggest(0.05, 0.05, 10 ** 3, beta=b, eta=20)) for b in my_range(0.5, 0.7, 0.02)]
k = 1
