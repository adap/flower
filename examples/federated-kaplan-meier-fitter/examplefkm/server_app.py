# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Strategy that supports many univariate fitters from lifelines library."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    Context,
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from lifelines import KaplanMeierFitter


class EventTimeFitterStrategy(Strategy):
    """Federated strategy to aggregate the data that consist of events and times.

    It works with the following uni-variate fitters from the lifelines library:
    AalenJohansenFitter, GeneralizedGammaFitter, KaplanMeierFitter, LogLogisticFitter,
    SplineFitter, WeibullFitter. Note that each of them might require slightly different
    initialization but constructed fitter object are required to be passed.

    This strategy recreates the event and time data based on the data received from the
    nodes.

    Parameters
    ----------
    min_num_clients: int
        pass
    fitter: Any
        uni-variate fitter from lifelines library that works with event, time data e.g.
        KaplanMeierFitter
    """

    def __init__(self, min_num_clients: int, fitter: Any):
        # Fitter can be access after the federated training ends
        self._min_num_clients = min_num_clients
        self.fitter = fitter

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the fit method."""
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self._min_num_clients,
            min_num_clients=self._min_num_clients,
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Merge data and perform the fitting of the fitter from lifelines library.

        Assume just a single federated learning round. Assume the data comes as a list
        with two elements of 1-dim numpy arrays of events and times.
        """
        remote_data = [
            (parameters_to_ndarrays(fit_res.parameters)) for _, fit_res in results
        ]

        combined_times = remote_data[0][0]
        combined_events = remote_data[0][1]

        for te_list in remote_data[1:]:
            combined_times = np.concatenate((combined_times, te_list[0]))
            combined_events = np.concatenate((combined_events, te_list[1]))

        args_sorted = np.argsort(combined_times)
        sorted_times = combined_times[args_sorted]
        sorted_events = combined_events[args_sorted]
        self.fitter.fit(sorted_times, sorted_events)
        print("Survival function:")
        print(self.fitter.survival_function_)
        print("Mean survival time:")
        print(self.fitter.median_survival_time_)
        return None, {}

    # The methods below return None or empty results.
    # They need to be implemented to since the methods are abstract in the parent class
    def initialize_parameters(
        self, client_manager: Optional[ClientManager] = None
    ) -> Optional[Parameters]:
        """No parameter initialization is needed."""
        return None

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """No centralized evaluation."""
        return None

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """No federated evaluation."""
        return None, {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """No federated evaluation."""
        return []


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Define the strategy
    fitter = KaplanMeierFitter()  # You can choose other method that work on E, T data
    min_num_clients = context.run_config["min-num-clients"]
    strategy = EventTimeFitterStrategy(min_num_clients=min_num_clients, fitter=fitter)

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
