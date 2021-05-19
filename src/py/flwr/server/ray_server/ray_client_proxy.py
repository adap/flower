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
"""Ray-based Flower ClientProxy implementation."""


from flwr import common
from flwr.common import serde
from flwr.server.client_proxy import ClientProxy


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self,
        cid: str,
    ):
        super().__init__(cid)

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""

        # TODO Ray: get parameters of one client
        parameters_res: common.ParametersRes = None

        return parameters_res

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Refine the provided weights using the locally held dataset."""

        # TODO Ray: create client, perform work, return result
        fit_res: common.FitRes = None

        return fit_res

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

        # TODO Ray: create client, evaluate parameters, return results
        evaluate_res: common.EvaluateRes = None

        return evaluate_res

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""

        # TODO Ray: unregister ClientProxy from ClientManager, re-register later
        disconnect: common.Disconnect = None

        return disconnect
