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


from typing import Dict, Union

import ray

import flwr
from flwr import common
from flwr.client import Client, NumPyClient
from flwr.server.client_proxy import ClientProxy


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(self, client_fn, cid: str, resources: Dict[str, int]):
        super().__init__(cid)
        self.client_fn = client_fn
        self.resources = resources

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""
        future_paramseters_res = launch_and_get_params.options(**self.resources).remote(
            self.client_fn, self.cid
        )
        return ray.get(future_paramseters_res)

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Refine the provided model parameters using the locally held
        dataset."""
        future_fit_res = launch_and_fit.options(**self.resources).remote(
            self.client_fn, self.cid, ins
        )
        return ray.get(future_fit_res)

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate the provided model parameters using the locally held
        dataset."""
        future_evaluate_res = launch_and_evaluate.options(**self.resources).remote(
            self.client_fn, self.cid, ins
        )
        return ray.get(future_evaluate_res)

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        return common.Disconnect(reason="")  # Nothing to do here (yet)


@ray.remote
def launch_and_get_params(client_fn, cid) -> common.ParametersRes:
    client: Client = _create_client(client_fn, cid)
    return client.get_parameters()


@ray.remote
def launch_and_fit(client_fn, cid, fit_ins) -> common.FitRes:
    client: Client = _create_client(client_fn, cid)
    return client.fit(fit_ins)


@ray.remote
def launch_and_evaluate(client_fn, cid, evaluate_ins) -> common.EvaluateRes:
    client: Client = _create_client(client_fn, cid)
    return client.evaluate(evaluate_ins)


def _create_client(client_fn, cid: str) -> Client:
    client: Union[Client, NumPyClient] = client_fn(cid)
    if isinstance(client, NumPyClient):
        client = flwr.client.numpy_client.NumPyClientWrapper(numpy_client=client)
    return client
