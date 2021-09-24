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


from typing import Callable, Dict, Union, cast

import ray

from flwr import common
from flwr.client import Client, NumPyClient
from flwr.client.numpy_client import NumPyClientWrapper
from flwr.server.client_proxy import ClientProxy

ClientFn = Callable[[str], Client]


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(self, client_fn: ClientFn, cid: str, resources: Dict[str, int]):
        super().__init__(cid)
        self.client_fn = client_fn
        self.resources = resources

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""
        future_paramseters_res = launch_and_get_parameters.options(
            **self.resources
        ).remote(self.client_fn, self.cid)
        res = ray.worker.get(future_paramseters_res)
        return cast(
            common.ParametersRes,
            res,
        )

    def get_properties(self, ins: common.PropertiesIns) -> common.PropertiesRes:
        """Returns client's properties."""
        future_properties_res = launch_and_get_properties.options(
            **self.resources
        ).remote(self.client_fn, self.cid, ins)
        future_properties_res = launch_and_get_properties(**self.resources).remote(
            self.client_fn, self.cid, ins
        )
        res = ray.worker.get(future_properties_res)
        return cast(
            common.PropertiesRes,
            res,
        )

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        future_fit_res = launch_and_fit.options(**self.resources).remote(
            self.client_fn, self.cid, ins
        )
        res = ray.worker.get(future_fit_res)
        return cast(
            common.FitRes,
            res,
        )

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        future_evaluate_res = launch_and_evaluate.options(**self.resources).remote(
            self.client_fn, self.cid, ins
        )
        res = ray.worker.get(future_evaluate_res)
        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        return common.Disconnect(reason="")  # Nothing to do here (yet)


@ray.remote  # type: ignore
def launch_and_get_parameters(client_fn: ClientFn, cid: str) -> common.ParametersRes:
    """Exectue get_parameters remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.get_parameters()


@ray.remote  # type: ignore
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.fit(fit_ins)


@ray.remote  # type: ignore
def launch_and_get_properties(
    client_fn: ClientFn, cid: str, properties_ins: common.PropertiesIns
) -> common.PropertiesRes:
    """Exectue get_properties remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.get_properties(properties_ins)


@ray.remote  # type: ignore
def launch_and_evaluate(
    client_fn: ClientFn, cid: str, evaluate_ins: common.EvaluateIns
) -> common.EvaluateRes:
    """Exectue evaluate remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.evaluate(evaluate_ins)


def _create_client(client_fn: ClientFn, cid: str) -> Client:
    """Create a client instance."""
    client: Union[Client, NumPyClient] = client_fn(cid)
    if isinstance(client, NumPyClient):
        client = NumPyClientWrapper(numpy_client=client)
    return client
