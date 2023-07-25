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


from copy import deepcopy
from logging import ERROR
from typing import Any, Callable, Dict, Optional, cast

import ray

from flwr import common
from flwr.client import (
    Client,
    ClientLike,
    to_client,
)
from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client import ClientState
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.virtual_client_state_manager import VirtualClientStateManager


ClientFn = Callable[[str], ClientLike]


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self,
        client_fn: ClientFn,
        state_manager: VirtualClientStateManager,
        cid: str,
        resources: Dict[str, float],
    ):
        super().__init__(cid)
        self.client_fn = client_fn
        self.resources = resources
        self.state_manager = state_manager

        self._register_client_state()


    def _register_client_state(self):
        """Register client to track state"""

        # At this point we don't know if the client is stateful
        # but it doesn't matter much. Tracking it in the state manager
        # adds virtually zero overhead
        self.state_manager.track_state(client_key=self.cid)

    def _fetch_proxy_state(self):
        """Load state before passing it to a virtual client."""
        return self.state_manager.get_client_state(self.cid)

    def _update_proxy_state(self, client_state: Dict[str, Any]):
        """Update persistent state for virtual client."""
        self.state_manager.update_client_state(self.cid, client_state)

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""

        # prepare state for the client to be spawned
        client_state = self._fetch_proxy_state()
        future_get_properties_res = launch_and_get_properties.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins, client_state)
        try:
            res, state = ray.get(future_get_properties_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
        # update state
        self._update_proxy_state(state)
        return cast(
            common.GetPropertiesRes,
            res,
        )

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""

        # prepare state for the client to be spawned
        client_state = self._fetch_proxy_state()
        future_paramseters_res = launch_and_get_parameters.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins, client_state)
        try:
            res, state = ray.get(future_paramseters_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
        # update state
        self._update_proxy_state(state)
        return cast(
            common.GetParametersRes,
            res,
        )

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""

        # prepare state for the client to be spawned
        client_state = self._fetch_proxy_state()
        future_fit_res = launch_and_fit.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins, client_state)
        try:
            res, state = ray.get(future_fit_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
        # update state
        self._update_proxy_state(state)
        return cast(
            common.FitRes,
            res,
        )

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""

        # prepare state for the client to be spawned
        client_state = self._fetch_proxy_state()
        future_evaluate_res = launch_and_evaluate.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins, client_state)
        try:
            res, state = ray.get(future_evaluate_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
        # update state
        self._update_proxy_state(state)
        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)


@ray.remote
def launch_and_get_properties(
    client_fn: ClientFn,
    cid: str,
    get_properties_ins: common.GetPropertiesIns,
    client_state: Dict[str, Any],
) -> common.GetPropertiesRes:
    """Execute get_properties remotely."""
    client: Client = _create_client(client_fn, cid, client_state)
    res = maybe_call_get_properties(
        client=client,
        get_properties_ins=get_properties_ins,
    )
    state = client.numpy_client.fetch_state() if is_stateful(client) else None
    return res, state


@ray.remote
def launch_and_get_parameters(
    client_fn: ClientFn,
    cid: str,
    get_parameters_ins: common.GetParametersIns,
    client_state: Dict[str, Any],
) -> common.GetParametersRes:
    """Execute get_parameters remotely."""
    client: Client = _create_client(client_fn, cid, client_state)
    res = maybe_call_get_parameters(
        client=client,
        get_parameters_ins=get_parameters_ins,
    )
    state = client.numpy_client.fetch_state() if is_stateful(client) else None
    return res, state

@ray.remote
def launch_and_fit(
    client_fn: ClientFn,
    cid: str,
    fit_ins: common.FitIns,
    client_state: Dict[str, Any],
) -> common.FitRes:
    """Execute fit remotely."""
    client: Client = _create_client(client_fn, cid, client_state)
    res = maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )

    state = client.numpy_client.fetch_state() if is_stateful(client) else None
    return res, state


@ray.remote
def launch_and_evaluate(
    client_fn: ClientFn,
    cid: str,
    evaluate_ins: common.EvaluateIns,
    client_state: Dict[str, Any],
) -> common.EvaluateRes:
    """Execute evaluate remotely."""
    client: Client = _create_client(client_fn, cid, client_state)
    res = maybe_call_evaluate(
        client=client,
        evaluate_ins=evaluate_ins,
    )
    state = client.numpy_client.fetch_state() if is_stateful(client) else None
    return res, state


def _create_client(client_fn: ClientFn, cid: str, client_state: Dict[str, Any]) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid)
    client = to_client(client_like=client_like)

    # set client state
    # TODO: we'll need to ensure this is a NumPyClientWrapper type -- how?
    if is_stateful(client):
        # TODO: if we expose all the ClientState relevant methods to the wrapper, we won't need to do this via numpy_client access
        client.numpy_client.update_state(client_state)
    return client


def is_stateful(client: Client) -> bool:
    return isinstance(client, ClientState)
