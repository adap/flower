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


from logging import ERROR
from typing import Callable, Dict, Optional, cast

import ray

from flwr import common
from flwr.client import Client, ClientLike, ClientState, InMemoryClientState, InFileSystemVirtualClientState, to_client
from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.simulation import VirtualClientTemplate

ClientFn = VirtualClientTemplate  # Callable[[str], ClientLike]


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self, client_template: ClientFn, cid: str, resources: Dict[str, float]
    ):
        super().__init__(cid)
        self.client_fn = client_template
        self.resources = resources
        self.state: ClientState = None

        self._prepare_client_state()

        # if self.
    def _prepare_client_state(self):
        client_state = self.client_fn._get_state()
        if isinstance(client_state, InMemoryClientState):
            # the client uses in-memory state, all good! this ClientProxy will
            # record the state once the client completes its task (e.g. fit())
            # and before deleting the client object.
            self.state = InMemoryClientState()

        elif isinstance(client_state, InFileSystemVirtualClientState):
            self.state = client_state
            # replace client's internal state with InMemoryClientState and process all
            # the read/write from/to the file system with the ClientProxy (this object)
            self.client_fn.client.state = InMemoryClientState()
        elif client_state is None:
            # stateless clients
            pass
        else:
            mssg = f"Clients with state {type(client_state)} are not supported for "\
                    "simulation. Please consider using InMemoryClientState or, if you "\
                    "need to save the state to disk, use InFileSystemVirtualClientState."
            raise NotImplementedError(mssg)

    def _fetch_proxy_state(self):
        """Load state before passing it to a virtual client."""

    def _update_proxy_state(self, client_state: ClientState):
        """Update persistent state for virtual client."""
        self.state = client_state

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        future_get_properties_res = launch_and_get_properties.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
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
        future_paramseters_res = launch_and_get_parameters.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
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
        future_fit_res = launch_and_fit.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
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
        future_evaluate_res = launch_and_evaluate.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
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
    client_fn: ClientFn, cid: str, get_properties_ins: common.GetPropertiesIns
) -> common.GetPropertiesRes:
    """Execute get_properties remotely."""
    client: Client = _create_client(client_fn, cid)
    res = maybe_call_get_properties(
        client=client,
        get_properties_ins=get_properties_ins,
    )
    client_state = client.state.fetch()
    return res, client_state


@ray.remote
def launch_and_get_parameters(
    client_fn: ClientFn, cid: str, get_parameters_ins: common.GetParametersIns
) -> common.GetParametersRes:
    """Execute get_parameters remotely."""
    client: Client = _create_client(client_fn, cid)
    res = maybe_call_get_parameters(
        client=client,
        get_parameters_ins=get_parameters_ins,
    )
    client_state = client.state.fetch()
    return res, client_state


@ray.remote
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Execute fit remotely."""
    client: Client = _create_client(client_fn, cid)
    res = maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )
    client_state = client.state.fetch()
    return res, client_state


@ray.remote
def launch_and_evaluate(
    client_fn: ClientFn, cid: str, evaluate_ins: common.EvaluateIns
) -> common.EvaluateRes:
    """Execute evaluate remotely."""
    client: Client = _create_client(client_fn, cid)
    res = maybe_call_evaluate(
        client=client,
        evaluate_ins=evaluate_ins,
    )
    client_state = client.state.fetch()
    return res, client_state


def _create_client(client_fn: ClientFn, cid: str, client_state: ClientState) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid)
    client = to_client(client_like=client_like)
    # set client state
    client.state = client_state
    return client
