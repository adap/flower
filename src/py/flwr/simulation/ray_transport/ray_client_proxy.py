# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


import traceback
from logging import ERROR
from typing import Dict, Optional, cast

import ray

from flwr import common
from flwr.client import Client, ClientFn, ClientLike, to_client
from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.ray_transport.ray_actor import (
    ClientRes,
    JobFn,
    VirtualClientEngineActorPool,
)


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(self, client_fn: ClientFn, cid: str, resources: Dict[str, float]):
        super().__init__(cid)
        self.client_fn = client_fn
        self.resources = resources

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        future_get_properties_res = launch_and_get_properties.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
        try:
            res = ray.get(future_get_properties_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
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
            res = ray.get(future_paramseters_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
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
            res = ray.get(future_fit_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
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
            res = ray.get(future_evaluate_res, timeout=timeout)
        except Exception as ex:
            log(ERROR, ex)
            raise ex
        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)


class RayActorClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self, client_fn: ClientFn, cid: str, actor_pool: VirtualClientEngineActorPool
    ):
        super().__init__(cid)
        self.client_fn = client_fn
        self.actor_pool = actor_pool

    def _submit_job(self, job_fn: JobFn, timeout: Optional[float]) -> ClientRes:
        try:
            self.actor_pool.submit_client_job(
                lambda a, c_fn, j_fn, cid: a.run.remote(c_fn, j_fn, cid),
                (self.client_fn, job_fn, self.cid),
            )
            res = self.actor_pool.get_client_result(self.cid, timeout)

        except Exception as ex:
            if self.actor_pool.num_actors == 0:
                # At this point we want to stop the simulation.
                # since no more client workloads will be executed
                log(ERROR, "ActorPool is empty!!!")
            log(ERROR, traceback.format_exc())
            log(ERROR, ex)
            raise ex

        return res

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""

        def get_properties(client: Client) -> common.GetPropertiesRes:
            return maybe_call_get_properties(
                client=client,
                get_properties_ins=ins,
            )

        res = self._submit_job(get_properties, timeout)

        return cast(
            common.GetPropertiesRes,
            res,
        )

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""

        def get_parameters(client: Client) -> common.GetParametersRes:
            return maybe_call_get_parameters(
                client=client,
                get_parameters_ins=ins,
            )

        res = self._submit_job(get_parameters, timeout)

        return cast(
            common.GetParametersRes,
            res,
        )

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""

        def fit(client: Client) -> common.FitRes:
            return maybe_call_fit(
                client=client,
                fit_ins=ins,
            )

        res = self._submit_job(fit, timeout)

        return cast(
            common.FitRes,
            res,
        )

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""

        def evaluate(client: Client) -> common.EvaluateRes:
            return maybe_call_evaluate(
                client=client,
                evaluate_ins=ins,
            )

        res = self._submit_job(evaluate, timeout)

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
    """Exectue get_properties remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_get_properties(
        client=client,
        get_properties_ins=get_properties_ins,
    )


@ray.remote
def launch_and_get_parameters(
    client_fn: ClientFn, cid: str, get_parameters_ins: common.GetParametersIns
) -> common.GetParametersRes:
    """Exectue get_parameters remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_get_parameters(
        client=client,
        get_parameters_ins=get_parameters_ins,
    )


@ray.remote
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )


@ray.remote
def launch_and_evaluate(
    client_fn: ClientFn, cid: str, evaluate_ins: common.EvaluateIns
) -> common.EvaluateRes:
    """Exectue evaluate remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_evaluate(
        client=client,
        evaluate_ins=evaluate_ins,
    )


def _create_client(client_fn: ClientFn, cid: str) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid)
    return to_client(client_like=client_like)
