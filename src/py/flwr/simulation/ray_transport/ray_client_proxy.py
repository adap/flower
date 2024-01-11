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
from typing import Optional

from flwr import common
from flwr.client import ClientFn
from flwr.client.node_state import NodeState
from flwr.common import serde
from flwr.common.logger import log
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.ray_transport.ray_actor import VirtualClientEngineActorPool


class RayActorClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self, client_fn: ClientFn, cid: str, actor_pool: VirtualClientEngineActorPool
    ):
        super().__init__(cid)
        self.client_fn = client_fn
        self.actor_pool = actor_pool
        self.proxy_state = NodeState()

    def _submit_taskins(self, task_ins: TaskIns, timeout: Optional[float]) -> TaskRes:
        # The VCE is not exposed to TaskIns, it won't handle multilple runs
        # For the time being, fixing run_id is a small compromise
        # This will be one of the first points to address integrating VCE + DriverAPI
        run_id = 0

        # Register state
        self.proxy_state.register_runstate(run_id=run_id)

        # Retrieve state
        state = self.proxy_state.retrieve_runstate(run_id=run_id)

        try:
            self.actor_pool.submit_task_ins(
                lambda a, c_fn, t_ins, cid, state: a.run.remote(
                    c_fn, t_ins, cid, state
                ),
                (self.client_fn, task_ins, self.cid, state),
            )
            task_res, updated_state = self.actor_pool.get_client_result(
                self.cid, timeout
            )

            # Update state
            self.proxy_state.update_runstate(run_id=run_id, run_state=updated_state)

        except Exception as ex:
            if self.actor_pool.num_actors == 0:
                # At this point we want to stop the simulation.
                # since no more client runs will be executed
                log(ERROR, "ActorPool is empty!!!")
            log(ERROR, traceback.format_exc())
            log(ERROR, ex)
            raise ex

        return task_res

    def _submit_server_message_to_pool(
        self, server_msg: ServerMessage, timeout
    ) -> ClientMessage:
        task_ins = TaskIns(
            task_id="",
            group_id="",
            run_id=0,
            task=Task(ancestry=[], legacy_server_message=server_msg),
        )

        # Submit
        task_res = self._submit_taskins(task_ins, timeout)

        # To client message
        return serde.client_message_from_proto(task_res.task.legacy_client_message)

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        ins_proto = serde.get_properties_ins_to_proto(ins)
        server_msg = ServerMessage(get_properties_ins=ins_proto)

        # Submit (block until completed)
        client_msg = self._submit_server_message_to_pool(server_msg, timeout)

        # Return as legacy type
        return client_msg.get_properties_res

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        ins_proto = serde.get_parameters_ins_to_proto(ins)
        server_msg = ServerMessage(get_parameters_ins=ins_proto)

        # Submit (block until completed)
        client_msg = self._submit_server_message_to_pool(server_msg, timeout)

        # Return as legacy type
        return client_msg.get_parameters_res

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        ins_proto = serde.fit_ins_to_proto(ins)
        server_msg = ServerMessage(fit_ins=ins_proto)

        # Submit (block until completed)
        client_msg = self._submit_server_message_to_pool(server_msg, timeout)

        # Return as legacy type
        return client_msg.fit_res

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        ins_proto = serde.evaluate_ins_to_proto(ins)
        server_msg = ServerMessage(evaluate_ins=ins_proto)

        # Submit (block until completed)
        client_msg = self._submit_server_message_to_pool(server_msg, timeout)

        # Return as legacy type
        return client_msg.evaluate_res

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)
