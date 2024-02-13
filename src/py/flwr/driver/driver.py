# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
"""Flower driver service client."""
import time
from typing import Iterable, List, Optional, Tuple

from flwr.common.message import Message
from flwr.common.serde import message_from_taskres, message_to_taskins
from flwr.driver.grpc_driver import DEFAULT_SERVER_ADDRESS_DRIVER, GrpcDriver
from flwr.proto.driver_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    GetNodesRequest,
    PullTaskResRequest,
    PushTaskInsRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611


class Driver:
    """`Driver` class provides an interface to the Driver API.

    Parameters
    ----------
    driver_service_address : Optional[str]
        The IPv4 or IPv6 address of the Driver API server.
        Defaults to `"[::]:9091"`.
    certificates : bytes (default: None)
        Tuple containing root certificate, server certificate, and private key
        to start a secure SSL-enabled server. The tuple is expected to have
        three bytes elements in the following order:
            * CA certificate.
            * server certificate.
            * server private key.
    """

    def __init__(
        self,
        driver_service_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        root_certificates: Optional[bytes] = None,
    ) -> None:
        self.addr = driver_service_address
        self.root_certificates = root_certificates
        self.grpc_driver: Optional[GrpcDriver] = None
        self.run_id: Optional[int] = None
        self.node = Node(node_id=0, anonymous=True)

    def _get_grpc_driver_and_run_id(self) -> Tuple[GrpcDriver, int]:
        # Check if the GrpcDriver is initialized
        if self.grpc_driver is None or self.run_id is None:
            # Connect and create run
            self.grpc_driver = GrpcDriver(
                driver_service_address=self.addr,
                root_certificates=self.root_certificates,
            )
            self.grpc_driver.connect()
            res = self.grpc_driver.create_run(CreateRunRequest())
            self.run_id = res.run_id
        return self.grpc_driver, self.run_id

    def get_node_ids(self) -> List[int]:
        """Get node IDs."""
        grpc_driver, run_id = self._get_grpc_driver_and_run_id()
        # Call GrpcDriver method
        res = grpc_driver.get_nodes(GetNodesRequest(run_id=run_id))
        return [node.node_id for node in res.nodes]

    def push_messages(self, messages: Iterable[Message]) -> List[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node it specifies in `node_id`.

        Parameters
        ----------
        messages : Iterable[Message]
            An iterable of messages to be sent.

        Returns
        -------
        message_ids : List[str]
            A list of IDs for the messages that were sent, which can be used
            to pull responses.
        """
        grpc_driver, run_id = self._get_grpc_driver_and_run_id()
        # Construct TaskIns
        task_ins_list: List[TaskIns] = []
        for msg in messages:
            # Check node_id
            if msg.metadata.node_id == 0:
                raise ValueError("Message has no node_id.")
            # Convert Message to TaskIns
            taskins = message_to_taskins(msg)
            # Set producer
            taskins.task.producer.node_id = self.node.node_id
            taskins.task.producer.anonymous = self.node.anonymous
            # Set run_id
            taskins.run_id = run_id
            # Add to list
            task_ins_list.append(taskins)
        # Call GrpcDriver method
        res = grpc_driver.push_task_ins(PushTaskInsRequest(task_ins_list=task_ins_list))
        return list(res.task_ids)

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect responses or messages from the network
        that correspond to a set of given task identifiers.

        Parameters
        ----------
        message_ids : Iterable[str]
            An iterable of task identifiers for which messages are to be retrieved.

        Returns
        -------
        Iterable[Tuple[Message, int]]
            An iterable of tuples, each containing a `Message` object and the ID of
            the node from which the message was received.
        """
        grpc_driver, _ = self._get_grpc_driver_and_run_id()
        # Pull TaskRes
        res = grpc_driver.pull_task_res(
            PullTaskResRequest(node=self.node, task_ids=message_ids)
        )
        # Convert TaskRes to Message
        msgs = [
            (message_from_taskres(taskres), taskres.task.producer.node_id)
            for taskres in res.task_res_list
        ]
        return msgs

    def send_and_receive(
        self,
        messages: Iterable[Tuple[Message, Iterable[int]]],
        *,
        time_out: Optional[float] = None,
    ) -> Iterable[Tuple[Message, int]]:
        """Push messages to specified node IDs and pull the responses.

        This method sends a list of messages to their target node IDs and then
        waits for the responses. It continues to pull responses until either all
        responses are received or the specified time out duration is exceeded.

        Parameters
        ----------
        messages : Iterable[Tuple[Message, Iterable[int]]]
            An iterable of tuples, each containing a `Message` object and an iterable
            of target node IDs to which the message should be sent.
        time_out : Optional[float], default=None
            The time out duration in seconds. If specified, the method will wait for
            responses for this duration. If None, there is no time limit and the method
            will wait until responses for all messages are received.

        Returns
        -------
        Iterable[Tuple[Message, int]]
            An iterable of tuples, each containing a `Message` object and the ID of
            the node from which the message was received.

        Notes
        -----
        This method uses `push_messages` to send the messages and `pull_messages`
        to collect the responses. If `time_out` is set, the method may not return
        responses for all sent messages.
        """
        # Push messages
        msg_ids = self.push_messages(messages)

        # Pull messages
        end_time = time.time() + time_out
        ret: List[Tuple[Message, int]] = []
        while time.time() < end_time:
            res_msgs = self.pull_messages(msg_ids)
            ret += res_msgs
            if len(ret) == len(msg_ids):
                break
            # Sleep
            time.sleep(3)
        return ret

    def __del__(self) -> None:
        """Disconnect GrpcDriver if connected."""
        # Check if GrpcDriver is initialized
        if self.grpc_driver is None:
            return
        # Disconnect
        self.grpc_driver.disconnect()
