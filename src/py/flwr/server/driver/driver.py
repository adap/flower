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
import warnings
from typing import Iterable, List, Optional, Tuple

from flwr.common import DEFAULT_TTL, Message, Metadata, RecordSet
from flwr.common.serde import message_from_taskres, message_to_taskins
from flwr.proto.driver_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    GetNodesRequest,
    PullTaskResRequest,
    PushTaskInsRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .grpc_driver import DEFAULT_SERVER_ADDRESS_DRIVER, GrpcDriver


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

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            message.metadata.run_id == self.run_id
            and message.metadata.src_node_id == self.node.node_id
            and message.metadata.message_id == ""
            and message.metadata.reply_to_message == ""
            and message.metadata.ttl > 0
        ):
            raise ValueError(f"Invalid message: {message}")

    def create_message(  # pylint: disable=too-many-arguments
        self,
        content: RecordSet,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: Optional[float] = None,
    ) -> Message:
        """Create a new message with specified parameters.

        This method constructs a new `Message` with given content and metadata.
        The `run_id` and `src_node_id` will be set automatically.

        Parameters
        ----------
        content : RecordSet
            The content for the new message. This holds records that are to be sent
            to the destination node.
        message_type : str
            The type of the message, defining the action to be executed on
            the receiving end.
        dst_node_id : int
            The ID of the destination node to which the message is being sent.
        group_id : str
            The ID of the group to which this message is associated. In some settings,
            this is used as the FL round.
        ttl : Optional[float] (default: None)
            Time-to-live for the round trip of this message, i.e., the time from sending
            this message to receiving a reply. It specifies in seconds the duration for
            which the message and its potential reply are considered valid. If unset,
            the default TTL (i.e., `common.DEFAULT_TTL`) will be used.

        Returns
        -------
        message : Message
            A new `Message` instance with the specified content and metadata.
        """
        _, run_id = self._get_grpc_driver_and_run_id()
        if ttl:
            warnings.warn(
                "A custom TTL was set, but note that the SuperLink does not enforce "
                "the TTL yet. The SuperLink will start enforcing the TTL in a future "
                "version of Flower.",
                stacklevel=2,
            )

        ttl_ = DEFAULT_TTL if ttl is None else ttl
        metadata = Metadata(
            run_id=run_id,
            message_id="",  # Will be set by the server
            src_node_id=self.node.node_id,
            dst_node_id=dst_node_id,
            reply_to_message="",
            group_id=group_id,
            ttl=ttl_,
            message_type=message_type,
        )
        return Message(metadata=metadata, content=content)

    def get_node_ids(self) -> List[int]:
        """Get node IDs."""
        grpc_driver, run_id = self._get_grpc_driver_and_run_id()
        # Call GrpcDriver method
        res = grpc_driver.get_nodes(GetNodesRequest(run_id=run_id))
        return [node.node_id for node in res.nodes]

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.

        Parameters
        ----------
        messages : Iterable[Message]
            An iterable of messages to be sent.

        Returns
        -------
        message_ids : Iterable[str]
            An iterable of IDs for the messages that were sent, which can be used
            to pull replies.
        """
        grpc_driver, _ = self._get_grpc_driver_and_run_id()
        # Construct TaskIns
        task_ins_list: List[TaskIns] = []
        for msg in messages:
            # Check message
            self._check_message(msg)
            # Convert Message to TaskIns
            taskins = message_to_taskins(msg)
            # Add to list
            task_ins_list.append(taskins)
        # Call GrpcDriver method
        res = grpc_driver.push_task_ins(PushTaskInsRequest(task_ins_list=task_ins_list))
        return list(res.task_ids)

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink
        that correspond to a set of given message IDs.

        Parameters
        ----------
        message_ids : Iterable[str]
            An iterable of message IDs for which reply messages are to be retrieved.

        Returns
        -------
        messages : Iterable[Message]
            An iterable of messages received.
        """
        grpc_driver, _ = self._get_grpc_driver_and_run_id()
        # Pull TaskRes
        res = grpc_driver.pull_task_res(
            PullTaskResRequest(node=self.node, task_ids=message_ids)
        )
        # Convert TaskRes to Message
        msgs = [message_from_taskres(taskres) for taskres in res.task_res_list]
        return msgs

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: Optional[float] = None,
    ) -> Iterable[Message]:
        """Push messages to specified node IDs and pull the reply messages.

        This method sends a list of messages to their destination node IDs and then
        waits for the replies. It continues to pull replies until either all
        replies are received or the specified timeout duration is exceeded.

        Parameters
        ----------
        messages : Iterable[Message]
            An iterable of messages to be sent.
        timeout : Optional[float] (default: None)
            The timeout duration in seconds. If specified, the method will wait for
            replies for this duration. If `None`, there is no time limit and the method
            will wait until replies for all messages are received.

        Returns
        -------
        replies : Iterable[Message]
            An iterable of reply messages received from the SuperLink.

        Notes
        -----
        This method uses `push_messages` to send the messages and `pull_messages`
        to collect the replies. If `timeout` is set, the method may not return
        replies for all sent messages. A message remains valid until its TTL,
        which is not affected by `timeout`.
        """
        # Push messages
        msg_ids = set(self.push_messages(messages))

        # Pull messages
        end_time = time.time() + (timeout if timeout is not None else 0.0)
        ret: List[Message] = []
        while timeout is None or time.time() < end_time:
            res_msgs = self.pull_messages(msg_ids)
            ret.extend(res_msgs)
            msg_ids.difference_update(
                {msg.metadata.reply_to_message for msg in res_msgs}
            )
            if len(msg_ids) == 0:
                break
            # Sleep
            time.sleep(3)
        return ret

    def close(self) -> None:
        """Disconnect from the SuperLink if connected."""
        # Check if GrpcDriver is initialized
        if self.grpc_driver is None:
            return
        # Disconnect
        self.grpc_driver.disconnect()
