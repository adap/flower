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
"""Message."""


from __future__ import annotations

from dataclasses import dataclass

from .record import RecordSet


@dataclass
class Metadata:  # pylint: disable=too-many-instance-attributes
    """A dataclass holding metadata associated with the current message.

    Parameters
    ----------
    run_id : int
        An identifier for the current run.
    message_id : str
        An identifier for the current message.
    src_node_id : int
        An identifier for the node sending this message.
    dst_node_id : int
        An identifier for the node receiving this message.
    reply_to_message : str
        An identifier for the message this message replies to.
    group_id : str
        An identifier for grouping messages. In some settings,
        this is used as the FL round.
    ttl : str
        Time-to-live for this message.
    message_type : str
        A string that encodes the action to be executed on
        the receiving end.
    """

    _run_id: int
    _message_id: str
    _src_node_id: int
    _dst_node_id: int
    _reply_to_message: str
    _group_id: str
    _ttl: str
    _message_type: str

    def __init__(  # pylint: disable=too-many-arguments
        self,
        run_id: int,
        message_id: str,
        src_node_id: int,
        dst_node_id: int,
        reply_to_message: str,
        group_id: str,
        ttl: str,
        message_type: str,
    ) -> None:
        self._run_id = run_id
        self._message_id = message_id
        self._src_node_id = src_node_id
        self._dst_node_id = dst_node_id
        self._reply_to_message = reply_to_message
        self._group_id = group_id
        self._ttl = ttl
        self._message_type = message_type

    @property
    def run_id(self) -> int:
        """An identifier for the current run."""
        return self._run_id

    @property
    def message_id(self) -> str:
        """An identifier for the current message."""
        return self._message_id

    @property
    def src_node_id(self) -> int:
        """An identifier for the node sending this message."""
        return self._src_node_id

    @property
    def reply_to_message(self) -> str:
        """An identifier for the message this message replies to."""
        return self._reply_to_message

    @property
    def dst_node_id(self) -> int:
        """An identifier for the node receiving this message."""
        return self._dst_node_id

    @dst_node_id.setter
    def dst_node_id(self, value: int) -> None:
        """Set dst_node_id."""
        self._dst_node_id = value

    @property
    def group_id(self) -> str:
        """An identifier for grouping messages."""
        return self._group_id

    @group_id.setter
    def group_id(self, value: str) -> None:
        """Set group_id."""
        self._group_id = value

    @property
    def ttl(self) -> str:
        """Time-to-live for this message."""
        return self._ttl

    @ttl.setter
    def ttl(self, value: str) -> None:
        """Set ttl."""
        self._ttl = value

    @property
    def message_type(self) -> str:
        """A string that encodes the action to be executed on the receiving end."""
        return self._message_type

    @message_type.setter
    def message_type(self, value: str) -> None:
        """Set message_type."""
        self._message_type = value


@dataclass
class Message:
    """State of your application from the viewpoint of the entity using it.

    Parameters
    ----------
    metadata : Metadata
        A dataclass including information about the message to be executed.
    content : RecordSet
        Holds records either sent by another entity (e.g. sent by the server-side
        logic to a client, or vice-versa) or that will be sent to it.
    """

    _metadata: Metadata
    _content: RecordSet

    def __init__(self, metadata: Metadata, content: RecordSet) -> None:
        self._metadata = metadata
        self._content = content

    @property
    def metadata(self) -> Metadata:
        """A dataclass including information about the message to be executed."""
        return self._metadata

    @property
    def content(self) -> RecordSet:
        """The content of this message."""
        return self._content

    @content.setter
    def content(self, value: RecordSet) -> None:
        """Set content."""
        self._content = value

    def create_reply(self, content: RecordSet, ttl: str) -> Message:
        """Create a reply to this message with specified content and TTL.

        The method generates a new `Message` as a reply to this message.
        It inherits 'run_id', 'src_node_id', 'dst_node_id', and 'message_type' from
        this message and sets 'reply_to_message' to the ID of this message.

        Parameters
        ----------
        content : RecordSet
            The content for the reply message.
        ttl : str
            Time-to-live for this message.

        Returns
        -------
        Message
            A new `Message` instance representing the reply.
        """
        return Message(
            metadata=Metadata(
                run_id=self.metadata.run_id,
                message_id="",
                src_node_id=self.metadata.dst_node_id,
                dst_node_id=self.metadata.src_node_id,
                reply_to_message=self.metadata.message_id,
                group_id=self.metadata.group_id,
                ttl=ttl,
                message_type=self.metadata.message_type,
            ),
            content=content,
        )
