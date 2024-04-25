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

import time
import warnings
from dataclasses import dataclass
from typing import cast

from .record import RecordSet

DEFAULT_TTL = 3600


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
    ttl : float
        Time-to-live for this message in seconds.
    message_type : str
        A string that encodes the action to be executed on
        the receiving end.
    partition_id : Optional[int]
        An identifier that can be used when loading a particular
        data partition for a ClientApp. Making use of this identifier
        is more relevant when conducting simulations.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        run_id: int,
        message_id: str,
        src_node_id: int,
        dst_node_id: int,
        reply_to_message: str,
        group_id: str,
        ttl: float,
        message_type: str,
        partition_id: int | None = None,
    ) -> None:
        var_dict = {
            "_run_id": run_id,
            "_message_id": message_id,
            "_src_node_id": src_node_id,
            "_dst_node_id": dst_node_id,
            "_reply_to_message": reply_to_message,
            "_group_id": group_id,
            "_ttl": ttl,
            "_message_type": message_type,
            "_partition_id": partition_id,
        }
        self.__dict__.update(var_dict)

    @property
    def run_id(self) -> int:
        """An identifier for the current run."""
        return cast(int, self.__dict__["_run_id"])

    @property
    def message_id(self) -> str:
        """An identifier for the current message."""
        return cast(str, self.__dict__["_message_id"])

    @property
    def src_node_id(self) -> int:
        """An identifier for the node sending this message."""
        return cast(int, self.__dict__["_src_node_id"])

    @property
    def reply_to_message(self) -> str:
        """An identifier for the message this message replies to."""
        return cast(str, self.__dict__["_reply_to_message"])

    @property
    def dst_node_id(self) -> int:
        """An identifier for the node receiving this message."""
        return cast(int, self.__dict__["_dst_node_id"])

    @dst_node_id.setter
    def dst_node_id(self, value: int) -> None:
        """Set dst_node_id."""
        self.__dict__["_dst_node_id"] = value

    @property
    def group_id(self) -> str:
        """An identifier for grouping messages."""
        return cast(str, self.__dict__["_group_id"])

    @group_id.setter
    def group_id(self, value: str) -> None:
        """Set group_id."""
        self.__dict__["_group_id"] = value

    @property
    def created_at(self) -> float:
        """Unix timestamp when the message was created."""
        return cast(float, self.__dict__["_created_at"])

    @created_at.setter
    def created_at(self, value: float) -> None:
        """Set creation timestamp for this message."""
        self.__dict__["_created_at"] = value

    @property
    def ttl(self) -> float:
        """Time-to-live for this message."""
        return cast(float, self.__dict__["_ttl"])

    @ttl.setter
    def ttl(self, value: float) -> None:
        """Set ttl."""
        self.__dict__["_ttl"] = value

    @property
    def message_type(self) -> str:
        """A string that encodes the action to be executed on the receiving end."""
        return cast(str, self.__dict__["_message_type"])

    @message_type.setter
    def message_type(self, value: str) -> None:
        """Set message_type."""
        self.__dict__["_message_type"] = value

    @property
    def partition_id(self) -> int | None:
        """An identifier telling which data partition a ClientApp should use."""
        return cast(int, self.__dict__["_partition_id"])

    @partition_id.setter
    def partition_id(self, value: int) -> None:
        """Set partition_id."""
        self.__dict__["_partition_id"] = value


@dataclass
class Error:
    """A dataclass that stores information about an error that occurred.

    Parameters
    ----------
    code : int
        An identifier for the error.
    reason : Optional[str]
        A reason for why the error arose (e.g. an exception stack-trace)
    """

    _code: int
    _reason: str | None = None

    def __init__(self, code: int, reason: str | None = None) -> None:
        self._code = code
        self._reason = reason

    @property
    def code(self) -> int:
        """Error code."""
        return self._code

    @property
    def reason(self) -> str | None:
        """Reason reported about the error."""
        return self._reason


@dataclass
class Message:
    """State of your application from the viewpoint of the entity using it.

    Parameters
    ----------
    metadata : Metadata
        A dataclass including information about the message to be executed.
    content : Optional[RecordSet]
        Holds records either sent by another entity (e.g. sent by the server-side
        logic to a client, or vice-versa) or that will be sent to it.
    error : Optional[Error]
        A dataclass that captures information about an error that took place
        when processing another message.
    """

    _metadata: Metadata
    _content: RecordSet | None = None
    _error: Error | None = None

    def __init__(
        self,
        metadata: Metadata,
        content: RecordSet | None = None,
        error: Error | None = None,
    ) -> None:
        self._metadata = metadata

        # Set message creation timestamp
        self._metadata.created_at = time.time()

        if not (content is None) ^ (error is None):
            raise ValueError("Either `content` or `error` must be set, but not both.")

        self._content = content
        self._error = error

    @property
    def metadata(self) -> Metadata:
        """A dataclass including information about the message to be executed."""
        return self._metadata

    @property
    def content(self) -> RecordSet:
        """The content of this message."""
        if self._content is None:
            raise ValueError(
                "Message content is None. Use <message>.has_content() "
                "to check if a message has content."
            )
        return self._content

    @content.setter
    def content(self, value: RecordSet) -> None:
        """Set content."""
        if self._error is None:
            self._content = value
        else:
            raise ValueError("A message with an error set cannot have content.")

    @property
    def error(self) -> Error:
        """Error captured by this message."""
        if self._error is None:
            raise ValueError(
                "Message error is None. Use <message>.has_error() "
                "to check first if a message carries an error."
            )
        return self._error

    @error.setter
    def error(self, value: Error) -> None:
        """Set error."""
        if self.has_content():
            raise ValueError("A message with content set cannot carry an error.")
        self._error = value

    def has_content(self) -> bool:
        """Return True if message has content, else False."""
        return self._content is not None

    def has_error(self) -> bool:
        """Return True if message has an error, else False."""
        return self._error is not None

    def _create_reply_metadata(self, ttl: float) -> Metadata:
        """Construct metadata for a reply message."""
        return Metadata(
            run_id=self.metadata.run_id,
            message_id="",
            src_node_id=self.metadata.dst_node_id,
            dst_node_id=self.metadata.src_node_id,
            reply_to_message=self.metadata.message_id,
            group_id=self.metadata.group_id,
            ttl=ttl,
            message_type=self.metadata.message_type,
            partition_id=self.metadata.partition_id,
        )

    def create_error_reply(self, error: Error, ttl: float | None = None) -> Message:
        """Construct a reply message indicating an error happened.

        Parameters
        ----------
        error : Error
            The error that was encountered.
        ttl : Optional[float] (default: None)
            Time-to-live for this message in seconds. If unset, it will be set based
            on the remaining time for the received message before it expires. This
            follows the equation:

            ttl = msg.meta.ttl - (reply.meta.created_at - msg.meta.created_at)
        """
        if ttl:
            warnings.warn(
                "A custom TTL was set, but note that the SuperLink does not enforce "
                "the TTL yet. The SuperLink will start enforcing the TTL in a future "
                "version of Flower.",
                stacklevel=2,
            )
        # If no TTL passed, use default for message creation (will update after
        # message creation)
        ttl_ = DEFAULT_TTL if ttl is None else ttl
        # Create reply with error
        message = Message(metadata=self._create_reply_metadata(ttl_), error=error)

        if ttl is None:
            # Set TTL equal to the remaining time for the received message to expire
            ttl = self.metadata.ttl - (
                message.metadata.created_at - self.metadata.created_at
            )
            message.metadata.ttl = ttl

        return message

    def create_reply(self, content: RecordSet, ttl: float | None = None) -> Message:
        """Create a reply to this message with specified content and TTL.

        The method generates a new `Message` as a reply to this message.
        It inherits 'run_id', 'src_node_id', 'dst_node_id', and 'message_type' from
        this message and sets 'reply_to_message' to the ID of this message.

        Parameters
        ----------
        content : RecordSet
            The content for the reply message.
        ttl : Optional[float] (default: None)
            Time-to-live for this message in seconds. If unset, it will be set based
            on the remaining time for the received message before it expires. This
            follows the equation:

            ttl = msg.meta.ttl - (reply.meta.created_at - msg.meta.created_at)

        Returns
        -------
        Message
            A new `Message` instance representing the reply.
        """
        if ttl:
            warnings.warn(
                "A custom TTL was set, but note that the SuperLink does not enforce "
                "the TTL yet. The SuperLink will start enforcing the TTL in a future "
                "version of Flower.",
                stacklevel=2,
            )
        # If no TTL passed, use default for message creation (will update after
        # message creation)
        ttl_ = DEFAULT_TTL if ttl is None else ttl

        message = Message(
            metadata=self._create_reply_metadata(ttl_),
            content=content,
        )

        if ttl is None:
            # Set TTL equal to the remaining time for the received message to expire
            ttl = self.metadata.ttl - (
                message.metadata.created_at - self.metadata.created_at
            )
            message.metadata.ttl = ttl

        return message
