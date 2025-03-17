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

from flwr.common.date import now
from logging import WARNING
from typing import Optional, cast, overload, Any
from flwr.common.logger import warn_deprecated_feature

from .constant import MESSAGE_TTL_TOLERANCE
from .logger import log
from .record import RecordSet

DEFAULT_TTL = 43200  # This is 12 hours


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
    created_at : float
        Unix timestamp when the message was created.
    ttl : float
        Time-to-live for this message in seconds.
    message_type : str
        A string that encodes the action to be executed on
        the receiving end.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        run_id: int,
        message_id: str,
        src_node_id: int,
        dst_node_id: int,
        reply_to_message: str,
        group_id: str,
        created_at: float,
        ttl: float,
        message_type: str,
    ) -> None:
        var_dict = {
            "_run_id": run_id,
            "_message_id": message_id,
            "_src_node_id": src_node_id,
            "_dst_node_id": dst_node_id,
            "_reply_to_message": reply_to_message,
            "_group_id": group_id,
            "_created_at": created_at,
            "_ttl": ttl,
            "_message_type": message_type,
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
    def delivered_at(self) -> str:
        """Unix timestamp when the message was delivered."""
        return cast(str, self.__dict__["_delivered_at"])

    @delivered_at.setter
    def delivered_at(self, value: str) -> None:
        """Set delivery timestamp of this message."""
        self.__dict__["_delivered_at"] = value

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

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        view = ", ".join([f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__qualname__}({view})"

    def __eq__(self, other: object) -> bool:
        """Compare two instances of the class."""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.__dict__ == other.__dict__


class Error:
    """A dataclass that stores information about an error that occurred.

    Parameters
    ----------
    code : int
        An identifier for the error.
    reason : Optional[str]
        A reason for why the error arose (e.g. an exception stack-trace)
    """

    def __init__(self, code: int, reason: str | None = None) -> None:
        var_dict = {
            "_code": code,
            "_reason": reason,
        }
        self.__dict__.update(var_dict)

    @property
    def code(self) -> int:
        """Error code."""
        return cast(int, self.__dict__["_code"])

    @property
    def reason(self) -> str | None:
        """Reason reported about the error."""
        return cast(Optional[str], self.__dict__["_reason"])

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        view = ", ".join([f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__qualname__}({view})"

    def __eq__(self, other: object) -> bool:
        """Compare two instances of the class."""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.__dict__ == other.__dict__


class Message:
    """State of your application from the viewpoint of the entity using it.

    TODO: Update docstring
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

    @overload
    def __init__(self, content: RecordSet, dst_node_id: int, message_type: str) -> None:
        ...

    @overload
    def __init__(self, content: RecordSet, dst_node_id: int, message_type: str, *, ttl: float) -> None:
        ...

    @overload
    def __init__(self, content: RecordSet, dst_node_id: int, message_type: str, *, ttl: float, group_id: str) -> None:
        ...

    @overload
    def __init__(self, content: RecordSet, *, reply_to: Message, ttl: float) -> None:
        ...

    @overload
    def __init__(self, error: Error, *, reply_to: Message, ttl: float) -> None:
        ...

    def __init__(
        self,
        content_or_error: RecordSet | Error | None = None,
        dst_node_id: int | None = None,
        message_type: str | None = None,
        *,
        content: RecordSet | None = None,
        error: Error | None = None,
        ttl: float | None = None,
        group_id: str | None = None,
        reply_to: Message | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        # One and only one of `content_or_error`, `content` and `error` must be set
        if sum(x is not None for x in [content_or_error, content, error]) != 1:
            _raise_msg_init_error()

        # Set `content` or `error` based on `content_or_error`
        if content_or_error is not None:  # This means `content` and `error` are None
            if isinstance(content_or_error, RecordSet):
                content = content_or_error
            elif isinstance(content_or_error, Error):
                error = content_or_error
            else:
                _raise_msg_init_error()
        
        # Create metadata for an instruction message
        if reply_to is None:
            # Check arguments
            # `content`, `dst_node_id` and `message_type` must be set
            if any(x is None for x in [content, dst_node_id, message_type]):
                _raise_msg_init_error()
            
            # Set metadata
            metadata = Metadata(
                run_id="", # Will be set before pushed
                message_id="",  # Will be set by the SuperLink
                src_node_id=0,  # Will be set before pushed
                dst_node_id=dst_node_id,
                reply_to_message="",  # Instruction messages do not reply to any message
                group_id=group_id or "",
                created_at=now().timestamp(),
                ttl=ttl or DEFAULT_TTL,
                message_type=message_type,
            )
            metadata.delivered_at = ""  # Backward compatibility
        
        # Create metadata for a reply message
        else:
            # Check arguments
            # `dst_node_id`, `message_type` and `group_id` must not be set
            if any(x is not None for x in [dst_node_id, message_type, group_id]):
                _raise_msg_init_error()
            
            # Set metadata
            current = now().timestamp()
            metadata = metadata or Metadata(
                run_id=reply_to.metadata.run_id,
                message_id="",
                src_node_id=reply_to.metadata.dst_node_id,
                dst_node_id=reply_to.metadata.src_node_id,
                reply_to_message=reply_to.metadata.message_id,
                group_id=reply_to.metadata.group_id,
                created_at=current,
                ttl=_limit_reply_ttl(current, ttl, reply_to),
                message_type=reply_to.metadata.message_type,
            )
            metadata.delivered_at = ""

        var_dict = {
            "_metadata": metadata,
            "_content": content,
            "_error": error,
        }
        self.__dict__.update(var_dict)

    @property
    def metadata(self) -> Metadata:
        """A dataclass including information about the message to be executed."""
        return cast(Metadata, self.__dict__["_metadata"])

    @property
    def content(self) -> RecordSet:
        """The content of this message."""
        if self.__dict__["_content"] is None:
            raise ValueError(
                "Message content is None. Use <message>.has_content() "
                "to check if a message has content."
            )
        return cast(RecordSet, self.__dict__["_content"])

    @content.setter
    def content(self, value: RecordSet) -> None:
        """Set content."""
        if self.__dict__["_error"] is None:
            self.__dict__["_content"] = value
        else:
            raise ValueError("A message with an error set cannot have content.")

    @property
    def error(self) -> Error:
        """Error captured by this message."""
        if self.__dict__["_error"] is None:
            raise ValueError(
                "Message error is None. Use <message>.has_error() "
                "to check first if a message carries an error."
            )
        return cast(Error, self.__dict__["_error"])

    @error.setter
    def error(self, value: Error) -> None:
        """Set error."""
        if self.has_content():
            raise ValueError("A message with content set cannot carry an error.")
        self.__dict__["_error"] = value

    def has_content(self) -> bool:
        """Return True if message has content, else False."""
        return self.__dict__["_content"] is not None

    def has_error(self) -> bool:
        """Return True if message has an error, else False."""
        return self.__dict__["_error"] is not None

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

        Returns
        -------
        message : Message
            A Message containing only the relevant error and metadata.
        """
        # TODO: Warn about deprecated feature
        return Message(error, reply_to=self, ttl=ttl)

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
        # TODO: Warn about deprecated feature
        return Message(content, reply_to=self, ttl=ttl)

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        view = ", ".join(
            [
                f"{k.lstrip('_')}={v!r}"
                for k, v in self.__dict__.items()
                if v is not None
            ]
        )
        return f"{self.__class__.__qualname__}({view})"


def _limit_reply_ttl(current: float, reply_ttl: float | None, reply_to: Message) -> float:
    """Limit the TTL of a reply message to not exceed the expiration time of the
    message it replies to."""
    # Calculate the maximum allowed TTL
    max_allowed_ttl = reply_to.metadata.created_at + reply_to.metadata.ttl - current

    if reply_ttl is not None and reply_ttl - max_allowed_ttl > MESSAGE_TTL_TOLERANCE:
        log(
            WARNING,
            "The reply TTL of %.2f seconds exceeded the "
            "allowed maximum of %.2f seconds. "
            "The TTL has been updated to the allowed maximum.",
            reply_ttl,
            max_allowed_ttl,
        )
        return max_allowed_ttl

    return reply_ttl or max_allowed_ttl


def _raise_msg_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {Message.__qualname__}. Expected one of the documented signatures: "
        "Message(content: RecordSet, dst_node_id: int, message_type: str, *, [ttl: float, group_id: str]) "
        "or Message(content: RecordSet | error: Error, *, reply_to: Message, [ttl: float])."
    )
