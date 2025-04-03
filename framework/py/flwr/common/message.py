# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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

from logging import WARNING
from typing import Any, Optional, cast, overload

from flwr.common.date import now
from flwr.common.logger import warn_deprecated_feature

from .constant import MESSAGE_TTL_TOLERANCE, MessageType, MessageTypeLegacy
from .logger import log
from .record import RecordDict

DEFAULT_TTL = 43200  # This is 12 hours
MESSAGE_INIT_ERROR_MESSAGE = (
    "Invalid arguments for Message. Expected one of the documented "
    "signatures: Message(content: RecordDict, dst_node_id: int, message_type: str,"
    " *, [ttl: float, group_id: str]) or Message(content: RecordDict | error: Error,"
    " *, reply_to: Message, [ttl: float])."
)


class MessageInitializationError(TypeError):
    """Error raised when initializing a message with invalid arguments."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or MESSAGE_INIT_ERROR_MESSAGE)


class Metadata:  # pylint: disable=too-many-instance-attributes
    """The class representing metadata associated with the current message.

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
    reply_to_message_id : str
        An identifier for the message to which this message is a reply.
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
        reply_to_message_id: str,
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
            "_reply_to_message_id": reply_to_message_id,
            "_group_id": group_id,
            "_created_at": created_at,
            "_ttl": ttl,
            "_message_type": message_type,
        }
        self.__dict__.update(var_dict)
        self.message_type = message_type  # Trigger validation

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
    def reply_to_message_id(self) -> str:
        """An identifier for the message to which this message is a reply."""
        return cast(str, self.__dict__["_reply_to_message_id"])

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
        """Set creation timestamp of this message."""
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
        # Validate message type
        if validate_legacy_message_type(value):
            pass  # Backward compatibility for legacy message types
        elif not validate_message_type(value):
            raise ValueError(
                f"Invalid message type: '{value}'. "
                "Expected format: '<category>' or '<category>.<action>', "
                "where <category> must be 'train', 'evaluate', or 'query', "
                "and <action> must be a valid Python identifier."
            )

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
    """The class storing information about an error that occurred.

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
    """Represents a message exchanged between ClientApp and ServerApp.

    This class encapsulates the payload and metadata necessary for communication
    between a ClientApp and a ServerApp.

    Parameters
    ----------
    content : Optional[RecordDict] (default: None)
        Holds records either sent by another entity (e.g. sent by the server-side
        logic to a client, or vice-versa) or that will be sent to it.
    error : Optional[Error] (default: None)
        A dataclass that captures information about an error that took place
        when processing another message.
    dst_node_id : Optional[int] (default: None)
        An identifier for the node receiving this message.
    message_type : Optional[str] (default: None)
        A string that encodes the action to be executed on
        the receiving end.
    ttl : Optional[float] (default: None)
        Time-to-live (TTL) for this message in seconds. If `None` (default),
        the TTL is set to 43,200 seconds (12 hours).
    group_id : Optional[str] (default: None)
        An identifier for grouping messages. In some settings, this is used as
        the FL round.
    reply_to : Optional[Message] (default: None)
        The instruction message to which this message is a reply. This message does
        not retain the original message's content but derives its metadata from it.
    """

    @overload
    def __init__(  # pylint: disable=too-many-arguments  # noqa: E704
        self,
        content: RecordDict,
        dst_node_id: int,
        message_type: str,
        *,
        ttl: float | None = None,
        group_id: str | None = None,
    ) -> None: ...

    @overload
    def __init__(  # noqa: E704
        self, content: RecordDict, *, reply_to: Message, ttl: float | None = None
    ) -> None: ...

    @overload
    def __init__(  # noqa: E704
        self, error: Error, *, reply_to: Message, ttl: float | None = None
    ) -> None: ...

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *args: Any,
        dst_node_id: int | None = None,
        message_type: str | None = None,
        content: RecordDict | None = None,
        error: Error | None = None,
        ttl: float | None = None,
        group_id: str | None = None,
        reply_to: Message | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        # Set positional arguments
        content, error, dst_node_id, message_type = _extract_positional_args(
            *args,
            content=content,
            error=error,
            dst_node_id=dst_node_id,
            message_type=message_type,
        )
        _check_arg_types(
            dst_node_id=dst_node_id,
            message_type=message_type,
            content=content,
            error=error,
            ttl=ttl,
            group_id=group_id,
            reply_to=reply_to,
            metadata=metadata,
        )

        # Set metadata directly (This is for internal use only)
        if metadata is not None:
            # When metadata is set, all other arguments must be None,
            # except `content`, `error`, or `content_or_error`
            if any(
                x is not None
                for x in [dst_node_id, message_type, ttl, group_id, reply_to]
            ):
                raise MessageInitializationError(
                    f"Invalid arguments for {Message.__qualname__}. "
                    "Expected only `metadata` to be set when creating a message "
                    "with provided metadata."
                )

        # Create metadata for an instruction message
        elif reply_to is None:
            # Check arguments
            # `content`, `dst_node_id` and `message_type` must be set
            if not (
                isinstance(content, RecordDict)
                and isinstance(dst_node_id, int)
                and isinstance(message_type, str)
            ):
                raise MessageInitializationError()

            # Set metadata
            metadata = Metadata(
                run_id=0,  # Will be set before pushed
                message_id="",  # Will be set by the SuperLink
                src_node_id=0,  # Will be set before pushed
                dst_node_id=dst_node_id,
                # Instruction messages do not reply to any message
                reply_to_message_id="",
                group_id=group_id or "",
                created_at=now().timestamp(),
                ttl=ttl or DEFAULT_TTL,
                message_type=message_type,
            )

        # Create metadata for a reply message
        else:
            # Check arguments
            # `dst_node_id`, `message_type` and `group_id` must not be set
            if any(x is not None for x in [dst_node_id, message_type, group_id]):
                raise MessageInitializationError()

            # Set metadata
            current = now().timestamp()
            metadata = Metadata(
                run_id=reply_to.metadata.run_id,
                message_id="",  # Will be set by the SuperLink
                src_node_id=reply_to.metadata.dst_node_id,
                dst_node_id=reply_to.metadata.src_node_id,
                reply_to_message_id=reply_to.metadata.message_id,
                group_id=reply_to.metadata.group_id,
                created_at=current,
                ttl=_limit_reply_ttl(current, ttl, reply_to),
                message_type=reply_to.metadata.message_type,
            )

        metadata.delivered_at = ""  # Backward compatibility
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
    def content(self) -> RecordDict:
        """The content of this message."""
        if self.__dict__["_content"] is None:
            raise ValueError(
                "Message content is None. Use <message>.has_content() "
                "to check if a message has content."
            )
        return cast(RecordDict, self.__dict__["_content"])

    @content.setter
    def content(self, value: RecordDict) -> None:
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
        warn_deprecated_feature(
            "`Message.create_error_reply` is deprecated. "
            "Instead of calling `some_message.create_error_reply(some_error, ttl=...)`"
            ", use `Message(some_error, reply_to=some_message, ttl=...)`."
        )
        if ttl is not None:
            return Message(error, reply_to=self, ttl=ttl)
        return Message(error, reply_to=self)

    def create_reply(self, content: RecordDict, ttl: float | None = None) -> Message:
        """Create a reply to this message with specified content and TTL.

        The method generates a new `Message` as a reply to this message.
        It inherits 'run_id', 'src_node_id', 'dst_node_id', and 'message_type' from
        this message and sets 'reply_to_message_id' to the ID of this message.

        Parameters
        ----------
        content : RecordDict
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
        warn_deprecated_feature(
            "`Message.create_reply` is deprecated. "
            "Instead of calling `some_message.create_reply(some_content, ttl=...)`"
            ", use `Message(some_content, reply_to=some_message, ttl=...)`."
        )
        if ttl is not None:
            return Message(content, reply_to=self, ttl=ttl)
        return Message(content, reply_to=self)

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


def make_message(
    metadata: Metadata, content: RecordDict | None = None, error: Error | None = None
) -> Message:
    """Create a message with the provided metadata, content, and error."""
    return Message(metadata=metadata, content=content, error=error)  # type: ignore


def _limit_reply_ttl(
    current: float, reply_ttl: float | None, reply_to: Message
) -> float:
    """Limit the TTL of a reply message such that it does exceed the expiration time of
    the message it replies to."""
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


def _extract_positional_args(
    *args: Any,
    content: RecordDict | None,
    error: Error | None,
    dst_node_id: int | None,
    message_type: str | None,
) -> tuple[RecordDict | None, Error | None, int | None, str | None]:
    """Extract positional arguments for the `Message` constructor."""
    content_or_error = args[0] if args else None
    if len(args) > 1:
        if dst_node_id is not None:
            raise MessageInitializationError()
        dst_node_id = args[1]
    if len(args) > 2:
        if message_type is not None:
            raise MessageInitializationError()
        message_type = args[2]
    if len(args) > 3:
        raise MessageInitializationError()

    # One and only one of `content_or_error`, `content` and `error` must be set
    if sum(x is not None for x in [content_or_error, content, error]) != 1:
        raise MessageInitializationError()

    # Set `content` or `error` based on `content_or_error`
    if content_or_error is not None:  # This means `content` and `error` are None
        if isinstance(content_or_error, RecordDict):
            content = content_or_error
        elif isinstance(content_or_error, Error):
            error = content_or_error
        else:
            raise MessageInitializationError()
    return content, error, dst_node_id, message_type


def _check_arg_types(  # pylint: disable=too-many-arguments, R0917
    dst_node_id: int | None = None,
    message_type: str | None = None,
    content: RecordDict | None = None,
    error: Error | None = None,
    ttl: float | None = None,
    group_id: str | None = None,
    reply_to: Message | None = None,
    metadata: Metadata | None = None,
) -> None:
    """Check argument types for the `Message` constructor."""
    # pylint: disable=too-many-boolean-expressions
    if (
        (dst_node_id is None or isinstance(dst_node_id, int))
        and (message_type is None or isinstance(message_type, str))
        and (content is None or isinstance(content, RecordDict))
        and (error is None or isinstance(error, Error))
        and (ttl is None or isinstance(ttl, (int, float)))
        and (group_id is None or isinstance(group_id, str))
        and (reply_to is None or isinstance(reply_to, Message))
        and (metadata is None or isinstance(metadata, Metadata))
    ):
        return
    raise MessageInitializationError()


def validate_message_type(message_type: str) -> bool:
    """Validate if the message type is valid.

    A valid message type format must be one of the following:

    - "<category>"
    - "<category>.<action>"

    where `category` must be one of "train", "evaluate", or "query",
    and `action` must be a valid Python identifier.
    """
    # Check if conforming to the format "<category>"
    valid_types = {
        MessageType.TRAIN,
        MessageType.EVALUATE,
        MessageType.QUERY,
        MessageType.SYSTEM,
    }
    if message_type in valid_types:
        return True

    # Check if conforming to the format "<category>.<action>"
    if message_type.count(".") != 1:
        return False

    category, action = message_type.split(".")
    if category in valid_types and action.isidentifier():
        return True

    return False


def validate_legacy_message_type(message_type: str) -> bool:
    """Validate if the legacy message type is valid."""
    # Backward compatibility for legacy message types
    if message_type in (
        MessageTypeLegacy.GET_PARAMETERS,
        MessageTypeLegacy.GET_PROPERTIES,
        "reconnect",
    ):
        return True

    return False
