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
"""Flower Logger."""

import logging
import re
import sys
import threading
import time
from io import StringIO
from logging import WARN, LogRecord
from logging.handlers import HTTPHandler
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Optional, TextIO, Union

import grpc

from flwr.proto.log_pb2 import PushLogsRequest  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611
from flwr.proto.simulationio_pb2_grpc import SimulationIoStub  # pylint: disable=E0611

from .constant import LOG_UPLOAD_INTERVAL

# Create logger
LOGGER_NAME = "flwr"
FLOWER_LOGGER = logging.getLogger(LOGGER_NAME)
FLOWER_LOGGER.setLevel(logging.DEBUG)

LOG_COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
    "RESET": "\033[0m",  # Reset to default
}

if TYPE_CHECKING:
    StreamHandler = logging.StreamHandler[Any]
else:
    StreamHandler = logging.StreamHandler


class ConsoleHandler(StreamHandler):
    """Console handler that allows configurable formatting."""

    def __init__(
        self,
        timestamps: bool = False,
        json: bool = False,
        colored: bool = True,
        stream: Optional[TextIO] = None,
    ) -> None:
        super().__init__(stream)
        self.timestamps = timestamps
        self.json = json
        self.colored = colored

    def emit(self, record: LogRecord) -> None:
        """Emit a record."""
        if self.json:
            record.message = record.getMessage().replace("\t", "").strip()

            # Check if the message is empty
            if not record.message:
                return

        super().emit(record)

    def format(self, record: LogRecord) -> str:
        """Format function that adds colors to log level."""
        seperator = " " * (8 - len(record.levelname))
        if self.json:
            log_fmt = "{lvl='%(levelname)s', time='%(asctime)s', msg='%(message)s'}"
        else:
            log_fmt = (
                f"{LOG_COLORS[record.levelname] if self.colored else ''}"
                f"%(levelname)s {'%(asctime)s' if self.timestamps else ''}"
                f"{LOG_COLORS['RESET'] if self.colored else ''}"
                f": {seperator} %(message)s"
            )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def update_console_handler(
    level: Optional[int] = None,
    timestamps: Optional[bool] = None,
    colored: Optional[bool] = None,
) -> None:
    """Update the logging handler."""
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        if isinstance(handler, ConsoleHandler):
            if level is not None:
                handler.setLevel(level)
            if timestamps is not None:
                handler.timestamps = timestamps
            if colored is not None:
                handler.colored = colored


# Configure console logger
console_handler = ConsoleHandler(
    timestamps=False,
    json=False,
    colored=True,
)
console_handler.setLevel(logging.INFO)
FLOWER_LOGGER.addHandler(console_handler)


class CustomHTTPHandler(HTTPHandler):
    """Custom HTTPHandler which overrides the mapLogRecords method."""

    # pylint: disable=too-many-arguments,bad-option-value,R1725,R0917
    def __init__(
        self,
        identifier: str,
        host: str,
        url: str,
        method: str = "GET",
        secure: bool = False,
        credentials: Optional[tuple[str, str]] = None,
    ) -> None:
        super().__init__(host, url, method, secure, credentials)
        self.identifier = identifier

    def mapLogRecord(self, record: LogRecord) -> dict[str, Any]:
        """Filter for the properties to be send to the logserver."""
        record_dict = record.__dict__
        return {
            "identifier": self.identifier,
            "levelname": record_dict["levelname"],
            "name": record_dict["name"],
            "asctime": record_dict["asctime"],
            "filename": record_dict["filename"],
            "lineno": record_dict["lineno"],
            "message": record_dict["message"],
        }


def configure(
    identifier: str, filename: Optional[str] = None, host: Optional[str] = None
) -> None:
    """Configure logging to file and/or remote log server."""
    # Create formatter
    string_to_input = f"{identifier} | %(levelname)s %(name)s %(asctime)s "
    string_to_input += "| %(filename)s:%(lineno)d | %(message)s"
    formatter = logging.Formatter(string_to_input)

    if filename:
        # Create file handler and log to disk
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        FLOWER_LOGGER.addHandler(file_handler)

    if host:
        # Create http handler which logs even debug messages
        http_handler = CustomHTTPHandler(
            identifier,
            host,
            "/log",
            method="POST",
        )
        http_handler.setLevel(logging.DEBUG)
        # Override mapLogRecords as setFormatter has no effect on what is send via http
        FLOWER_LOGGER.addHandler(http_handler)


logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name
log = logger.log  # pylint: disable=invalid-name


def warn_preview_feature(name: str) -> None:
    """Warn the user when they use a preview feature."""
    log(
        WARN,
        """PREVIEW FEATURE: %s

            This is a preview feature. It could change significantly or be removed
            entirely in future versions of Flower.
        """,
        name,
    )


def warn_deprecated_feature(name: str) -> None:
    """Warn the user when they use a deprecated feature."""
    log(
        WARN,
        """DEPRECATED FEATURE: %s

            This is a deprecated feature. It will be removed
            entirely in future versions of Flower.
        """,
        name,
    )


def warn_deprecated_feature_with_example(
    deprecation_message: str, example_message: str, code_example: str
) -> None:
    """Warn if a feature is deprecated and show code example."""
    log(
        WARN,
        """DEPRECATED FEATURE: %s

            Check the following `FEATURE UPDATE` warning message for the preferred
            new mechanism to use this feature in Flower.
        """,
        deprecation_message,
    )
    log(
        WARN,
        """FEATURE UPDATE: %s
        ------------------------------------------------------------
        %s
        ------------------------------------------------------------
        """,
        example_message,
        code_example,
    )


def warn_unsupported_feature(name: str) -> None:
    """Warn the user when they use an unsupported feature."""
    log(
        WARN,
        """UNSUPPORTED FEATURE: %s

            This is an unsupported feature. It will be removed
            entirely in future versions of Flower.
        """,
        name,
    )


def set_logger_propagation(
    child_logger: logging.Logger, value: bool = True
) -> logging.Logger:
    """Set the logger propagation attribute.

    Parameters
    ----------
    child_logger : logging.Logger
        Child logger object
    value : bool
        Boolean setting for propagation. If True, both parent and child logger
        display messages. Otherwise, only the child logger displays a message.
        This False setting prevents duplicate logs in Colab notebooks.
        Reference: https://stackoverflow.com/a/19561320

    Returns
    -------
    logging.Logger
        Child logger object with updated propagation setting
    """
    child_logger.propagate = value
    if not child_logger.propagate:
        child_logger.log(logging.DEBUG, "Logger propagate set to False")
    return child_logger


def mirror_output_to_queue(log_queue: Queue[Optional[str]]) -> None:
    """Mirror stdout and stderr output to the provided queue."""

    def get_write_fn(stream: TextIO) -> Any:
        original_write = stream.write

        def fn(s: str) -> int:
            ret = original_write(s)
            stream.flush()
            log_queue.put(s)
            return ret

        return fn

    sys.stdout.write = get_write_fn(sys.stdout)  # type: ignore[method-assign]
    sys.stderr.write = get_write_fn(sys.stderr)  # type: ignore[method-assign]
    console_handler.stream = sys.stdout


def restore_output() -> None:
    """Restore stdout and stderr.

    This will stop mirroring output to queues.
    """
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    console_handler.stream = sys.stdout


def redirect_output(output_buffer: StringIO) -> None:
    """Redirect stdout and stderr to text I/O buffer."""
    sys.stdout = output_buffer
    sys.stderr = output_buffer
    console_handler.stream = sys.stdout


def _log_uploader(
    log_queue: Queue[Optional[str]], node_id: int, run_id: int, stub: ServerAppIoStub
) -> None:
    """Upload logs to the SuperLink."""
    exit_flag = False
    node = Node(node_id=node_id, anonymous=False)
    msgs: list[str] = []
    while True:
        # Fetch all messages from the queue
        try:
            while True:
                msg = log_queue.get_nowait()
                # Quit the loops if the returned message is `None`
                # This is a signal that the run has finished
                if msg is None:
                    exit_flag = True
                    break
                msgs.append(msg)
        except Empty:
            pass

        # Upload if any logs
        if msgs:
            req = PushLogsRequest(
                node=node,
                run_id=run_id,
                logs=msgs,
            )
            try:
                stub.PushLogs(req)
                msgs.clear()
            except grpc.RpcError as e:
                # Ignore minor network errors
                # pylint: disable-next=no-member
                if e.code() != grpc.StatusCode.UNAVAILABLE:
                    raise e

        if exit_flag:
            break

        time.sleep(LOG_UPLOAD_INTERVAL)


def start_log_uploader(
    log_queue: Queue[Optional[str]],
    node_id: int,
    run_id: int,
    stub: Union[ServerAppIoStub, SimulationIoStub],
) -> threading.Thread:
    """Start the log uploader thread and return it."""
    thread = threading.Thread(
        target=_log_uploader, args=(log_queue, node_id, run_id, stub)
    )
    thread.start()
    return thread


def stop_log_uploader(
    log_queue: Queue[Optional[str]], log_uploader: threading.Thread
) -> None:
    """Stop the log uploader thread."""
    log_queue.put(None)
    log_uploader.join()


def remove_emojis(text: str) -> str:
    """Remove emojis from the provided text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)
