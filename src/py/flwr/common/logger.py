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
from logging import WARN, LogRecord
from logging.handlers import HTTPHandler
from typing import Any, Dict, Optional, Tuple

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


class ColorFormatter(logging.Formatter):
    """Log formatter that adds color to the log level."""

    def format(self, record):
        """Format function that adds colors to log level."""
        seperator = " " * (8 - len(record.levelname))
        log_fmt = (
            f"{LOG_COLORS[record.levelname]}"
            f"%(levelname)s{LOG_COLORS['RESET']}: {seperator} %(message)s"
        )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Configure console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColorFormatter())
FLOWER_LOGGER.addHandler(console_handler)


class CustomHTTPHandler(HTTPHandler):
    """Custom HTTPHandler which overrides the mapLogRecords method."""

    # pylint: disable=too-many-arguments,bad-option-value,R1725
    def __init__(
        self,
        identifier: str,
        host: str,
        url: str,
        method: str = "GET",
        secure: bool = False,
        credentials: Optional[Tuple[str, str]] = None,
    ) -> None:
        super().__init__(host, url, method, secure, credentials)
        self.identifier = identifier

    def mapLogRecord(self, record: LogRecord) -> Dict[str, Any]:
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


def warn_experimental_feature(name: str) -> None:
    """Warn the user when they use an experimental feature."""
    log(
        WARN,
        """
        EXPERIMENTAL FEATURE: %s

        This is an experimental feature. It could change significantly or be removed
        entirely in future versions of Flower.
        """,
        name,
    )


def warn_deprecated_feature(name: str) -> None:
    """Warn the user when they use a deprecated feature."""
    log(
        WARN,
        """
        DEPRECATED FEATURE: %s

        This is a deprecated feature. It will be removed
        entirely in future versions of Flower.
        """,
        name,
    )
