# Copyright 2020 Adap GmbH. All Rights Reserved.
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
import logging.handlers
import os

LOGGER_NAME = "flower"

FLOWER_LOG_FILE = os.getenv("FLOWER_LOG_FILE")
FLOWER_LOG_HTTP = os.getenv("FLOWER_LOG_HTTP")


def configure() -> None:
    """Configure logger."""
    # create logger
    _logger = logging.getLogger(LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
    )

    # Console logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    if isinstance(FLOWER_LOG_FILE, str):
        # Create file handler and log to disk
        file_handler = logging.FileHandler(FLOWER_LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    if isinstance(FLOWER_LOG_HTTP, str):
        # Create http handler which logs even debug messages
        http_handler = logging.handlers.HTTPHandler(
            FLOWER_LOG_HTTP, "/log", method="POST",
        )
        http_handler.setLevel(logging.DEBUG)
        http_handler.setFormatter(formatter)
        _logger.addHandler(http_handler)


configure()

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name
log = logger.log  # pylint: disable=invalid-name
