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
import os
from datetime import datetime

LOGGER_NAME = "flower"
DEFAULT_LOGFILE = f"flower_{datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]}_{os.getpid()}.log"  # pylint: disable=C0301


def configure(logfile: str = DEFAULT_LOGFILE) -> None:
    """Configure logger."""
    # create logger
    _logger = logging.getLogger(LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
    )

    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # create file handler which logs even debug messages
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # add ch and fh to logger
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)


configure()

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name
log = logger.log  # pylint: disable=invalid-name
