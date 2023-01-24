# Copyright 2023 Adap GmbH. All Rights Reserved.
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
"""Singleton holding shared state."""


from logging import DEBUG
from typing import Optional

from flwr.common.logger import log
from flwr.server.driver.state import DriverState


class Singleton(object):
    """The Singleton singleton is a container to hold shared references.

    There are to several components that are needed to facilitate
    communication between the main thread (running the Flower server's
    fit method) and the REST API server thread (running the FastAPI
    server through uvicorn). At the time of writing, there is one shared
    component: DriverState.
    """

    _instance = None
    _driver_state = None

    def __init__(self) -> None:
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls) -> Singleton:
        # This is not threadsafe
        if cls._instance is None:
            log(DEBUG, "Creating new instance")
            cls._instance = cls.__new__(cls)
        return cls._instance

    def set_driver_state(self, driver_state: DriverState) -> None:
        self._driver_state = driver_state

    def get_driver_state(self) -> Optional[DriverState]:
        return self._driver_state
