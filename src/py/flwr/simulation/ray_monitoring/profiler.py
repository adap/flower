# Copyright 2022 Adap GmbH. All Rights Reserved.
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
import os
from logging import DEBUG
from functools import wraps
from typing import Callable, Dict, List, Optional, TypeVar, Union, cast

import ray
from ray.experimental.state.api import list_nodes, list_tasks

from flwr import common
from flwr.client import Client
from flwr.common.logger import log
from flwr.monitoring.profiler import SystemMonitor
from flwr.simulation.ray_transport.ray_client_proxy import (
    ClientFn,
    RayClientProxy,
    _create_client,
)


@ray.remote
class RaySystemMonitor(SystemMonitor):
    def __init__(self, *, node_id: str, interval_s: float = 0.1):
        super().__init__(node_id=node_id, interval_s=interval_s)
