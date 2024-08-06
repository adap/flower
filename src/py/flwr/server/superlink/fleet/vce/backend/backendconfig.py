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
"""Backend config."""


from typing import Optional
from dataclasses import dataclass
from flwr.common.typing import UserConfig # = Dict[str, UserConfigValue]


@dataclass
class ClientAppResources:
    """Resources for a `ClientApp`"""
    num_cpus: float
    num_gpus: float


@dataclass
class BackendConfig:

    name: str
    clientapp_resources: ClientAppResources
    config: Optional[UserConfig]

    def __init__(self,
                 name: Optional[str] = "ray", 
                 clientapp_resources: Optional[ClientAppResources] = ClientAppResources(2.0, 0.0),
                config: Optional[UserConfig] = {},
                )
        self.name = name
        self.clientapp_resources = clientapp_resources
        self.config = config



