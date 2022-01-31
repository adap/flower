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
"""Provides dataclass Instance."""
from dataclasses import dataclass
from typing import Optional


# pylint: disable=too-many-instance-attributes
@dataclass
class Instance:
    """Represents an instance."""

    # Specs
    name: str
    group: str
    num_cpu: int
    num_ram: float
    gpu: bool = False

    # Runtime information
    instance_id: Optional[str] = None
    private_ip: Optional[str] = None
    public_ip: Optional[str] = None
    ssh_port: Optional[int] = None
    state: Optional[str] = None
