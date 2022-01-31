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
"""Provides a standardised interface for provisioning compute resources."""


from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

AdapterInstance = Tuple[
    str, str, Optional[str], int, str
]  # (InstanceId, PrivateIpAddress, PublicIpAddress, SSHPort, State)


class Adapter(ABC):
    """Base class for different Adapter implementations, for example, AWS
    EC2."""

    # pylint: disable=too-many-arguments
    @abstractmethod
    def create_instances(
        self,
        num_cpu: int,
        num_ram: float,
        timeout: int,
        num_instance: int = 1,
        gpu: bool = False,
    ) -> List[AdapterInstance]:
        """Create one or more instance(s) of the same type.

        Args:
            num_cpu (int): Number of instance CPU
            num_ram (int): RAM in GB
            num_instance (int): Number of instances to start if currently available
            timeout (int): Timeout in minutes
            commands (:obj:`str`, optional): List of bash commands which will be joined into a
                single string with newline as a seperator
            gpu (bool): If true will only consider instances with GPU
        """

    @abstractmethod
    def list_instances(
        self, instance_ids: Optional[List[str]] = None
    ) -> List[AdapterInstance]:
        """List all instances with tags belonging to this adapter.

        Args:
            instance_ids (:obj:`list` of :obj:`str`, optional): If provided, filter by instance_ids
        """

    @abstractmethod
    def terminate_instances(self, instance_ids: List[str]) -> None:
        """Terminate instances.

        Should raise an error if something goes wrong.
        """

    @abstractmethod
    def terminate_all_instances(self) -> None:
        """Terminate all instances.

        Will raise an error if something goes wrong.
        """
