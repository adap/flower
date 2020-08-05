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
"""Provides an Adapter implementation for Docker."""


import socket
import time
from contextlib import closing
from typing import List, Optional
from uuid import uuid4

import docker

from .adapter import Adapter, AdapterInstance


class NoPublicFacingPortFound(Exception):
    """Raise if public-facing port of container was not bound to private port of host."""


def get_free_port() -> int:
    """Returns a free port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as socks:
        socks.bind(("", 0))
        socks.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(socks.getsockname()[1])


def _get_container_port(container_id: str) -> int:
    """Return container port on host machine."""
    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    result = client.port(container_id, 22)
    client.close()
    if len(result) == 0:
        raise NoPublicFacingPortFound
    return int(result[0]["HostPort"])


class DockerAdapter(Adapter):
    """Adapter for Docker."""

    def __init__(self, name: str = "flower", network: str = "flower"):
        self.name = name
        self.network = network
        self._create_network()

    def _create_network(self) -> None:
        """Create Docker network if it does not exist."""
        client = docker.from_env()
        try:
            client.networks.get(self.network)
        except docker.errors.NotFound:
            client.networks.create(self.network, driver="bridge")
        client.close()

    # pylint: disable=too-many-arguments
    def create_instances(
        self,
        num_cpu: int,
        num_ram: float,
        timeout: int,
        num_instance: int = 1,
        gpu: bool = False,
    ) -> List[AdapterInstance]:
        """Create one or more docker container instance(s) of the same type.

            Args:
                num_cpu (int): Number of instance CPU cores (currently ignored)
                num_ram (int): RAM in GB (currently ignored)
                timeout (int): Timeout in minutes
                num_instance (int): Number of instances to start

        """
        instances: List[AdapterInstance] = []

        client = docker.from_env()
        for _ in range(num_instance):
            port = get_free_port()
            container = client.containers.run(
                "flower-sshd:latest",
                auto_remove=True,
                detach=True,
                ports={"22/tcp": port},
                network=self.network,
                labels={"adapter_name": self.name},
                # We have to assign a name as the default random name will not work
                # as hostname so the containers can reach each other
                name=str(uuid4().hex[:8]),
            )

            # Docker needs a little time to start the container
            time.sleep(1)

            port = _get_container_port(container.short_id)
            instances.append(
                (container.short_id, container.name, "127.0.0.1", port, "started")
            )

        client.close()

        return instances

    def list_instances(
        self, instance_ids: Optional[List[str]] = None
    ) -> List[AdapterInstance]:
        """List all container instances with tags belonging to this adapter.

        Args:
            instance_ids ([str[]]): If provided, filter by instance_ids
        """
        instances: List[AdapterInstance] = []

        client = docker.from_env()
        containers = client.containers.list(
            filters={"label": f"adapter_name={self.name}"}
        )
        for container in containers:
            port = _get_container_port(container.short_id)
            instances.append(
                (
                    container.short_id,
                    container.name,
                    "127.0.0.1",
                    port,
                    container.status,
                )
            )
        client.close()

        return instances

    def terminate_instances(self, instance_ids: List[str]) -> None:
        """Terminate container instance(s).

        Will raise an error if something goes wrong.
        """
        client = docker.from_env()
        for instance_id in instance_ids:
            container = client.containers.get(instance_id)
            container.remove(force=True)
        client.close()

    def terminate_all_instances(self) -> None:
        """Terminate all instances.

        Will raise an error if something goes wrong.
        """
        client = docker.from_env()
        containers = client.containers.list(
            filters={"label": f"adapter_name={self.name}"}
        )
        for container in containers:
            container.remove(force=True)
        client.close()
