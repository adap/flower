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
"""Tests DockerAdapter."""
import os
import time
import unittest

import docker

from .docker_adapter import DockerAdapter, get_free_port

if os.getenv("FLOWER_INTEGRATION"):

    class DockerAdapterIntegrationTestCase(unittest.TestCase):
        """Test suite for class DockerAdapter.

        Required docker to be available on the host machine.
        """

        def setUp(self) -> None:
            """Prepare tests."""
            self.name = "flower_test"
            self.client = docker.from_env()
            self.adapter = DockerAdapter(name=self.name)

        def tearDown(self) -> None:
            """Cleanup tests."""
            containers = self.client.containers.list(
                filters={"label": f"adapter_name={self.name}"}
            )
            for container in containers:
                container.remove(force=True)
            self.client.close()

        def test_create_instances(self) -> None:
            """Create and start an instance."""
            # Execute
            instances = self.adapter.create_instances(
                num_cpu=2, num_ram=2, timeout=1, num_instance=2, gpu=False
            )

            # Assert
            assert len(instances) == 2

            containers = self.client.containers.list(
                filters={"label": f"adapter_name={self.name}"}
            )
            assert len(containers) == 2

        def test_list_instances(self) -> None:
            """List all instances."""
            # Prepare
            for _ in range(2):
                port = get_free_port()
                self.client.containers.run(
                    "flower-sshd:latest",
                    auto_remove=True,
                    detach=True,
                    ports={"22/tcp": port},
                    labels={"adapter_name": self.name},
                )

            # Execute
            instances = self.adapter.list_instances()

            # Assert
            assert len(instances) == 2, "Expected to find two instances."
            ports = {i[3] for i in instances}
            assert len(ports) == 2, "Each instance should have a distinct port."

        def test_terminate_instance(self) -> None:
            """Destroy all instances."""
            # Prepare
            port = get_free_port()
            container = self.client.containers.run(
                "flower-sshd:latest",
                name=f"{self.name}_{int(time.time() * 1000)}",
                auto_remove=True,
                detach=True,
                ports={"22/tcp": port},
                labels={"adapter_name": self.name},
            )

            # Execute
            self.adapter.terminate_instances([container.short_id])

            # Assert
            containers = self.client.containers.list(
                filters={"label": f"adapter_name={self.name}"}
            )
            assert len(containers) == 0

        def test_terminate_all_instances(self) -> None:
            """Destroy all instances."""
            # Prepare
            for _ in range(2):
                port = get_free_port()
                self.client.containers.run(
                    "flower-sshd:latest",
                    name=f"{self.name}_{int(time.time() * 1000)}",
                    auto_remove=True,
                    detach=True,
                    ports={"22/tcp": port},
                )

            # Execute
            self.adapter.terminate_all_instances()

            # Assert
            containers = self.client.containers.list(
                filters={"label": f"adapter_name={self.name}"}
            )
            assert len(containers) == 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
