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
"""Implments compute classes for EC2."""

import os
import unittest
import warnings
from typing import Sized, cast
from unittest.mock import MagicMock

from .cluster import (
    Cluster,
    InstanceMismatch,
    create_instances,
    group_instances_by_specs,
)
from .compute.ec2_adapter import EC2Adapter
from .instance import Instance

IMAGE_ID = "ami-0370b0294d7241341"
KEY_NAME = "flower"
SSH_CREDENTIALS = ("ubuntu", "/Users/tanto/.ssh/flower.pem")
SUBNET_ID = "subnet-23da286f"
SECURITY_GROUP_IDS = ["sg-0dd0f0080bcf86400"]


class CreateInstancesTestCase(unittest.TestCase):
    """Test cases for create_instances."""

    def setUp(self) -> None:
        """Prepare tests."""
        self.mock_adapter = MagicMock()
        self.mock_adapter.create_instances.return_value = [
            (1, "1.1.1.1", "2.2.2.1", 22, "running"),
            (2, "1.1.1.2", "2.2.2.2", 22, "running"),
        ]
        self.timeout = 10

    def test_create_instances(self) -> None:
        """Test if create_instances works correctly."""
        # Prepare
        instances = [
            Instance(name="client_0", group="clients", num_cpu=2, num_ram=8),
            Instance(name="client_1", group="clients", num_cpu=2, num_ram=8),
        ]

        # Execute
        create_instances(
            adapter=self.mock_adapter, instances=instances, timeout=self.timeout
        )

        # Assert
        self.mock_adapter.create_instances.assert_called_once_with(
            num_cpu=instances[0].num_cpu,
            num_ram=instances[0].num_ram,
            num_instance=len(instances),
            timeout=10,
            gpu=False,
        )
        for ins in instances:
            assert ins.instance_id is not None
            assert ins.private_ip is not None
            assert ins.public_ip is not None
            assert ins.ssh_port is not None
            assert ins.state is not None

    def test_create_instances_fail(self) -> None:
        """Test if create_instances fails when instances list is invalid."""
        # Prepare
        instances = [
            Instance(name="client_0", group="clients", num_cpu=2, num_ram=8),
            Instance(name="client_1", group="clients", num_cpu=1, num_ram=4),
        ]

        # Execute
        with self.assertRaises(InstanceMismatch):
            create_instances(
                adapter=self.mock_adapter, instances=instances, timeout=self.timeout
            )


def test_group_instances_by_specs() -> None:
    """Test that function works correctly."""
    # Prepare
    instances = [
        Instance(name="server", group="server", num_cpu=2, num_ram=4),
        Instance(name="client_0", group="clients", num_cpu=2, num_ram=8),
        Instance(name="logserver", group="logserver", num_cpu=2, num_ram=4),
        Instance(name="client_1", group="clients", num_cpu=2, num_ram=8),
    ]
    expected_groups = [[instances[0], instances[2]], [instances[1], instances[3]]]

    # Execute
    groups = group_instances_by_specs(instances)

    # Assert
    assert len(groups) == 2
    assert groups == expected_groups


if os.getenv("FLOWER_INTEGRATION"):

    class ClusterIntegrationTestCase(unittest.TestCase):
        """Integration tests class Cluster.

        This TestCase will not mock anythin and use a live EC2Adapter
        which will be used to provision a single machine and execute a
        single command on it. Afterwards the machines will be shut down.
        """

        # pylint: disable=too-many-instance-attributes
        def setUp(self) -> None:
            """Create an instance."""
            # Filter false positiv warning
            warnings.filterwarnings(
                "ignore",
                category=ResourceWarning,
                message="unclosed.*<ssl.SSLSocket.*>",
            )

            adapter = EC2Adapter(
                image_id=IMAGE_ID,
                key_name=KEY_NAME,
                subnet_id=SUBNET_ID,
                security_group_ids=SECURITY_GROUP_IDS,
                tags=[
                    ("Purpose", "integration_test"),
                    ("Test Name", "ClusterIntegrationTestCase"),
                ],
            )
            self.cluster = Cluster(
                adapter=adapter,
                ssh_credentials=SSH_CREDENTIALS,
                instances=[
                    Instance(name="server", group="server", num_cpu=2, num_ram=2)
                ],
                # In case the tearDown fails for some reason the machines
                # should automatically terminate after 10 minutes
                timeout=10,
            )
            self.cluster.start()

        def tearDown(self) -> None:
            self.cluster.terminate()

        def test_exec(self) -> None:
            """Execute on all clients."""
            # Prepare
            command = "nproc"
            expected_result = "2\n"

            # Execute
            stdout, stderr = self.cluster.exec("server", command)

            casted_stderr: Sized = cast(Sized, stderr)
            casted_stdout: Sized = cast(Sized, stdout)

            # Assert
            assert len(casted_stderr) == 0
            assert len(casted_stdout) == 1
            assert "".join(stdout) == expected_result


if __name__ == "__main__":
    unittest.main(verbosity=2)
