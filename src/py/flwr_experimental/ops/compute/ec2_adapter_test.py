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
"""Tests EC2Adapter."""
import os
import unittest
import warnings
from unittest.mock import MagicMock

from .ec2_adapter import EC2Adapter

IMAGE_ID = "ami-0370b0294d7241341"
KEY_NAME = "flower"
SUBNET_ID = "subnet-23da286f"
SECURITY_GROUP_IDS = ["sg-0dd0f0080bcf86400"]


class EC2AdapterTestCase(unittest.TestCase):
    """Test suite for class EC2Adapter."""

    def setUp(self) -> None:
        """Create an instance."""
        self.ec2_mock = MagicMock()

        self.ec2_mock.run_instances.return_value = {
            "Instances": [
                {
                    "InstanceId": "1",
                    "PrivateIpAddress": "1.1.1.1",
                    "PublicIpAddress": "2.1.1.1",
                    "State": {"Name": "pending"},
                }
            ]
        }

        self.ec2_mock.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "1",
                            "PrivateIpAddress": "1.1.1.1",
                            "PublicIpAddress": "2.1.1.1",
                            "State": {"Name": "running"},
                        }
                    ]
                }
            ]
        }

        self.adapter = EC2Adapter(
            image_id="ami-0370b0294d7241341",
            key_name="flower",
            subnet_id="subnet-23da286f",
            security_group_ids=["sg-0dd0f0080bcf86400"],
            tags=[("Purpose", "integration_test"), ("Test Name", "EC2AdapterTestCase")],
            boto_ec2_client=self.ec2_mock,
        )

    def test_create_instances(self) -> None:
        """Create and start an instance."""
        # Prepare
        reservations = self.ec2_mock.describe_instances.return_value["Reservations"]
        ec2_instance = reservations[0]["Instances"][0]

        expected_return_value = (
            ec2_instance["InstanceId"],
            ec2_instance["PrivateIpAddress"],
            ec2_instance["PublicIpAddress"],
            22,
            ec2_instance["State"]["Name"],
        )

        # Execute
        instances = self.adapter.create_instances(num_cpu=2, num_ram=2, timeout=1)

        # Assert
        assert len(instances) == 1
        assert isinstance(instances[0], tuple)
        assert instances[0] == expected_return_value

    def test_list_instances(self) -> None:
        """List all instances."""
        # Prepare
        reservations = self.ec2_mock.describe_instances.return_value["Reservations"]
        ec2_instance = reservations[0]["Instances"][0]

        expected_return_value = (
            ec2_instance["InstanceId"],
            ec2_instance["PrivateIpAddress"],
            ec2_instance["PublicIpAddress"],
            22,
            ec2_instance["State"]["Name"],
        )

        # Execute
        instances = self.adapter.list_instances()

        # Assert
        assert len(instances) == 1
        assert instances[0] == expected_return_value

    def test_terminate_instances(self) -> None:
        """Destroy all instances."""
        # Prepare
        instance_id = "1"
        result = {"TerminatingInstances": [{"CurrentState": {"Name": "shutting-down"}}]}
        self.ec2_mock.terminate_instances.return_value = result

        # Execute
        self.adapter.terminate_instances([instance_id])


if os.getenv("FLOWER_INTEGRATION"):

    class EC2AdapterIntegrationTestCase(unittest.TestCase):
        """Test suite for class EC2Adapter."""

        def setUp(self) -> None:
            """Prepare tests."""
            # Filter false positiv warning
            warnings.filterwarnings(
                "ignore",
                category=ResourceWarning,
                message="unclosed.*<ssl.SSLSocket.*>",
            )

            self.adapter = EC2Adapter(
                image_id="ami-0370b0294d7241341",
                key_name="flower",
                subnet_id="subnet-23da286f",
                security_group_ids=["sg-0dd0f0080bcf86400"],
            )

        def test_workflow(self) -> None:
            """Create, list and terminate an instance."""
            # Execute & Assert
            instances = self.adapter.create_instances(
                num_cpu=2, num_ram=2, num_instance=1, timeout=10
            )
            instances = self.adapter.list_instances()

            assert len(instances) == 1

            self.adapter.terminate_instances([instances[0][0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
