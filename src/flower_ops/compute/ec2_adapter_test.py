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
import time
import unittest
from unittest.mock import MagicMock

from .ec2_adapter import EC2Adapter

IMAGE_ID = "ami-0b418580298265d5c"
INSTANCE_TYPE = "t3.nano"
KEY_NAME = "flower"
SUBNET_ID = "subnet-23da286f"
SECURITY_GROUP_IDS = ["sg-0dd0f0080bcf86400"]
TAGS = [f"test_case_{int(time.time())}"]
USER_DATA = "#!/bin/bash\nsudo shutdown -P 1"


class EC2AdapterTestCase(unittest.TestCase):
    """Test suite for class EC2Adapter."""

    def setUp(self) -> None:
        """Create an instance."""
        self.boto_ec2_client_mock = MagicMock()

        self.bridge = EC2Adapter(
            image_id="ami-0b418580298265d5c",
            key_name="flower",
            subnet_id="subnet-23da286f",
            security_group_ids=["sg-0dd0f0080bcf86400"],
            tags=TAGS,
            boto_ec2_client=self.boto_ec2_client_mock,
        )

    def test_create_instances(self):
        """Create and start an instance."""
        # Prepare
        result = {"Instances": [{"InstanceId": "1", "PrivateIpAddress": "2"}]}
        self.boto_ec2_client_mock.run_instances.return_value = result

        # Execute
        instances = self.bridge.create_instances(num_cpu=2, num_ram=0.5, timeout=1)

        # Assert
        for ins in instances:
            assert isinstance(ins, tuple)
            assert ins[0] == result["Instances"][0]["InstanceId"]
            assert ins[1] == result["Instances"][0]["PrivateIpAddress"]

    def test_terminate_instances(self):
        """Destroy all instances."""
        # Prepare
        instance_id = "1"
        result = {"TerminatingInstances": [{"CurrentState": {"Name": "shutting-down"}}]}
        self.boto_ec2_client_mock.terminate_instances.return_value = result

        # Execute
        self.bridge.terminate_instances([instance_id])


if __name__ == "__main__":
    unittest.main(verbosity=2)
