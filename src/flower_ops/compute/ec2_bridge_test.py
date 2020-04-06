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
"""Tests EC2Bridge."""
import time
import unittest
import warnings

import boto3

from .ec2_bridge import EC2Bridge

IMAGE_ID = "ami-0b418580298265d5c"
INSTANCE_TYPE = "t3.nano"
KEY_NAME = "flower"
SUBNET_ID = "subnet-23da286f"
SECURITY_GROUP_IDS = ["sg-0dd0f0080bcf86400"]
TAGS = [f"test_case_{int(time.time())}"]
USER_DATA = "#!/bin/bash\nsudo shutdown -P 1"


def create_instance() -> str:
    "Create EC2 instance and return ID."
    ec2 = boto3.client("ec2")
    res = ec2.run_instances(
        ImageId=IMAGE_ID,
        MinCount=1,
        MaxCount=1,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SubnetId=SUBNET_ID,
        SecurityGroupIds=SECURITY_GROUP_IDS,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": tag} for tag in TAGS],
            }
        ],
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=USER_DATA,
    )

    instance_id: str = res["Instances"][0]["InstanceId"]

    return instance_id


class EC2BridgeTestCase(unittest.TestCase):
    """Test suite for class EC2Bridge.

    This is an integration test with the boto3 API.
    """

    def setUp(self) -> None:
        """Create an instance."""
        # This warning seems to be a false positiv
        # https://github.com/boto/boto3/issues/454
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
        )

        self.bridge = EC2Bridge(
            image_id="ami-0b418580298265d5c",
            key_name="flower",
            subnet_id="subnet-23da286f",
            security_group_ids=["sg-0dd0f0080bcf86400"],
            tags=TAGS,
        )

    def test_create_instances(self):
        """Create and start an instance."""
        # Execute
        instances = self.bridge.create_instances(num_cpu=2, num_ram=0.5, timeout=1)

        # Assert
        for ins in instances:
            assert isinstance(ins, tuple)
            assert isinstance(ins[0], str)
            assert isinstance(ins[1], str)

    def test_terminate_instances(self):
        """Destroy all instances."""
        # Prepare
        instance_id = create_instance()
        self.bridge.terminate_instances([instance_id])


if __name__ == "__main__":
    unittest.main(verbosity=2)
