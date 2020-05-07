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

from .cluster import Cluster
from .compute.ec2_adapter import EC2Adapter

IMAGE_ID = "ami-0b418580298265d5c"
KEY_NAME = "flower"
SSH_CREDENTIALS = ("ubuntu", "/Users/tanto/.ssh/flower.pem")
SUBNET_ID = "subnet-23da286f"
SECURITY_GROUP_IDS = ["sg-0dd0f0080bcf86400"]


if os.getenv("FLOWER_INTEGRATION"):

    class ClusterIntegrationTestCase(unittest.TestCase):
        """Integration tests class Cluster.

        This TestCase will not mock anythin and use a live EC2Adapter
        which will be used to provision a single machine and execute a single
        command on it. Afterwards the machines will be shut down.
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
                specs={"server": (2, 0.5, 1)},
                # In case the tearDown fails for some reason the machines
                # should automatically terminate after 10 minutes
                timeout=10,
            )
            self.cluster.start()

        def tearDown(self) -> None:
            self.cluster.terminate()

        def test_exec(self):
            """Execute on all clients."""
            # Prepare
            command = "nproc"
            expected_result = "2\n"

            # Execute
            instance_id = self.cluster.instances["server"][0][0]
            stdout, stderr = self.cluster.exec(instance_id, command)

            # Assert
            assert len(stderr) == 0
            assert len(stdout) == 1
            assert "".join(stdout) == expected_result


if __name__ == "__main__":
    unittest.main(verbosity=2)
