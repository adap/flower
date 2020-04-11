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

from typing import Dict, List, Tuple

from paramiko.client import SSHClient

from .compute.adapter import Adapter, Instance, Optional

Spec = Tuple[int, float, int]  # (num_cpu, num_ram, num_instances)


class IgnoreHostKeyPolicy:
    """Policy for accepting any unknown host key. This is used by `paramiko.client.SSHClient`."""

    # pylint: disable=no-self-use, unused-argument
    def missing_host_key(self, client, hostname, key):
        """Simply return to ignore the host key.

        As we create and destroy machines quite regularly and don't reuse them
        we will not store the host key in the local system to avoid pollution the
        local known_hosts file.
        """
        return


SSHCredentials = Tuple[str, str]  # username, key_filename


class Cluster:
    """Compute enviroment independend compute cluster."""

    def __init__(
        self,
        adapter: Adapter,
        ssh_credentials: SSHCredentials,
        specs: Dict[str, Spec],
        timeout: int = 10,
    ):
        self.adapter = adapter
        self.ssh_credentials = ssh_credentials
        self.specs = specs
        self.timeout = timeout

        self.instances: Dict[str, List[Instance]] = {}

    def get_instance(self, instance_id: str) -> Optional[Instance]:
        """Return instance by instance_id."""
        for ins_group in self.instances.values():
            for ins in ins_group:
                if ins[0] == instance_id:
                    return ins

        return None

    def start(self) -> None:
        """Start the instance."""
        # Create Instances
        for group, spec in self.specs.items():
            print(f"Starting group: {group}")

            instances = self.adapter.create_instances(
                num_cpu=spec[0],
                num_ram=spec[1],
                num_instances=spec[2],
                timeout=self.timeout,
            )

            self.instances[group] = instances

    def terminate(self) -> None:
        """Terminate all instances and shutdown cluster."""
        self.adapter.terminate_all_instances()

    def exec(self, instance_id: str, command: str) -> Tuple[str, str]:
        """Run command on instance and return stdout."""
        instance = self.get_instance(instance_id)

        if instance is None:
            raise Exception("Instance not found.")

        _, _, public_ip, _ = instance
        username, key_filename = self.ssh_credentials

        client = SSHClient()
        client.set_missing_host_key_policy(IgnoreHostKeyPolicy)
        client.connect(hostname=public_ip, username=username, key_filename=key_filename)

        _, stdout, stderr = client.exec_command(command)
        stdout = stdout.readlines()
        stderr = stderr.readlines()

        client.close()

        return stdout, stderr
