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
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

from paramiko.client import SSHClient
from paramiko.sftp_attr import SFTPAttributes

from .compute.adapter import Adapter, Instance

Spec = Tuple[int, float, int]  # (num_cpu, num_ram, num_instances)


class InstanceIdNotFound(Exception):
    """Raised when there was no instance with given id."""


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


@contextmanager
def ssh_connection(
    instance: Instance, ssh_credentials: SSHCredentials
) -> Iterator[SSHClient]:
    """Connect to server and yield SSH client."""
    _, _, public_ip, _ = instance
    username, key_filename = ssh_credentials

    client = SSHClient()
    client.set_missing_host_key_policy(IgnoreHostKeyPolicy)
    client.connect(hostname=public_ip, username=username, key_filename=key_filename)

    yield client

    client.close()


class Cluster:
    """Compute enviroment independend compute cluster."""

    def __init__(
        self,
        adapter: Adapter,
        ssh_credentials: SSHCredentials,
        specs: Dict[str, Spec],
        timeout: int = 10,
    ):
        """Create cluster.

        Example:
            To start two groups of instances where the first one has one instance and the
            second one has two instances you might define the spec as following:

            specs = {
                group_name: (vCPU count, RAM in GB, number of instances)
            }

            e.g.

            specs = {
                 # First group named server with total 3 instances.
                 # The first group has 2 vCPU and 1.0 GB RAM per instance consists of 1 instance
                'server': (2, 1.0, 1),
                # The second group has 4 vCPU and 2.0 GB RAM per instance consists of 2 instances
                'clients': (4, 2.0, 2),
            }

            Depending on the adapter used not every combination of vCPU and RAM might be available.
        """
        self.adapter = adapter
        self.ssh_credentials = ssh_credentials
        self.specs = specs
        self.timeout = timeout

        self.instances: Dict[str, List[Instance]] = {}

    def get_instance(self, instance_id: str) -> Instance:
        """Return instance by instance_id."""
        for ins_group in self.instances.values():
            for ins in ins_group:
                if ins[0] == instance_id:
                    return ins

        # If instance_id could not be found raise an exception
        raise InstanceIdNotFound()

    def get_instance_ids(self) -> List[str]:
        """Return a list of all instance_ids."""
        ids: List[str] = []
        for ins_group in self.instances.values():
            for ins in ins_group:
                ids.append(ins[0])
        return ids

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

    def upload(
        self, instance_id: str, local_path: str, remote_path: str
    ) -> SFTPAttributes:
        """Upload local file to remote instance."""
        instance = self.get_instance(instance_id)

        with ssh_connection(instance, self.ssh_credentials) as client:
            sftp = client.open_sftp()
            sftp_file_attributes = sftp.put(local_path, remote_path)

        return sftp_file_attributes

    def upload_all(self, local_path: str, remote_path: str) -> List[SFTPAttributes]:
        """Upload file to all instances."""
        return [
            self.upload(instance_id, local_path, remote_path)
            for instance_id in self.get_instance_ids()
        ]

    def exec(self, instance_id: str, command: str) -> Tuple[str, str]:
        """Run command on instance and return stdout."""
        print(f"Exec on {instance_id}: {command}")

        instance = self.get_instance(instance_id)

        with ssh_connection(instance, self.ssh_credentials) as client:
            _, stdout, stderr = client.exec_command(command)
            stdout = stdout.readlines()
            stderr = stderr.readlines()

        return stdout, stderr

    def exec_all(self, command: str) -> List[Tuple[str, str]]:
        """Run command on all instances and return List of (stdout, stderr) tuples."""
        return [
            self.exec(instance_id, command) for instance_id in self.get_instance_ids()
        ]

    def exec_group(self, group: str, command: str) -> List[Tuple[str, str]]:
        """Run command on all instances in group and return List of (stdout, stderr) tuples."""
        return [self.exec(ins[0], command) for ins in self.instances[group]]
