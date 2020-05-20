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
import concurrent.futures
from contextlib import contextmanager
from logging import ERROR
from typing import Dict, Iterator, List, Optional, Tuple

from paramiko.client import SSHClient
from paramiko.sftp_attr import SFTPAttributes

from flower.logger import log

from .compute.adapter import Adapter, Instance

Spec = Tuple[int, float, int]  # (num_cpu, num_ram, num_instances)


class StartFailedException(Exception):
    """Raised when cluster could not start."""


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
    _, _, public_ip, ssh_port, _ = instance
    username, key_filename = ssh_credentials

    client = SSHClient()
    client.set_missing_host_key_policy(IgnoreHostKeyPolicy)
    client.connect(
        hostname=public_ip, port=ssh_port, username=username, key_filename=key_filename
    )

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

        Args:
            timeout (int): Minutes after which the machine will shutdown and terminate.
                           This is a safety mechanism to avoid run aways cost. The user should still
                           make sure to monitor the progress in case this mechanism fails.

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

        # The keys in the dict are the group names used in self.specs
        self.instances: Dict[str, List[Instance]] = {}

    def get_instance(self, instance_id: str) -> Instance:
        """Return instance by instance_id."""
        for ins_group in self.instances.values():
            for ins in ins_group:
                if ins[0] == instance_id:
                    return ins

        # If instance_id could not be found raise an exception
        raise InstanceIdNotFound()

    def get_instance_ids(self, groups: Optional[List[str]] = None) -> List[str]:
        """Return a list of all instance_ids."""
        ids: List[str] = []

        selected_instances: List[Instance] = []

        if groups is None:
            for instances in self.instances.values():
                selected_instances += instances
        else:
            for group in groups:
                if group in self.instances:
                    selected_instances += self.instances[group]

        for ins in selected_instances:
            ids.append(ins[0])

        return ids

    def start(self) -> None:
        """Start the instance."""
        # Create Instances
        def job(
            group: str, num_cpu: int, num_ram: float, num_instances: int, timeout: int
        ) -> None:
            instances = self.adapter.create_instances(
                num_cpu=num_cpu,
                num_ram=num_ram,
                num_instances=num_instances,
                timeout=timeout,
            )

            self.instances[group] = instances

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(job, group, spec[0], spec[1], spec[2], self.timeout)
                for group, spec in self.specs.items()
            ]
            concurrent.futures.wait(futures)

            try:
                for future in futures:
                    future.result()
            # pylint: disable=broad-except
            except Exception as exc:
                log(
                    ERROR, "Failed to start the cluster completely. Shutting down...",
                )
                log(ERROR, exc)

                for future in futures:
                    future.cancel()

                self.terminate()
                raise StartFailedException()

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

    def exec_all(
        self, command: str, groups: Optional[List[str]] = None
    ) -> Dict[str, Tuple[str, str]]:
        """Run command on all instances. If provided filter by group."""
        instance_ids = (
            self.get_instance_ids() if groups is None else self.get_instance_ids(groups)
        )

        results: Dict[str, Tuple[str, str]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Start the load operations and mark each future with its URL
            future_to_result = {
                executor.submit(self.exec, instance_id, command): instance_id
                for instance_id in instance_ids
            }

            for future in concurrent.futures.as_completed(future_to_result):
                instance_id = future_to_result[future]
                try:
                    results[instance_id] = future.result()
                # pylint: disable=broad-except
                except Exception as exc:
                    log(ERROR, (instance_id, exc))

        return results
