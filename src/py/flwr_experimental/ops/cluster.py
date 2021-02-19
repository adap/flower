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
"""Implements compute classes for EC2."""
import concurrent.futures
from contextlib import contextmanager
from itertools import groupby
from logging import DEBUG, ERROR
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union, cast

from paramiko.client import MissingHostKeyPolicy, SSHClient
from paramiko.sftp_attr import SFTPAttributes

from flwr.common.logger import log

from .compute.adapter import Adapter
from .instance import Instance

ExecInfo = Tuple[List[str], List[str]]


class StartFailed(Exception):
    """Raised when cluster could not start."""


class InstanceIdNotFound(Exception):
    """Raised when there was no instance with given id."""


class InstanceMismatch(Exception):
    """Raised when instances passed to create_instances do not have the same
    values for RAM or CPU."""


class IgnoreHostKeyPolicy:
    """Policy for accepting any unknown host key.

    This is used by `paramiko.client.SSHClient`.
    """

    # pylint: disable=no-self-use, unused-argument
    def missing_host_key(self, client: SSHClient, hostname: str, key: str) -> None:
        """Simply return to ignore the host key.

        As we create and destroy machines quite regularly and don't
        reuse them we will not store the host key in the local system to
        avoid pollution the local known_hosts file.
        """
        return None


SSHCredentials = Tuple[str, str]  # username, key_filename


@contextmanager
def ssh_connection(
    instance: Instance, ssh_credentials: SSHCredentials
) -> Iterator[SSHClient]:
    """Connect to server and yield SSH client."""
    username, key_filename = ssh_credentials

    instance_ssh_port: int = cast(int, instance.ssh_port)
    ignore_host_key_policy: Union[
        Type[MissingHostKeyPolicy], MissingHostKeyPolicy
    ] = cast(
        Union[Type[MissingHostKeyPolicy], MissingHostKeyPolicy], IgnoreHostKeyPolicy
    )

    client = SSHClient()
    client.set_missing_host_key_policy(ignore_host_key_policy)
    client.connect(
        hostname=str(instance.public_ip),
        port=instance_ssh_port,
        username=username,
        key_filename=key_filename,
    )

    yield client

    client.close()


def create_instances(adapter: Adapter, instances: List[Instance], timeout: int) -> None:
    """Start instances and set props of each instance.

    Fails if CPU and RAM of instances are not all the same.
    """
    if not all(
        [
            ins.num_cpu == instances[0].num_cpu and ins.num_ram == instances[0].num_ram
            for ins in instances
        ]
    ):
        raise InstanceMismatch(
            "Values of num_cpu and num_ram have to be equal for all instances."
        )

    # As checked before that each instance has the same num_cpu and num_ram
    # we can just take the values from the first => instances[0]
    adapter_instances = adapter.create_instances(
        num_cpu=instances[0].num_cpu,
        num_ram=instances[0].num_ram,
        num_instance=len(instances),
        gpu=instances[0].gpu,
        timeout=timeout,
    )

    for i, adp_ins in enumerate(adapter_instances):
        instance_id, private_ip, public_ip, ssh_port, state = adp_ins

        instances[i].instance_id = instance_id
        instances[i].private_ip = private_ip
        instances[i].public_ip = public_ip
        instances[i].ssh_port = ssh_port
        instances[i].state = state


def group_instances_by_specs(instances: List[Instance]) -> List[List[Instance]]:
    """Group instances by num_cpu and num_ram."""
    groups: List[List[Instance]] = []
    keyfunc = lambda ins: f"{ins.num_cpu}-{ins.num_ram}"
    instances = sorted(instances, key=keyfunc)
    for _, group in groupby(instances, keyfunc):
        groups.append(list(group))
    return groups


class Cluster:
    """Compute enviroment independend compute cluster."""

    def __init__(
        self,
        adapter: Adapter,
        ssh_credentials: SSHCredentials,
        instances: List[Instance],
        timeout: int,
    ):
        """Create cluster.

        Args:
            timeout (int): Minutes after which the machine will shutdown and terminate.
                           This is a safety mechanism to avoid run aways cost. The user should still
                           make sure to monitor the progress in case this mechanism fails.

        Example:
            To start two groups of instances where the first one has one instance and the
            second one has two instances you might define the following list of instances:

            instances = [
                Instance(name='server', group='server', num_cpu=2, num_ram=1.0),
                Instance(name='client_0', group='clients', num_cpu=4, num_ram=16.0),
                Instance(name='client_1', group='clients', num_cpu=4, num_ram=16.0),
            ]

            Depending on the adapter used not every combination of vCPU and RAM might be available.
        """
        instance_names = {ins.name for ins in instances}
        assert len(instance_names) == len(instances), "Instance names must be unique."

        self.adapter = adapter
        self.ssh_credentials = ssh_credentials
        self.instances = instances
        self.timeout = timeout

    def get_instance(self, instance_name: str) -> Instance:
        """Return instance by instance_name."""
        for ins in self.instances:
            if ins.name == instance_name:
                return ins

        raise InstanceIdNotFound()

    def get_instance_names(self, groups: Optional[List[str]] = None) -> List[str]:
        """Return a list of all instance names."""
        return [
            ins.name for ins in self.instances if groups is None or ins.group in groups
        ]

    def start(self) -> None:
        """Start the instance."""
        instance_groups = group_instances_by_specs(self.instances)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    create_instances, self.adapter, instance_group, self.timeout
                )
                for instance_group in instance_groups
            ]
            concurrent.futures.wait(futures)

            try:
                for future in futures:
                    future.result()
            # pylint: disable=broad-except
            except Exception as exc:
                log(
                    ERROR,
                    "Failed to start the cluster completely. Shutting down...",
                )
                log(ERROR, exc)

                for future in futures:
                    future.cancel()

                self.terminate()
                raise StartFailed() from exc

        for ins in self.instances:
            log(DEBUG, ins)

    def terminate(self) -> None:
        """Terminate all instances and shutdown cluster."""
        self.adapter.terminate_all_instances()

    def upload(
        self, instance_name: str, local_path: str, remote_path: str
    ) -> SFTPAttributes:
        """Upload local file to remote instance."""
        instance = self.get_instance(instance_name)

        with ssh_connection(instance, self.ssh_credentials) as client:
            sftp = client.open_sftp()

            if sftp is not None:
                sftp_file_attributes = sftp.put(local_path, remote_path)

        return sftp_file_attributes

    def upload_all(
        self, local_path: str, remote_path: str
    ) -> Dict[str, SFTPAttributes]:
        """Upload file to all instances."""
        results: Dict[str, SFTPAttributes] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            future_to_result = {
                executor.submit(
                    self.upload, instance_name, local_path, remote_path
                ): instance_name
                for instance_name in self.get_instance_names()
            }

            for future in concurrent.futures.as_completed(future_to_result):
                instance_name = future_to_result[future]
                try:
                    results[instance_name] = future.result()
                # pylint: disable=broad-except
                except Exception as exc:
                    log(ERROR, (instance_name, exc))

        return results

    def exec(self, instance_name: str, command: str) -> ExecInfo:
        """Run command on instance and return stdout."""
        log(DEBUG, "Exec on %s: %s", instance_name, command)

        instance = self.get_instance(instance_name)

        with ssh_connection(instance, self.ssh_credentials) as client:
            _, stdout, stderr = client.exec_command(command)
            lines_stdout = stdout.readlines()
            lines_stderr = stderr.readlines()

        print(lines_stdout, lines_stderr)

        return lines_stdout, lines_stderr

    def exec_all(
        self, command: str, groups: Optional[List[str]] = None
    ) -> Dict[str, ExecInfo]:
        """Run command on all instances.

        If provided filter by group.
        """
        instance_names = self.get_instance_names(groups)

        results: Dict[str, ExecInfo] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            future_to_result = {
                executor.submit(self.exec, instance_name, command): instance_name
                for instance_name in instance_names
            }

            for future in concurrent.futures.as_completed(future_to_result):
                instance_name = future_to_result[future]
                try:
                    results[instance_name] = future.result()
                # pylint: disable=broad-except
                except Exception as exc:
                    log(ERROR, (instance_name, exc))

        return results
