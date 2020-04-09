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
"""Provides an Adapter implementation for AWS EC2."""

import math
import time
from typing import Dict, List, Optional, Tuple

import boto3
from boto3_type_annotations import ec2

from .adapter import Adapter, Instance

EC2RunInstancesResult = Dict[str, List[ec2.Instance]]
EC2DescribeInstancesResult = Dict[str, List[Dict[str, List[ec2.Instance]]]]


class NoMatchingInstanceType(Exception):
    """No matching instance type exists."""


class EC2TerminationFailure(Exception):
    """Something went wrong while terminating EC2 instances.

    EC2 should be manually checked to check what went wrong and
    the instances might need manual shutdown and terminatation.
    """


# List of AWS instance types with
# (instance_type, vCPU, Mem)
INSTANCE_TYPES = [
    ("t3.nano", 2, 0.5),  # 6 CPU Credits/hour
    ("t3.micro", 2, 1),  # 12 CPU Credits/hour
    ("t3.small", 2, 2),  # 24 CPU Credits/hour
    ("t3.medium", 2, 4),  # 24 CPU Credits/hour
    ("m5a.large", 2, 8),
    ("m5a.xlarge", 4, 16),
    ("m5a.2xlarge", 8, 32),
    ("m5a.4xlarge", 16, 64),
]


def find_instance_type(
    num_cpu: int, num_ram: float, instance_types: List[Tuple[str, int, float]]
) -> str:
    """Return the first matching instance type if one exists, raise otherwise."""
    for instance_type in instance_types:
        if instance_type[1] == num_cpu and instance_type[2] == num_ram:
            return instance_type[0]

    raise NoMatchingInstanceType


def flatten_reservations(
    reservations: EC2DescribeInstancesResult,
) -> List[ec2.Instance]:
    """Extract instances from reservations returned by a call to describe_instances."""
    instances: List[ec2.Instance] = []

    # Flatten list of lists
    for ins in [res["Instances"] for res in reservations["Reservations"]]:
        instances += ins

    return instances


def are_all_instances_running(instances: List[ec2.Instance]) -> bool:
    """Return True if all instances are running."""
    for ins in instances:
        if ins["State"]["Name"] != "running":
            return False

    return True


class EC2Adapter(Adapter):
    """Adapter for AWS EC2."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image_id: str,
        key_name: str,
        subnet_id: str,
        security_group_ids: List[str],
        tags: List[str],
        boto_ec2_client: Optional[boto3.session.Session] = None,
    ):
        self.image_id = image_id
        self.key_name = key_name
        self.subnet_id = subnet_id
        self.security_group_ids = security_group_ids
        self.tags = tags
        self.tag_specifications = [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": tag} for tag in tags],
            }
        ]

        self.ec2 = boto3.client("ec2") if boto_ec2_client is None else boto_ec2_client

    # pylint: disable=too-many-arguments
    def create_instances(
        self,
        num_cpu: int,
        num_ram: float,
        timeout: int,
        num_instances: int = 1,
        commands: Optional[List[str]] = None,
    ) -> List[Instance]:
        """Create one or more EC2 instance(s) of the same type.

            Args:
                num_cpu (int): Number of instance vCPU (values in ec2_adapter.INSTANCE_TYPES)
                num_ram (int): RAM in GB (values in ec2_adapter.INSTANCE_TYPES)
                timeout (int): Timeout in minutes
                num_instances (int): Number of instances to start if currently available in EC2
                commands ([str]): List of bash commands which will be joined into a single string
                    with "\n" as a seperator.
        """
        # The instance will be set to terminate after stutdown
        # This is a fail safe in case something happens and the instances
        # are not correctly shutdown
        user_data = ["#!/bin/bash", f"sudo shutdown -P {timeout}"]

        if commands:
            user_data += commands

        user_data_str = "\n".join(user_data)

        result: EC2RunInstancesResult = self.ec2.run_instances(
            ImageId=self.image_id,
            # We always want an exact number of instances
            MinCount=num_instances,
            MaxCount=num_instances,
            InstanceType=find_instance_type(num_cpu, num_ram, INSTANCE_TYPES),
            KeyName=self.key_name,
            SubnetId=self.subnet_id,
            SecurityGroupIds=self.security_group_ids,
            TagSpecifications=self.tag_specifications,
            InstanceInitiatedShutdownBehavior="terminate",
            UserData=user_data_str,
        )

        instance_ids = [ins["InstanceId"] for ins in result["Instances"]]

        return self.list_instances(instance_ids=instance_ids)

    def list_instances(
        self, instance_ids: Optional[List[str]] = None
    ) -> List[Instance]:
        """List all instances with tags belonging to this adapter.

        Args:
            instance_ids ([str[]]): If provided, filter by instance_ids
        """

        if instance_ids is None:
            instance_ids = []

        for i in range(5):
            result: EC2DescribeInstancesResult = self.ec2.describe_instances(
                InstanceIds=instance_ids,
                Filters=[{"Name": "tag:Name", "Values": self.tags}],
            )

            instances = flatten_reservations(result)

            # Block until all instances have state running as otherwise
            # the public IP address will not be available
            if are_all_instances_running(instances):
                break

            secs = int(math.pow(2, i))
            print(f"Waiting for all instances to be ready... Retry in {secs}s")
            time.sleep(secs)

        # Return empty list and terminate every instance which is in
        # any state (e.g. pending, running, etc.)
        if not are_all_instances_running(instances):
            self.terminate_all_instances()
            return []

        instances = [
            (
                ins["InstanceId"],
                ins["PrivateIpAddress"],
                ins["PublicIpAddress"],
                ins["State"]["Name"],
            )
            for ins in instances
        ]

        return instances

    def terminate_instances(self, instance_ids: List[str]) -> None:
        """Terminate instances.

        Will raise an error if something goes wrong.
        """
        res = self.ec2.terminate_instances(InstanceIds=instance_ids)

        for tin in res["TerminatingInstances"]:
            if tin["CurrentState"]["Name"] != "shutting-down":
                raise EC2TerminationFailure

    def terminate_all_instances(self) -> None:
        """Terminate all instances.

        Will raise an error if something goes wrong.
        """
        result: EC2DescribeInstancesResult = self.ec2.describe_instances(
            Filters=[{"Name": "tag:Name", "Values": self.tags}],
        )

        instances = flatten_reservations(result)
        instance_ids = [ins["InstanceId"] for ins in instances]

        if not instance_ids:
            return

        self.terminate_instances(instance_ids)
