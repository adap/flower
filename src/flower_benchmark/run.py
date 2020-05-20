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
"""Execute Fashion-MNIST benchmark locally in Docker."""

import argparse
import configparser
from os import path
from time import strftime
from typing import List, Optional

import flower_benchmark.tf_cifar.settings as tf_cifar_settings
import flower_benchmark.tf_fashion_mnist.settings as tf_fashion_mnist_settings
from flower_benchmark import command
from flower_benchmark.setting import ClientSetting
from flower_ops.cluster import Cluster
from flower_ops.compute.adapter import Adapter
from flower_ops.compute.docker_adapter import DockerAdapter
from flower_ops.compute.ec2_adapter import EC2Adapter

OPS_INI_PATH = path.normpath(
    f"{path.dirname(path.realpath(__file__))}/../../.flower_ops"
)

# Read config file and extract all values which are needed further down.
CONFIG = configparser.ConfigParser()
CONFIG.read(OPS_INI_PATH)

WHEEL_FILENAME = CONFIG.get("paths", "wheel_filename")
WHEEL_LOCAL_PATH = path.expanduser(CONFIG.get("paths", "wheel_dir")) + WHEEL_FILENAME


def now() -> str:
    """Return current date and time as string."""
    return strftime("%Y%m%dT%H%M%S")


def configure_cluster(adapter: str, benchmark: str) -> Cluster:
    """Return configured compute cluster."""
    adapter_instance: Optional[Adapter] = None
    private_key: Optional[str] = None

    if adapter == "docker":
        adapter_instance = DockerAdapter()
        user = "root"
        private_key = f"{path.dirname(path.realpath(__file__))}/../../docker/ssh_key"
    elif adapter == "ec2":
        adapter_instance = EC2Adapter(
            image_id=CONFIG.get("aws", "image_id"),
            key_name=path.expanduser(CONFIG.get("aws", "key_name")),
            subnet_id=CONFIG.get("aws", "subnet_id"),
            security_group_ids=CONFIG.get("aws", "security_group_ids").split(","),
            tags=[("Purpose", "flower_benchmark"), ("Benchmark Name", benchmark)],
        )
        user = "ubuntu"
        private_key = path.expanduser(CONFIG.get("ssh", "private_key"))
    else:
        raise Exception(f"Adapter of type {adapter} does not exist.")

    cluster = Cluster(
        adapter=adapter_instance,
        ssh_credentials=(user, private_key),
        specs={"logserver": (2, 2, 1), "server": (4, 16, 1), "clients": (4, 16, 2)},
        timeout=60,
    )

    return cluster


# pylint: disable=too-many-arguments, too-many-locals
def run(benchmark: str, setting: str, adapter: str) -> None:
    """Run benchmark."""
    print(f"Starting benchmark with {setting} settings.")

    wheel_remote_path = (
        f"/root/{WHEEL_FILENAME}"
        if adapter == "docker"
        else f"/home/ubuntu/{WHEEL_FILENAME}"
    )

    client_settings: Optional[List[ClientSetting]] = None

    if benchmark == "tf_cifar":
        client_settings = tf_cifar_settings.get_setting(setting).clients
    elif benchmark == "tf_fashion_mnist":
        client_settings = tf_cifar_settings.get_setting(setting).clients
    else:
        raise Exception("Setting not found.")

    # Configure cluster
    cluster = configure_cluster(adapter, benchmark)

    # Start the cluster; this takes some time
    cluster.start()

    # Upload wheel to all instances
    cluster.upload_all(WHEEL_LOCAL_PATH, wheel_remote_path)

    # Install the wheel on all instances
    cluster.exec_all(command.install_wheel(wheel_remote_path))

    # An instance is a tuple of the following values
    # (InstanceId, PrivateIpAddress, PublicIpAddress, State)
    logserver_id, logserver_private_ip, _, _, _ = cluster.instances["logserver"][0]
    cluster.exec(
        logserver_id,
        command.start_logserver(
            logserver_s3_bucket=CONFIG.get("aws", "logserver_s3_bucket"),
            logserver_s3_key=f"{benchmark}_{setting}_{now()}.log",
        ),
    )

    # Start Flower server on Flower server instances
    server_id, server_private_ip, _, _, _ = cluster.instances["server"][0]
    cluster.exec(server_id, command.download_dataset(benchmark=benchmark))
    cluster.exec(
        server_id,
        command.start_server(
            log_host=f"{logserver_private_ip}:8081",
            benchmark=benchmark,
            setting=setting,
        ),
    )

    # Start Flower clients
    client_instances = cluster.instances["clients"]

    # Make dataset locally available
    cluster.exec_group("clients", command.download_dataset(benchmark=benchmark))

    num_client_processes = len(client_settings)

    for i in range(0, num_client_processes):
        instance_id = (
            client_instances[0][0]
            if i < int(num_client_processes / 2)
            else client_instances[1][0]
        )
        cluster.exec(
            instance_id,
            command.start_client(
                log_host=f"{logserver_private_ip}:8081",
                grpc_server_address=server_private_ip,
                benchmark=benchmark,
                setting=setting,
                index=i,
            ),
        )

    # Shutdown server and client instance after 10min if not at least one Flower
    # process is running it
    cluster.exec_group(
        "logserver", command.watch_and_shutdown("[f]lower_logserver", adapter)
    )
    cluster.exec_group(
        "server", command.watch_and_shutdown("[f]lower_benchmark", adapter)
    )
    cluster.exec_group(
        "clients", command.watch_and_shutdown("[f]lower_benchmark", adapter)
    )

    print(cluster.instances)


def main() -> None:
    """Start Flower benchmark."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["tf_cifar", "tf_fashion_mnist"],
        help="Name of benchmark name to run.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=list(tf_cifar_settings.SETTINGS.keys())
        + list(tf_fashion_mnist_settings.SETTINGS.keys()),
        help="Name of setting to run.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        choices=["docker", "ec2"],
        help="Set adapter to be used.",
    )
    args = parser.parse_args()

    run(benchmark=args.benchmark, setting=args.setting, adapter=args.adapter)


if __name__ == "__main__":
    main()
