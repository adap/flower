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
from typing import Optional

from flower_benchmark.tf_fashion_mnist import command
from flower_benchmark.tf_fashion_mnist.settings import SETTINGS, get_setting
from flower_ops.cluster import Cluster
from flower_ops.compute.adapter import Adapter
from flower_ops.compute.docker_adapter import DockerAdapter
from flower_ops.compute.ec2_adapter import EC2Adapter

OPS_INI_PATH = path.normpath(
    f"{path.dirname(path.realpath(__file__))}/../../../.flower_ops"
)

# Read config file and extract all values which are needed further down.
CONFIG = configparser.ConfigParser()
CONFIG.read(OPS_INI_PATH)


def configure_cluster(adapter: str, benchmark_name: str) -> Cluster:
    """Return configured compute cluster."""
    adapter_instance: Optional[Adapter] = None
    private_key: Optional[str] = None

    if adapter == "docker":
        adapter_instance = DockerAdapter()
        user = "root"
        private_key = f"{path.dirname(path.realpath(__file__))}/../../../docker/ssh_key"
    elif adapter == "ec2":
        adapter_instance = EC2Adapter(
            image_id=CONFIG.get("aws", "image_id"),
            key_name=path.expanduser(CONFIG.get("aws", "key_name")),
            subnet_id=CONFIG.get("aws", "subnet_id"),
            security_group_ids=CONFIG.get("aws", "security_group_ids").split(","),
            tags=[("Purpose", "flower_benchmark"), ("Benchmark Name", benchmark_name)],
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
def run(
    # Global paramters
    rounds: int,
    # Server paramters
    sample_fraction: float,
    min_sample_size: int,
    min_num_clients: int,
    training_round_timeout: int,
    # Client parameters
    num_clients: int,
    dry_run: bool,
    # Environment
    logserver_s3_bucket: Optional[str] = None,
    adapter: str = "docker",
    benchmark_name: str = "fashion_mnist",
) -> None:
    """Run benchmark."""
    wheel_filename = CONFIG.get("paths", "wheel_filename")
    wheel_local_path = (
        path.expanduser(CONFIG.get("paths", "wheel_dir")) + wheel_filename
    )
    wheel_remote_path = (
        f"/root/{wheel_filename}"
        if adapter == "docker"
        else f"/home/ubuntu/{wheel_filename}"
    )

    # Configure cluster
    cluster = configure_cluster(adapter, benchmark_name)

    # Start the cluster; this takes some time
    cluster.start()

    # Upload wheel to all instances
    cluster.upload_all(wheel_local_path, wheel_remote_path)

    # Install the wheel on all instances
    cluster.exec_all(command.install_wheel(wheel_remote_path))

    # An instance is a tuple of the following values
    # (InstanceId, PrivateIpAddress, PublicIpAddress, State)
    logserver_id, logserver_private_ip, _, _, _ = cluster.instances["logserver"][0]
    cluster.exec(
        logserver_id,
        command.start_logserver(
            logserver_s3_bucket=logserver_s3_bucket,
            logserver_s3_key=f"{benchmark_name}.log",
        ),
    )

    # Start Flower server on Flower server instances
    server_id, server_private_ip, _, _, _ = cluster.instances["server"][0]
    cluster.exec(server_id, command.download_dataset())
    cluster.exec(
        server_id,
        command.start_server(
            log_host=f"{logserver_private_ip}:8081",
            rounds=rounds,
            sample_fraction=sample_fraction,
            min_sample_size=min_sample_size,
            min_num_clients=min_num_clients,
            training_round_timeout=training_round_timeout,
        ),
    )

    # Start Flower clients
    client_instances = cluster.instances["clients"]

    # Make dataset locally available
    cluster.exec(client_instances[0][0], command.download_dataset())
    cluster.exec(client_instances[1][0], command.download_dataset())

    for i in range(0, num_clients):
        client_id = (
            client_instances[0][0]
            if i < int(num_clients / 2)
            else client_instances[1][0]
        )
        cluster.exec(
            client_id,
            command.start_client(
                log_host=f"{logserver_private_ip}:8081",
                grpc_server_address=server_private_ip,
                cid=str(i),
                partition=i,
                num_partitions=num_clients,
                dry_run=dry_run,
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
        "--setting",
        type=str,
        required=True,
        help=f"Name of setting to run. Possible options: {list(SETTINGS.keys())}.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        choices=["docker", "ec2"],
        help="Set adapter to be used.",
    )
    parser.add_argument(
        "--logserver_s3_bucket",
        type=str,
        help="S3 bucket where the logfile should be uploaded to.",
    )
    args = parser.parse_args()

    setting = get_setting(args.setting)

    print("Starting benchmark with the following settings:")
    print(setting)

    run(
        rounds=setting.rounds,
        sample_fraction=setting.sample_fraction,
        min_sample_size=setting.min_sample_size,
        min_num_clients=setting.min_num_clients,
        training_round_timeout=setting.training_round_timeout,
        num_clients=setting.num_clients,
        dry_run=setting.dry_run,
        adapter=args.adapter,
        benchmark_name=args.setting,
        logserver_s3_bucket=args.logserver_s3_bucket,
    )


if __name__ == "__main__":
    main()
