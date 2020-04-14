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
"""Execute Fashion-MNIST benchmark on AWS."""

import configparser
from os import path

from flower_ops.cluster import Cluster
from flower_ops.compute.ec2_adapter import EC2Adapter

OPS_INI_PATH = path.normpath(
    f"{path.dirname(path.realpath(__file__))}/../../../.flower_ops"
)

# Read config file and extract all values which are needed further down.
CONFIG = configparser.ConfigParser()
CONFIG.read(OPS_INI_PATH)


def server_command(
    rounds: int, sample_fraction: float, min_sample_size: int, min_num_clients: int
) -> str:
    """Build command to run server."""
    return f"screen -d -m python3.7 -m flower_benchmark.tf_fashion_mnist.server \
            --rounds={rounds} \
            --sample_fraction={sample_fraction} \
            --min_sample_size={min_sample_size} \
            --min_num_clients={min_num_clients}"


def client_command(cid: int, num_clients: int, server_ip: str) -> str:
    """Build command to run client."""
    return f"screen -d -m python3.7 -m flower_benchmark.tf_fashion_mnist.client \
                    --cid={cid} \
                    --partition={cid} \
                    --clients={num_clients} \
                    --grpc_server_address={server_ip} \
                    --grpc_server_port=8080"


def watch_and_shutdown_command() -> str:
    """Return command which shuts down the instance after no benchmark is running anymore."""
    return (
        "screen -d -m bash -c 'while [[ $(ps a | grep flower_benchmark) ]]; "
        + "do sleep 1; done; sudo shutdown -P now'"
    )


def run(
    rounds: int,
    num_clients: int,
    sample_fraction: float,
    min_sample_size: int,
    min_num_clients: int,
) -> None:
    """Run benchmark."""
    wheel_filename = CONFIG.get("paths", "wheel_filename")
    wheel_local_path = (
        path.expanduser(CONFIG.get("paths", "wheel_dir")) + wheel_filename
    )
    wheel_remote_path = "/home/ubuntu/" + wheel_filename

    ec2_adapter = EC2Adapter(
        image_id=CONFIG.get("aws", "image_id"),
        key_name=path.expanduser(CONFIG.get("aws", "key_name")),
        subnet_id=CONFIG.get("aws", "subnet_id"),
        security_group_ids=CONFIG.get("aws", "security_group_ids").split(","),
        tags=[("Purpose", "benchmark"), ("Benchmark Name", "fashion_mnist")],
    )

    cluster = Cluster(
        adapter=ec2_adapter,
        ssh_credentials=("ubuntu", path.expanduser(CONFIG.get("ssh", "private_key"))),
        specs={"server": (2, 2, 1), "clients": (2, 4, 2)},
        timeout=20,
    )

    # Start the cluster; this takes some time
    cluster.start()

    # Upload wheel to all instances
    cluster.upload_all(wheel_local_path, wheel_remote_path)

    # Install the wheel on all instances
    cluster.exec_all(f"python3.7 -m pip install {wheel_remote_path}")

    # An instance is a tuple of the following values
    # (InstanceId, PrivateIpAddress, PublicIpAddress, State)
    server_id, server_private_ip, _, _ = cluster.instances["server"][0]

    # Start flower server on flower server instances
    cluster.exec(
        server_id,
        server_command(rounds, sample_fraction, min_sample_size, min_num_clients),
    )

    client_instances = cluster.instances["clients"]

    # Start flower clients on first client instance
    for i in range(0, int(num_clients / 2)):
        cluster.exec(
            client_instances[0][0],  # first client instance
            client_command(i, num_clients, server_private_ip),
        )

    # Start flower clients on second client instance
    for i in range(int(num_clients / 2), num_clients):
        cluster.exec(
            client_instances[1][0],
            client_command(i, num_clients, server_private_ip),  # second client instance
        )

    # Shutdown any instance after 10min if not at least one flower_benchmark is running it
    cluster.exec_all(watch_and_shutdown_command())


def run_10_clients() -> None:
    """Run 10 clients."""
    run(
        rounds=2,
        num_clients=10,
        sample_fraction=1.0,
        min_sample_size=10,
        min_num_clients=10,
    )


if __name__ == "__main__":
    run_10_clients()
