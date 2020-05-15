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

from flower_benchmark.tf_fashion_mnist.settings import SETTINGS, get_setting
from flower_ops.cluster import Cluster
from flower_ops.compute.docker_adapter import DockerAdapter

OPS_INI_PATH = path.normpath(
    f"{path.dirname(path.realpath(__file__))}/../../../.flower_ops"
)

# Read config file and extract all values which are needed further down.
CONFIG = configparser.ConfigParser()
CONFIG.read(OPS_INI_PATH)


def logserver_command() -> str:
    """Run log server."""
    return f"screen -d -m python3.7 -m flower_logserver"


# pylint: disable=too-many-arguments
def server_command(
    log_host: str,
    rounds: int,
    sample_fraction: float,
    min_sample_size: int,
    min_num_clients: int,
    training_round_timeout: int,
) -> str:
    """Build command to run server."""
    return f"screen -d -m python3.7 -m \
flower_benchmark.tf_fashion_mnist.server \
--log_host={log_host} \
--rounds={rounds} \
--sample_fraction={sample_fraction} \
--min_sample_size={min_sample_size} \
--min_num_clients={min_num_clients} \
--training_round_timeout={training_round_timeout}"


def client_command(
    log_host: str,
    grpc_server_address: str,
    cid: str,
    partition: int,
    num_partitions: int,
) -> str:
    """Build command to run client."""
    return f"screen -d -m python3.7 -m \
flower_benchmark.tf_fashion_mnist.client \
--log_host={log_host} \
--grpc_server_address={grpc_server_address} \
--grpc_server_port=8080 \
--cid={cid} \
--partition={partition} \
--clients={num_partitions} \
"


def watch_and_shutdown_command(keyword: str) -> str:
    """Return command which shuts down the instance after no benchmark is running anymore."""
    return (
        f"screen -d -m bash -c 'while [[ $(ps a | grep {keyword}) ]]; "
        + "do sleep 1; done; kill 1'"
    )


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
) -> None:
    """Run benchmark."""
    wheel_filename = CONFIG.get("paths", "wheel_filename")
    wheel_local_path = (
        path.expanduser(CONFIG.get("paths", "wheel_dir")) + wheel_filename
    )
    wheel_remote_path = "/root/" + wheel_filename

    docker_adapter = DockerAdapter(name="flower")
    cluster = Cluster(
        adapter=docker_adapter,
        ssh_credentials=("root", path.expanduser(CONFIG.get("ssh", "private_key"))),
        specs={"logserver": (1, 1, 1), "server": (2, 2, 1), "clients": (2, 4, 1)},
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
    logserver_id, logserver_private_ip, _, _, _ = cluster.instances["logserver"][0]
    cluster.exec(
        logserver_id, logserver_command(),
    )

    # Start flower server on flower server instances
    server_id, server_private_ip, _, _, _ = cluster.instances["server"][0]
    cluster.exec(server_id, f"python3.7 -m flower_benchmark.tf_fashion_mnist.download")
    cluster.exec(
        server_id,
        server_command(
            log_host=f"{logserver_private_ip}:8081",
            rounds=rounds,
            sample_fraction=sample_fraction,
            min_sample_size=min_sample_size,
            min_num_clients=min_num_clients,
            training_round_timeout=training_round_timeout,
        ),
    )

    # Start flower clients
    client_id, _, _, _, _ = cluster.instances["clients"][0]
    cluster.exec(client_id, f"python3.7 -m flower_benchmark.tf_fashion_mnist.download")
    for i in range(0, int(num_clients)):
        cluster.exec(
            client_id,
            client_command(
                log_host=f"{logserver_private_ip}:8081",
                grpc_server_address=server_private_ip,
                cid=str(i),
                partition=i,
                num_partitions=num_clients,
            ),
        )

    # Shutdown server and client instance after 10min if not at least
    # one flower_benchmark is running it
    cluster.exec(logserver_id, watch_and_shutdown_command("[f]lower_logserver"))
    cluster.exec(server_id, watch_and_shutdown_command("[f]lower_benchmark"))
    cluster.exec(client_id, watch_and_shutdown_command("[f]lower_benchmark"))

    print(cluster.instances)


def main() -> None:
    """Start server and train `--rounds` number of rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help=f"Name of setting to run. Possible options: {list(SETTINGS.keys())}",
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
    )


if __name__ == "__main__":
    main()
