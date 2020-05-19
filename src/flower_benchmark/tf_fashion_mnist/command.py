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
"""Provides functions to construct various flower CLI commands."""

from typing import Optional


def install_wheel(wheel_remote_path: str) -> str:
    "Return install command for wheel."
    return f"python3.7 -m pip install {wheel_remote_path}"


def start_logserver(
    logserver_s3_bucket: Optional[str] = None, logserver_s3_key: Optional[str] = None
) -> str:
    """Return command to run logserver."""
    cmd = "screen -L -Logfile logserver.log -d -m python3.7 -m flower_logserver"

    if logserver_s3_bucket is not None and logserver_s3_key is not None:
        cmd += f" --s3_bucket={logserver_s3_bucket}" + f" --s3_key={logserver_s3_key}"

    return cmd


# pylint: disable=too-many-arguments
def start_server(
    log_host: str,
    rounds: int,
    sample_fraction: float,
    min_sample_size: int,
    min_num_clients: int,
    training_round_timeout: int,
) -> str:
    """Build command to run server."""
    return (
        "screen -L -Logfile server.log -d -m"
        + " python3.7 -m flower_benchmark.tf_fashion_mnist.server"
        + f" --log_host={log_host}"
        + f" --rounds={rounds}"
        + f" --sample_fraction={sample_fraction}"
        + f" --min_sample_size={min_sample_size}"
        + f" --min_num_clients={min_num_clients}"
        + f" --training_round_timeout={training_round_timeout}"
    )


def start_client(
    log_host: str,
    grpc_server_address: str,
    cid: str,
    partition: int,
    num_partitions: int,
    dry_run: bool,
) -> str:
    """Build command to run client."""
    cmd = (
        f"screen -L -Logfile client_{cid}.log -d -m"
        + " python3.7 -m flower_benchmark.tf_fashion_mnist.client"
        + f" --log_host={log_host}"
        + f" --grpc_server_address={grpc_server_address}"
        + " --grpc_server_port=8080"
        + f" --cid={cid}"
        + f" --partition={partition}"
        + f" --clients={num_partitions}"
    )
    if dry_run:
        cmd += " --dry_run=1"
    return cmd


def download_dataset() -> str:
    "Return command which makes dataset locally available."
    return "python3.7 -m flower_benchmark.tf_fashion_mnist.download"


def watch_and_shutdown(keyword: str, adapter: str) -> str:
    """Return command which shuts down the instance after no benchmark is running anymore."""
    cmd = f"screen -d -m bash -c 'while [[ $(ps a | grep {keyword}) ]]; do sleep 1; done; "

    if adapter == "docker":
        cmd += "kill 1'"
    elif adapter == "ec2":
        # Shutdown after 2 minutes to allow a logged in user
        # to chancel the shutdown manually just in case
        cmd += "sudo shutdown -P 2'"
    else:
        raise Exception("Unknown Adapter")

    return cmd
