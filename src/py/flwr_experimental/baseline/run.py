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
"""Execute Fashion-MNIST baseline locally in Docker."""


import argparse
import concurrent.futures
import configparser
from logging import INFO
from os import path
from time import strftime
from typing import List, Optional

import flwr_experimental.baseline.tf_cifar.settings as tf_cifar_settings
import flwr_experimental.baseline.tf_fashion_mnist.settings as tf_fashion_mnist_settings
import flwr_experimental.baseline.tf_hotkey.settings as tf_hotkey_settings
from flwr.common.logger import configure, log
from flwr_experimental.baseline import command
from flwr_experimental.ops.cluster import Cluster, Instance
from flwr_experimental.ops.compute.adapter import Adapter
from flwr_experimental.ops.compute.docker_adapter import DockerAdapter
from flwr_experimental.ops.compute.ec2_adapter import EC2Adapter

OPS_INI_PATH = path.normpath(
    f"{path.dirname(path.realpath(__file__))}/../../../.flower_ops"
)

# Read config file and extract all values which are needed further down.
CONFIG = configparser.ConfigParser()
CONFIG.read(OPS_INI_PATH)

WHEEL_FILENAME = CONFIG.get("paths", "wheel_filename")
WHEEL_LOCAL_PATH = path.expanduser(CONFIG.get("paths", "wheel_dir")) + WHEEL_FILENAME

DOCKER_PRIVATE_KEY = path.realpath(path.dirname(__file__) + "/../../../docker/ssh_key")


def now() -> str:
    """Return current date and time as string."""
    return strftime("%Y%m%dT%H%M%S")


def configure_cluster(
    adapter: str, instances: List[Instance], baseline: str, setting: str
) -> Cluster:
    """Return configured compute cluster."""
    adapter_instance: Optional[Adapter] = None
    private_key: Optional[str] = None

    if adapter == "docker":
        adapter_instance = DockerAdapter()
        user = "root"
        private_key = DOCKER_PRIVATE_KEY
    elif adapter == "ec2":
        adapter_instance = EC2Adapter(
            image_id=CONFIG.get("aws", "image_id"),
            key_name=path.expanduser(CONFIG.get("aws", "key_name")),
            subnet_id=CONFIG.get("aws", "subnet_id"),
            security_group_ids=CONFIG.get("aws", "security_group_ids").split(","),
            tags=[
                ("Purpose", "flwr_experimental.baseline"),
                ("Baseline Name", baseline),
                ("Baseline Setting", setting),
            ],
        )
        user = "ubuntu"
        private_key = path.expanduser(CONFIG.get("ssh", "private_key"))
    else:
        raise Exception(f"Adapter of type {adapter} does not exist.")

    cluster = Cluster(
        adapter=adapter_instance,
        ssh_credentials=(user, private_key),
        instances=instances,
        timeout=60,
    )

    return cluster


# pylint: disable=too-many-arguments, too-many-locals
def run(baseline: str, setting: str, adapter: str) -> None:
    """Run baseline."""
    print(f"Starting baseline with {setting} settings.")

    wheel_remote_path = (
        f"/root/{WHEEL_FILENAME}"
        if adapter == "docker"
        else f"/home/ubuntu/{WHEEL_FILENAME}"
    )

    if baseline == "tf_cifar":
        settings = tf_cifar_settings.get_setting(setting)
    elif baseline == "tf_fashion_mnist":
        settings = tf_fashion_mnist_settings.get_setting(setting)
    elif baseline == "tf_hotkey":
        settings = tf_hotkey_settings.get_setting(setting)
    else:
        raise Exception("Setting not found.")

    # Get instances and add a logserver to the list
    instances = settings.instances
    instances.append(
        Instance(name="logserver", group="logserver", num_cpu=2, num_ram=2)
    )

    # Configure cluster
    log(INFO, "(1/9) Configure cluster.")
    cluster = configure_cluster(adapter, instances, baseline, setting)

    # Start the cluster; this takes some time
    log(INFO, "(2/9) Start cluster.")
    cluster.start()

    # Upload wheel to all instances
    log(INFO, "(3/9) Upload wheel to all instances.")
    cluster.upload_all(WHEEL_LOCAL_PATH, wheel_remote_path)

    # Install the wheel on all instances
    log(INFO, "(4/9) Install wheel on all instances.")
    cluster.exec_all(command.install_wheel(wheel_remote_path))

    # Download datasets in server and clients
    log(INFO, "(5/9) Download dataset on server and clients.")
    cluster.exec_all(
        command.download_dataset(baseline=baseline), groups=["server", "clients"]
    )

    # Start logserver
    log(INFO, "(6/9) Start logserver.")
    logserver = cluster.get_instance("logserver")
    cluster.exec(
        logserver.name,
        command.start_logserver(
            logserver_s3_bucket=CONFIG.get("aws", "logserver_s3_bucket"),
            logserver_s3_key=f"{baseline}_{setting}_{now()}.log",
        ),
    )

    # Start Flower server on Flower server instances
    log(INFO, "(7/9) Start server.")
    cluster.exec(
        "server",
        command.start_server(
            log_host=f"{logserver.private_ip}:8081", baseline=baseline, setting=setting,
        ),
    )

    # Start Flower clients
    log(INFO, "(8/9) Start clients.")
    server = cluster.get_instance("server")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Start the load operations and mark each future with its URL
        concurrent.futures.wait(
            [
                executor.submit(
                    cluster.exec,
                    client_setting.instance_name,
                    command.start_client(
                        log_host=f"{logserver.private_ip}:8081",
                        server_address=f"{server.private_ip}:8080",
                        baseline=baseline,
                        setting=setting,
                        cid=client_setting.cid,
                    ),
                )
                for client_setting in settings.clients
            ]
        )

    # Shutdown server and client instance after 10min if not at least one Flower
    # process is running it
    log(INFO, "(9/9) Start shutdown watcher script.")
    cluster.exec_all(command.watch_and_shutdown("flower", adapter))

    # Give user info how to tail logfile
    private_key = (
        DOCKER_PRIVATE_KEY
        if adapter == "docker"
        else path.expanduser(CONFIG.get("ssh", "private_key"))
    )

    log(
        INFO,
        "If you would like to tail the central logfile run:\n\n\t%s\n",
        command.tail_logfile(adapter, private_key, logserver),
    )


def main() -> None:
    """Start Flower baseline."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["tf_cifar", "tf_fashion_mnist", "tf_hotkey"],
        help="Name of baseline name to run.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=list(
            set(
                list(tf_cifar_settings.SETTINGS.keys())
                + list(tf_fashion_mnist_settings.SETTINGS.keys())
                + list(tf_hotkey_settings.SETTINGS.keys())
            )
        ),
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

    # Configure logger
    configure(f"flower_{args.baseline}_{args.setting}")

    run(baseline=args.baseline, setting=args.setting, adapter=args.adapter)


if __name__ == "__main__":
    main()
