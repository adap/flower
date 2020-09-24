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
"""Baseline settings for CIFAR-10 scale experiments."""


from typing import List

from flwr_experimental.baseline.common import configure_client_instances
from flwr_experimental.baseline.setting import Baseline, ClientSetting, ServerSetting
from flwr_experimental.ops.cluster import Instance

ROUNDS = 2
SAMPLE_FRACTION = 0.2
LR_INITIAL = 0.01


def get_setting(name: str) -> Baseline:
    """Return appropriate setting."""
    if name not in SETTINGS:
        raise Exception(
            f"Baseline {name} does not exist. Valid settings are: {list(SETTINGS.keys())}"
        )
    return SETTINGS[name]


def get_instance_name(
    instance_names: List[str], num_clients: int, client_index: int
) -> str:
    """Return instance_name."""
    idx = client_index // (num_clients // len(instance_names))
    idx = min([idx, len(instance_names) - 1])
    return instance_names[min(idx, len(instance_names))]


def configure_uniform_clients(
    instance_names: List[str],
    num_clients: int,
    dry_run: bool,
) -> List[ClientSetting]:
    """Configure `num_clients` ClientSetting instances."""
    clients = []
    for i in range(num_clients):
        client = ClientSetting(
            # Set instance on which to run
            instance_name=get_instance_name(instance_names, num_clients, i),
            # Individual
            cid=str(i),
            partition=i,
            delay_factor=0.0,
            # Shared
            iid_fraction=1.0,
            num_clients=num_clients,
            dry_run=dry_run,
        )
        clients.append(client)
    return clients

client_instances_2, client_names_2 = configure_client_instances(
    num_clients=2, num_cpu=2, num_ram=8
)

client_instances_10, client_names_10 = configure_client_instances(
    num_clients=10, num_cpu=2, num_ram=8
)

client_instances_50, client_names_50 = configure_client_instances(
    num_clients=50, num_cpu=2, num_ram=8
)

client_instances_100, client_names_100 = configure_client_instances(
    num_clients=100, num_cpu=2, num_ram=8
)

client_instances_500, client_names_500 = configure_client_instances(
    num_clients=500, num_cpu=2, num_ram=8
)

client_instances_1000, client_names_1000 = configure_client_instances(
    num_clients=1000, num_cpu=2, num_ram=8
)

SETTINGS = {
    "scale-min": Baseline(
        instances=[Instance(name="server", group="server", num_cpu=4, num_ram=16)]
        + client_instances_2,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=2,
            min_num_clients=2,
            sample_fraction=1.0,
            min_sample_size=2,
            training_round_timeout=None,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_uniform_clients(
            instance_names=client_names_2,
            num_clients=2,
            dry_run=False,
        ),
    ),
    "scale-10": Baseline(
        instances=[Instance(name="server", group="server", num_cpu=4, num_ram=16)]
        + client_instances_10,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=ROUNDS,
            min_num_clients=10,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=2,
            training_round_timeout=None,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_uniform_clients(
            instance_names=client_names_10,
            num_clients=10,
            dry_run=False,
        ),
    ),
    "scale-1000": Baseline(
        instances=[Instance(name="server", group="server", num_cpu=4, num_ram=16)]
        + client_instances_1000,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=ROUNDS,
            min_num_clients=1000,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=200,
            training_round_timeout=None,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_uniform_clients(
            instance_names=client_names_1000,
            num_clients=1000,
            dry_run=False,
        ),
    ),
}
