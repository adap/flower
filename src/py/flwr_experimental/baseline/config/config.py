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
"""Provides a variaty of baseline settings for Fashion-MNIST."""


import random
from typing import List, Optional, Tuple

import numpy as np

from flwr_experimental.ops.instance import Instance

# We assume that devices which are older will have at most
# ~80% of the the Samsung Galaxy Note 5 compute performance.
SCORE_MISSING = int(226 * 0.80)

DEVICE_DISTRIBUTION = [
    ("10.0", "Note 10", 0.1612, 729),
    ("Pie 9", "Samsung Galaxy Note 9", 0.374, 607),
    ("Oreo 8.0/8.1", "Samsung Galaxy S8", 0.1129 + 0.0737, 359),
    ("Nougat 7.0/7.1", "Samsung Galaxy S7", 0.0624 + 0.043, 343),
    ("Marshmallow 6.0", "Samsung Galaxy Note 5", 0.0872, 226),
    ("Lollipop 5.1", "Samsung Galaxy Note 4", 0.0484, SCORE_MISSING),
    ("KitKat 4.4", "Samsung Galaxy Note 4", 0.0187, SCORE_MISSING),
    ("Other", "Samsung Galaxy S III", 0.0185, SCORE_MISSING),
]


def sample_delay_factors(
    num_clients: int, max_delay: float, seed: Optional[int]
) -> List[float]:
    """Sample delay factors."""
    np.random.seed(seed)
    # pylint: disable-msg=invalid-name
    ps = [float(p) for p in np.random.rand(num_clients)]
    step_size = max_delay / num_clients
    ds = [(i + 1) * step_size for i in range(num_clients)]
    return [p * d for p, d in zip(ps, ds)]


def sample_real_delay_factors(num_clients: int, seed: int = 2021) -> List[float]:
    """Split list of floats into two buckets."""
    random.seed(seed)

    if num_clients % 2 != 0:
        raise Exception("num_clients has to be divisible by two")

    factors = sorted([get_delay_factor() for _ in range(num_clients)])

    buckets: Tuple[List[float], List[float]] = (
        factors[: num_clients // 2],  # fast, lower factor
        factors[num_clients // 2 :],  # slow, higher factor
    )

    final_factors: List[float] = []

    for idx in range(num_clients):
        # higher probability to pick bucket 0 with low idx
        bucket_idx = random.choices([0, 1], [num_clients - idx, idx])[0]
        picked_bucket = buckets[bucket_idx]
        other_bucket = buckets[bucket_idx - 1]

        if picked_bucket == other_bucket:
            raise Exception("Picked and other bucket can't be same")

        if len(picked_bucket) > 0:
            value = picked_bucket.pop(0)
        else:
            value = other_bucket.pop(0)

        final_factors.append(value)

    return final_factors


def get_delay_factor() -> float:
    """Return a delay factor"""
    values_prob = [val[2] for val in DEVICE_DISTRIBUTION]
    values_perf = [val[3] for val in DEVICE_DISTRIBUTION]
    max_perf = max(values_perf)
    chosen_score = random.choices(values_perf, values_prob)[0]
    return round(max_perf / chosen_score - 1, 4)


def configure_client_instances(
    num_clients: int, num_cpu: int, num_ram: float, gpu: bool = False
) -> Tuple[List[Instance], List[str]]:
    """Return list of client instances and a list of instance names."""
    instance_names = [f"client_{i}" for i in range(num_clients)]

    instances = [
        Instance(
            name=instance_name,
            group="clients",
            num_cpu=num_cpu,
            num_ram=num_ram,
            gpu=gpu,
        )
        for instance_name in instance_names
    ]

    return instances, instance_names
