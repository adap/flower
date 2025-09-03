# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Strategy results."""

import pprint
from dataclasses import dataclass, field

from flwr.common import ArrayRecord, MetricRecord


@dataclass
class Result:
    """Data class carrying records generated during the execution of a strategy."""

    arrays: ArrayRecord = field(default_factory=ArrayRecord)
    train_metrics_clientapp: dict[int, MetricRecord] = field(default_factory=dict)
    evaluate_metrics_clientapp: dict[int, MetricRecord] = field(default_factory=dict)
    evaluate_metrics_serverapp: dict[int, MetricRecord] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Create a representation of the Result instance."""
        rep = ""
        arr_size = sum(len(array.data) for array in self.arrays.values()) / (1024**2)
        rep += "Result.arrays:\n" + f"\tArrayRecord ({arr_size:.3f} MB)\n" + "\n"
        rep += (
            "Result.train_metrics_clientapp (per-round training metrics "
            "from ClientApps):\n"
            + pprint.pformat(self.train_metrics_clientapp, indent=2)
            + "\n\n"
        )

        rep += (
            "Result.evaluate_metrics_clientapp (per-round evaluation metrics "
            "from ClientApps):\n"
            + pprint.pformat(self.evaluate_metrics_clientapp, indent=2)
            + "\n\n"
        )

        rep += (
            "Result.evaluate_metrics_serverapp (per-round evaluation metrics "
            "from ServerApp):\n"
            + pprint.pformat(self.evaluate_metrics_serverapp, indent=2)
            + "\n"
        )

        return rep
