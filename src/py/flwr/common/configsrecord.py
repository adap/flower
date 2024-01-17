# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""ConfigsRecord."""


from dataclasses import dataclass
from typing import Dict

from .metricsrecord import MetricsRecord, MetricsRecordValues


@dataclass
class ConfigsRecord(MetricsRecord):
    """Configs record."""

    def set_configs(self, configs_dict: Dict[str, MetricsRecordValues]) -> None:
        """Add configs to record.

        This not implemented as a constructor so we can cleanly create and empyt
        ConfigsRecord object.
        """
        super().set_metrics(configs_dict)
