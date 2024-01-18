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
"""MetricsRecord."""


from dataclasses import dataclass, field
from typing import Dict, Optional, get_args

from .typing import MetricsRecordValues, MetricsScalar


@dataclass
class MetricsRecord:
    """Metrics record."""

    keep_input: bool
    data: Dict[str, MetricsRecordValues] = field(default_factory=dict)

    def __init__(
        self,
        metrics_dict: Optional[Dict[str, MetricsRecordValues]] = None,
        keep_input: bool = True,
    ):
        """Construct a MetricsRecord object.

        Parameters
        ----------
        metrics_dict : Optional[Dict[str, MetricsRecordValues]]
            A dictionary that stores basic types (i.e. `int`, `float` as defined
            in `MetricsScalar`) and list of such types (see `MetricsScalarList`).
        keep_input : bool (default: True)
            A boolean indicating whether metrics should be deleted from the input
            dictionary immediately after adding them to the record. When set
            to True, the data is duplicated in memory. If memory is a concern, set
            it to False.
        """
        self.keep_input = keep_input
        self.data = {}
        if metrics_dict:
            self.set_metrics(metrics_dict)

    def set_metrics(self, metrics_dict: Dict[str, MetricsRecordValues]) -> None:
        """Add metrics to the record.

        Parameters
        ----------
        metrics_dict : Dict[str, MetricsRecordValues]
            A dictionary that stores basic types (i.e. `int`, `float` as defined
            in `MetricsScalar`) and list of such types (see `MetricsScalarList`).
        """
        if any(not isinstance(k, str) for k in metrics_dict.keys()):
            raise TypeError(f"Not all keys are of valid type. Expected {str}.")

        def is_valid(value: MetricsScalar) -> None:
            """Check if value is of expected type."""
            if not isinstance(value, get_args(MetricsScalar)):
                raise TypeError(
                    "Not all values are of valid type."
                    f" Expected {MetricsRecordValues} but you passed {type(value)}."
                )

        # Check types of values
        # Split between those values that are list and those that aren't
        # then process in the same way
        for value in metrics_dict.values():
            if isinstance(value, list):
                # If your lists are large (e.g. 1M+ elements) this will be slow
                # 1s to check 10M element list on a M2 Pro
                # In such settings, you'd be better of treating such metric as
                # an array and pass it to a ParametersRecord.
                for list_value in value:
                    is_valid(list_value)
            else:
                is_valid(value)

        # Add metrics to record
        if self.keep_input:
            # Copy
            self.data = metrics_dict.copy()
        else:
            # Add entries to dataclass without duplicating memory
            for key in list(metrics_dict.keys()):
                self.data[key] = metrics_dict[key]
                del metrics_dict[key]
