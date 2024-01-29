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
        self.data = {}
        if metrics_dict:
            self.set_metrics(metrics_dict, keep_input=keep_input)

    def set_metrics(
        self, metrics_dict: Dict[str, MetricsRecordValues], keep_input: bool = True
    ) -> None:
        """Add metrics to the record.

        Parameters
        ----------
        metrics_dict : Dict[str, MetricsRecordValues]
            A dictionary that stores basic types (i.e. `int`, `float` as defined
            in `MetricsScalar`) and list of such types (see `MetricsScalarList`).
        keep_input : bool (default: True)
            A boolean indicating whether metrics should be deleted from the input
            dictionary immediately after adding them to the record. When set
            to True, the data is duplicated in memory. If memory is a concern, set
            it to False.
        """
        if any(not isinstance(k, str) for k in metrics_dict.keys()):
            raise TypeError(f"Not all keys are of valid type. Expected {str}.")

        def is_valid(value: MetricsScalar) -> None:
            """Check if value is of expected type."""
            if not isinstance(value, get_args(MetricsScalar)) or isinstance(
                value, bool
            ):
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
                # Empty lists are valid
                if len(value) > 0:
                    is_valid(value[0])
                    # all elements in the list must be of the same valid type
                    # this is needed for protobuf
                    value_type = type(value[0])
                    if not all(isinstance(v, value_type) for v in value):
                        raise TypeError(
                            "All values in a list must be of the same valid type. "
                            f"One of {MetricsScalar}."
                        )
            else:
                is_valid(value)

        # Add metrics to record
        if keep_input:
            # Copy
            self.data = metrics_dict.copy()
        else:
            # Add entries to dataclass without duplicating memory
            for key in list(metrics_dict.keys()):
                self.data[key] = metrics_dict[key]
                del metrics_dict[key]

    def __getitem__(self, key: str) -> MetricsRecordValues:
        """Retrieve an element stored in record."""
        return self.data[key]
