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


from typing import Dict, List, Optional, get_args

from flwr.common.typing import MetricsRecordValues, MetricsScalar

from .typeddict import TypedDict


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: MetricsRecordValues) -> None:
    def is_valid(__v: MetricsScalar) -> None:
        """Check if value is of expected type."""
        if not isinstance(__v, get_args(MetricsScalar)) or isinstance(__v, bool):
            raise TypeError(
                "Not all values are of valid type."
                f" Expected `{MetricsRecordValues}` but `{type(__v)}` was passed."
            )

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


class MetricsRecord(TypedDict[str, MetricsRecordValues]):
    """Metrics record."""

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
        super().__init__(_check_key, _check_value)
        if metrics_dict:
            for k in list(metrics_dict.keys()):
                self[k] = metrics_dict[k]
                if not keep_input:
                    del metrics_dict[k]

    def count_bytes(self) -> int:
        """Return number of Bytes stored in this object."""
        num_bytes = 0

        for k, v in self.items():
            if isinstance(v, List):
                # both int and float normally take 4 bytes
                # But MetricRecords are mapped to 64bit int/float
                # during protobuffing
                num_bytes += 8 * len(v)
            else:
                num_bytes += 8
            # We also count the bytes footprint of the keys
            num_bytes += len(k)
        return num_bytes