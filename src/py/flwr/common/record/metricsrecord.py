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


from typing import Optional, get_args

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
    """Metrics recod.

    A :code:`MetricsRecord` is a Python dictionary designed to ensure that
    each key-value pair adheres to specified data types. A :code:`MetricsRecord`
    is one of the types of records that a
    `flwr.common.RecordSet <flwr.common.RecordSet.html#recordset>`_ supports and
    can therefore be used to construct :code:`common.Message` objects.

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

    Examples
    --------
    The usage of a :code:`MetricsRecord` is envisioned for communicating results
    obtained when a node performs an action. A few typical examples include:
    communicating the training accuracy after a model is trained locally by a
    :code:`ClientApp`, reporting the validation loss obtained at a :code:`ClientApp`,
    or, more generally, the output of executing a query by the :code:`ClientApp`.
    Common to these examples is that the output can be typically represented by
    a single scalar (:code:`int`, :code:`float`) or list of scalars.

    Let's see some examples of how to construct a :code:`MetricsRecord` from scratch:

    >>> from flwr.common import MetricsRecord
    >>>
    >>> # A `MetricsRecord` is a specialized Python dictionary
    >>> record = MetricsRecord({"accuracy": 0.94})
    >>> # You can add more content to an existing record
    >>> record["loss"] = 0.01
    >>> # It also supports lists
    >>> record["loss-historic"] = [0.9, 0.5, 0.01]

    Since types are enforced, the types of the objects inserted are checked. For a
    :code:`MetricsRecord`, value types allowed are those in defined in
    :code:`flwr.common.MetricsRecordValues`. Similarly, only :code:`str` keys are
    allowed.

    >>> from flwr.common import MetricsRecord
    >>>
    >>> record = MetricsRecord() # an empty record
    >>> # Add unsupported value
    >>> record["something-unsupported"] = {'a': 123} # Will throw a `TypeError`

    If you need a more versatily type of record try :code:`ConfigsRecord` or
    :code:`ParametersRecord`.
    """

    def __init__(
        self,
        metrics_dict: Optional[dict[str, MetricsRecordValues]] = None,
        keep_input: bool = True,
    ):
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
            if isinstance(v, list):
                # both int and float normally take 4 bytes
                # But MetricRecords are mapped to 64bit int/float
                # during protobuffing
                num_bytes += 8 * len(v)
            else:
                num_bytes += 8
            # We also count the bytes footprint of the keys
            num_bytes += len(k)
        return num_bytes
