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
"""ConfigRecord."""


from logging import WARN
from typing import Optional, get_args

from flwr.common.typing import ConfigRecordValues, ConfigScalar

from ..logger import log
from .typeddict import TypedDict


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: ConfigRecordValues) -> None:
    def is_valid(__v: ConfigScalar) -> None:
        """Check if value is of expected type."""
        if not isinstance(__v, get_args(ConfigScalar)):
            raise TypeError(
                "Not all values are of valid type."
                f" Expected `{ConfigRecordValues}` but `{type(__v)}` was passed."
            )

    if isinstance(value, list):
        # If your lists are large (e.g. 1M+ elements) this will be slow
        # 1s to check 10M element list on a M2 Pro
        # In such settings, you'd be better of treating such config as
        # an array and pass it to a ArrayRecord.
        # Empty lists are valid
        if len(value) > 0:
            is_valid(value[0])
            # all elements in the list must be of the same valid type
            # this is needed for protobuf
            value_type = type(value[0])
            if not all(isinstance(v, value_type) for v in value):
                raise TypeError(
                    "All values in a list must be of the same valid type. "
                    f"One of {ConfigScalar}."
                )
    else:
        is_valid(value)


class ConfigRecord(TypedDict[str, ConfigRecordValues]):
    """Configs record.

    A :code:`ConfigRecord` is a Python dictionary designed to ensure that
    each key-value pair adheres to specified data types. A :code:`ConfigRecord`
    is one of the types of records that a
    `flwr.common.RecordDict <flwr.common.RecordDict.html#recorddict>`_ supports and
    can therefore be used to construct :code:`common.Message` objects.

    Parameters
    ----------
    config_dict : Optional[Dict[str, ConfigRecordValues]]
        A dictionary that stores basic types (i.e. `str`, `int`, `float`, `bytes` as
        defined in `ConfigsScalar`) and lists of such types (see
        `ConfigsScalarList`).
    keep_input : bool (default: True)
        A boolean indicating whether config passed should be deleted from the input
        dictionary immediately after adding them to the record. When set
        to True, the data is duplicated in memory. If memory is a concern, set
        it to False.

    Examples
    --------
    The usage of a :code:`ConfigRecord` is envisioned for sending configuration values
    telling the target node how to perform a certain action (e.g. train/evaluate a model
    ). You can use standard Python built-in types such as :code:`float`, :code:`str`
    , :code:`bytes`. All types allowed are defined in
    :code:`flwr.common.ConfigRecordValues`. While lists are supported, we
    encourage you to use a :code:`ArrayRecord` instead if these are of high
    dimensionality.

    Let's see some examples of how to construct a :code:`ConfigRecord` from scratch:

    >>> from flwr.common import ConfigRecord
    >>>
    >>> # A `ConfigRecord` is a specialized Python dictionary
    >>> record = ConfigRecord({"lr": 0.1, "batch-size": 128})
    >>> # You can add more content to an existing record
    >>> record["compute-average"] = True
    >>> # It also supports lists
    >>> record["loss-fn-coefficients"] = [0.4, 0.25, 0.35]
    >>> # And string values (among other types)
    >>> record["path-to-S3"] = "s3://bucket_name/folder1/fileA.json"

    Just like the other types of records in a :code:`flwr.common.RecordDict`, types are
    enforced. If you need to add a custom data structure or object, we recommend to
    serialise it into bytes and save it as such (bytes are allowed in a
    :code:`ConfigRecord`)
    """

    def __init__(
        self,
        config_dict: Optional[dict[str, ConfigRecordValues]] = None,
        keep_input: bool = True,
    ) -> None:

        super().__init__(_check_key, _check_value)
        if config_dict:
            for k in list(config_dict.keys()):
                self[k] = config_dict[k]
                if not keep_input:
                    del config_dict[k]

    def count_bytes(self) -> int:
        """Return number of Bytes stored in this object.

        This function counts booleans as occupying 1 Byte.
        """

        def get_var_bytes(value: ConfigScalar) -> int:
            """Return Bytes of value passed."""
            var_bytes = 0
            if isinstance(value, bool):
                var_bytes = 1
            elif isinstance(value, (int, float)):
                var_bytes = (
                    8  # the profobufing represents int/floats in ConfigRecords as 64bit
                )
            if isinstance(value, (str, bytes)):
                var_bytes = len(value)
            if var_bytes == 0:
                raise ValueError(
                    "Config values must be either `bool`, `int`, `float`, "
                    "`str`, or `bytes`"
                )
            return var_bytes

        num_bytes = 0

        for k, v in self.items():
            if isinstance(v, list):
                if isinstance(v[0], (bytes, str)):
                    # not all str are of equal length necessarily
                    # for both the footprint of each element is 1 Byte
                    num_bytes += int(sum(len(s) for s in v))  # type: ignore
                else:
                    num_bytes += get_var_bytes(v[0]) * len(v)
            else:
                num_bytes += get_var_bytes(v)

            # We also count the bytes footprint of the keys
            num_bytes += len(k)

        return num_bytes


class ConfigsRecord(ConfigRecord):
    """Deprecated class ``ConfigsRecord``, use ``ConfigRecord`` instead.

    This class exists solely for backward compatibility with legacy
    code that previously used ``ConfigsRecord``. It has been renamed
    to ``ConfigRecord``.

    .. warning::
        ``ConfigsRecord`` is deprecated and will be removed in a future release.
        Use ``ConfigRecord`` instead.

    Examples
    --------
    Legacy (deprecated) usage::

        from flwr.common import ConfigsRecord

        record = ConfigsRecord()

    Updated usage::

        from flwr.common import ConfigRecord

        record = ConfigRecord()
    """

    _warning_logged = False

    def __init__(
        self,
        config_dict: Optional[dict[str, ConfigRecordValues]] = None,
        keep_input: bool = True,
    ):
        if not ConfigsRecord._warning_logged:
            ConfigsRecord._warning_logged = True
            log(
                WARN,
                "The `ConfigsRecord` class has been renamed to `ConfigRecord`. "
                "Support for `ConfigsRecord` will be removed in a future release. "
                "Please update your code accordingly.",
            )
        super().__init__(config_dict, keep_input)
