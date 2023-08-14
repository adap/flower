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
"""The SecAgg+ protocol handler tests."""

import unittest
from itertools import product
from typing import Any, Dict, List, cast

from flwr.client import NumPyClient
from flwr.common.typing import Value

from .secaggplus_handler import (
    STAGE_COLLECT_MASKED_INPUT,
    STAGE_SETUP,
    STAGE_SHARE_KEYS,
    STAGE_UNMASKING,
    STAGES,
    SecAggPlusHandler,
)


class EmptyFlowerNumPyClient(NumPyClient, SecAggPlusHandler):
    """Empty NumPyClient."""


class TestSecAggPlusHandler(unittest.TestCase):
    """Test the SecAgg+ protocol handler."""

    def test_invalid_handler(self) -> None:
        """Test invalid handler."""
        handler = SecAggPlusHandler()

        with self.assertRaises(TypeError):
            handler.handle_secure_aggregation({})

    def test_stage_transition(self) -> None:
        """Test stage transition."""
        handler = EmptyFlowerNumPyClient()

        assert STAGES == (
            STAGE_SETUP,
            STAGE_SHARE_KEYS,
            STAGE_COLLECT_MASKED_INPUT,
            STAGE_UNMASKING,
        )

        valid_transitions = set(
            [
                # From one stage to the next stage
                (STAGES[i], STAGES[(i + 1) % len(STAGES)])
                for i in range(len(STAGES))
            ]
            + [
                # From any stage to the initial stage
                (stage, STAGES[0])
                for stage in STAGES
            ]
        )

        invalid_transitions = set(product(STAGES, STAGES)).difference(valid_transitions)

        # Test valid transitions
        # If the next stage is valid, the function should update the current stage
        # and then raise KeyError or other exceptions when trying to execute SA.
        for current_stage, next_stage in valid_transitions:
            # pylint: disable-next=protected-access
            handler._current_stage = current_stage

            with self.assertRaises(KeyError):
                handler.handle_secure_aggregation({"stage": next_stage})
            # pylint: disable-next=protected-access
            assert handler._current_stage == next_stage

        # Test invalid transitions
        # If the next stage is invalid, the function should raise ValueError
        for current_stage, next_stage in invalid_transitions:
            # pylint: disable-next=protected-access
            handler._current_stage = current_stage

            with self.assertRaises(ValueError):
                handler.handle_secure_aggregation({"stage": next_stage})
            # pylint: disable-next=protected-access
            assert handler._current_stage == current_stage

    def test_stage_setup_check(self) -> None:
        """Test content checking for the setup stage."""
        handler = EmptyFlowerNumPyClient()

        valid_key_type_pairs = [
            ("sample_num", int),
            ("secure_id", int),
            ("share_num", int),
            ("threshold", int),
            ("clipping_range", float),
            ("target_range", int),
            ("mod_range", int),
        ]

        type_to_test_value: Dict[type, Value] = {
            int: 10,
            bool: True,
            float: 1.0,
            str: "test",
            bytes: b"test",
        }

        valid_named_values: Dict[str, Value] = {
            key: type_to_test_value[value_type]
            for key, value_type in valid_key_type_pairs
        }

        # Test valid `named_values`
        try:
            # pylint: disable=protected-access
            handler._current_stage = STAGE_SETUP
            handler._check_named_values(valid_named_values.copy())
            # pylint: enable=protected-access
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(
                "SecAggPlusHandler._check_named_values() "
                f"raised {type(exc)} unexpectedly!"
            )

        # Set the stage
        valid_named_values["stage"] = STAGE_SETUP

        # Test invalid `named_values`
        for key, value_type in valid_key_type_pairs:
            invalid_named_values = valid_named_values.copy()

            # Test wrong value type for the key
            for other_type, other_value in type_to_test_value.items():
                if other_type == value_type:
                    continue
                invalid_named_values[key] = other_value
                # pylint: disable-next=protected-access
                handler._current_stage = STAGE_UNMASKING
                with self.assertRaises(TypeError):
                    handler.handle_secure_aggregation(invalid_named_values.copy())

            # Test missing key
            invalid_named_values.pop(key)
            # pylint: disable-next=protected-access
            handler._current_stage = STAGE_UNMASKING
            with self.assertRaises(KeyError):
                handler.handle_secure_aggregation(invalid_named_values.copy())

    def test_stage_share_keys_check(self) -> None:
        """Test content checking for the share keys stage."""
        handler = EmptyFlowerNumPyClient()

        valid_named_values: Dict[str, Value] = {
            "1": [b"public key 1", b"public key 2"],
            "2": [b"public key 1", b"public key 2"],
            "3": [b"public key 1", b"public key 2"],
        }

        # Test valid `named_values`
        try:
            # pylint: disable=protected-access
            handler._current_stage = STAGE_SHARE_KEYS
            handler._check_named_values(valid_named_values.copy())
            # pylint: enable=protected-access
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(
                "SecAggPlusHandler._check_named_values() "
                f"raised {type(exc)} unexpectedly!"
            )

        # Set the stage
        valid_named_values["stage"] = STAGE_SHARE_KEYS

        # Test invalid `named_values`
        invalid_values: List[Value] = [
            b"public key 1",
            [b"public key 1"],
            [b"public key 1", b"public key 2", b"public key 3"],
        ]

        for value in invalid_values:
            invalid_named_values = valid_named_values.copy()
            invalid_named_values["1"] = value

            # pylint: disable-next=protected-access
            handler._current_stage = STAGE_SETUP
            with self.assertRaises(TypeError):
                handler.handle_secure_aggregation(invalid_named_values.copy())

    def test_stage_collect_masked_input_check(self) -> None:
        """Test content checking for the collect masked input stage."""
        handler = EmptyFlowerNumPyClient()

        valid_named_values: Dict[str, Value] = {
            "ciphertexts": [b"ctxt!", b"ctxt@", b"ctxt#", b"ctxt?"],
            "srcs": [32, 51324, 32324123, -3],
            "parameters": [b"params1", b"params2"],
        }

        # Test valid `named_values`
        try:
            # pylint: disable=protected-access
            handler._current_stage = STAGE_COLLECT_MASKED_INPUT
            handler._check_named_values(valid_named_values.copy())
            # pylint: enable=protected-access
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(
                "SecAggPlusHandler._check_named_values() "
                f"raised {type(exc)} unexpectedly!"
            )

        # Set the stage
        valid_named_values["stage"] = STAGE_COLLECT_MASKED_INPUT

        # Test invalid `named_values`
        # Test missing keys
        for key in list(valid_named_values.keys()):
            if key == "stage":
                continue
            invalid_named_values = valid_named_values.copy()
            invalid_named_values.pop(key)
            # pylint: disable-next=protected-access
            handler._current_stage = STAGE_SHARE_KEYS
            with self.assertRaises(KeyError):
                handler.handle_secure_aggregation(invalid_named_values)

        # Test wrong value type for the key
        for key in valid_named_values:
            if key == "stage":
                continue
            invalid_named_values = valid_named_values.copy()
            cast(List[Any], invalid_named_values[key]).append(3.1415926)
            # pylint: disable-next=protected-access
            handler._current_stage = STAGE_SHARE_KEYS
            with self.assertRaises(TypeError):
                handler.handle_secure_aggregation(invalid_named_values)

    def test_stage_unmasking_check(self) -> None:
        """Test content checking for the unmasking stage."""
        handler = EmptyFlowerNumPyClient()

        valid_named_values: Dict[str, Value] = {
            "active_sids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "dead_sids": [32, 51324, 32324123, -3],
        }

        # Test valid `named_values`
        try:
            # pylint: disable=protected-access
            handler._current_stage = STAGE_UNMASKING
            handler._check_named_values(valid_named_values.copy())
            # pylint: enable=protected-access
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(
                "SecAggPlusHandler._check_named_values() "
                f"raised {type(exc)} unexpectedly!"
            )

        # Set the stage
        valid_named_values["stage"] = STAGE_UNMASKING

        # Test invalid `named_values`
        # Test missing keys
        for key in list(valid_named_values.keys()):
            if key == "stage":
                continue
            invalid_named_values = valid_named_values.copy()
            invalid_named_values.pop(key)
            # pylint: disable-next=protected-access
            handler._current_stage = STAGE_COLLECT_MASKED_INPUT
            with self.assertRaises(KeyError):
                handler.handle_secure_aggregation(invalid_named_values)

        # Test wrong value type for the key
        for key in valid_named_values:
            if key == "stage":
                continue
            invalid_named_values = valid_named_values.copy()
            cast(List[Any], invalid_named_values[key]).append(True)
            # pylint: disable-next=protected-access
            handler._current_stage = STAGE_COLLECT_MASKED_INPUT
            with self.assertRaises(TypeError):
                handler.handle_secure_aggregation(invalid_named_values)
