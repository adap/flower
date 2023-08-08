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

from flwr.client import NumPyClient

from .secaggplus_handler import (
    STAGE_COLLECT_MASKED_INPUT,
    STAGE_SETUP,
    STAGE_SHARE_KEYS,
    STAGE_UNMASKING,
    STAGES,
    SecAggPlusHandler,
)


class EmptyFlowerNumPyClient(NumPyClient, SecAggPlusHandler):
    """Empty Numpy client."""


class TestSecAggPlusHandler(unittest.TestCase):
    """Test the SecAgg+ protocol handler."""

    def test_invalid_handler(self) -> None:
        """Test invalid handler."""
        handler = SecAggPlusHandler()

        with self.assertRaises(TypeError):
            handler.handle_secure_aggregation({})

    def test_stage_transition(self) -> None:
        """Test stage transition in the SecAggPlusHandler."""
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
