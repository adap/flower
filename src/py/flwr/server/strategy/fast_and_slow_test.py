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
"""Tests for Fast-and-Slow strategy."""

import unittest

from flwr.server.strategy import fast_and_slow


# pylint: disable=no-self-use,missing-class-docstring,missing-function-docstring
class FastAndSlowTestCase(unittest.TestCase):
    def test_fast_round(self) -> None:
        # Prepare
        values = [
            # 1 fast, 1 slow
            (0, 1, 1, True),
            (1, 1, 1, False),
            (2, 1, 1, True),
            # 2 fast, 1 slow
            (0, 2, 1, True),
            (1, 2, 1, True),
            (2, 2, 1, False),
            (3, 2, 1, True),
            (4, 2, 1, True),
            (5, 2, 1, False),
            # 1 fast, 2 slow
            (0, 1, 2, True),
            (1, 1, 2, False),
            (2, 1, 2, False),
            (3, 1, 2, True),
            (4, 1, 2, False),
            (5, 1, 2, False),
            # 3 fast, 2 slow
            (0, 3, 2, True),
            (1, 3, 2, True),
            (2, 3, 2, True),
            (3, 3, 2, False),
            (4, 3, 2, False),
            (5, 3, 2, True),
        ]

        # Execute and assert
        for rnd, r_fast, r_slow, expected in values:
            actual = fast_and_slow.is_fast_round(rnd, r_fast, r_slow)
            assert actual == expected

    def test_next_timeout_below_max(self) -> None:
        # Prepare
        durations = [15.6, 13.1, 18.7]
        percentile = 0.5
        expected = 16

        # Execute
        actual = fast_and_slow.next_timeout(durations, percentile)

        # Assert
        assert actual == expected


if __name__ == "__main__":
    unittest.main(verbosity=2)
