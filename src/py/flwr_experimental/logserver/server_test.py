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
"""Test for server."""

import os.path
import tempfile
import unittest

from flwr_experimental.logserver.server import (
    CONFIG,
    parse_plot_message,
    plot_accuracies,
)


# pylint: disable=no-self-use
class LogserverTest(unittest.TestCase):
    """Tests for functions in the server module."""

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        # Cleanup the directory after the test
        # self.test_dir.cleanup()
        pass

    def test_parse_plot_message(self) -> None:
        """Test parse_plot_message function."""
        # Prepare
        message = "app_fit: accuracies_centralized [(0, 0.019), (1, 0.460)]"
        expected_plot_type = "accuracies"
        expected_values = [(0, 0.019), (1, 0.460)]

        # Execute
        plot_type, values = parse_plot_message(message)

        # Assert
        assert plot_type == expected_plot_type
        assert values == expected_values

    def test_plot_accuracies(self) -> None:
        """Test plot accuracies function."""
        # Prepare
        values = [(0, 0.019), (1, 0.460), (2, 0.665), (3, 0.845)]
        CONFIG["s3_key"] = os.path.join(self.test_dir.name, "foo.log")

        expected_filepath = os.path.join(
            self.test_dir.name, f'{CONFIG["s3_key"]}.accuracies.pdf'
        )

        # Execute
        plot_accuracies(values)

        # Assert
        assert os.path.isfile(expected_filepath)


if __name__ == "__main__":
    unittest.main()
