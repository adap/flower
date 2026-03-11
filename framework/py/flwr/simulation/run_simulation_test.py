# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for simulation startup."""


from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from flwr.common import Context, EventType, RecordDict
from flwr.common.typing import Run

from .run_simulation import _main_loop


class TestMainLoop(TestCase):
    """Tests for `_main_loop`."""

    @patch("flwr.simulation.run_simulation.event")
    @patch("flwr.simulation.run_simulation.vce.start_vce")
    @patch("flwr.simulation.run_simulation.run_serverapp_th")
    @patch("flwr.simulation.run_simulation.InMemoryGrid")
    @patch("flwr.simulation.run_simulation.LinkStateFactory")
    def test_main_loop_sets_run_with_run_object(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_linkstate_factory_cls: Mock,
        mock_grid_cls: Mock,
        mock_run_serverapp_th: Mock,
        _mock_start_vce: Mock,
        _mock_event: Mock,
    ) -> None:
        """`_main_loop` should call `grid.set_run` with a `Run` object."""
        run = Run.create_empty(run_id=31415)
        context = Context(
            run_id=run.run_id,
            node_id=0,
            node_config={},
            state=RecordDict(),
            run_config={},
        )

        mock_state_factory = MagicMock()
        mock_state_factory.state.return_value = MagicMock(run_ids={})
        mock_linkstate_factory_cls.return_value = mock_state_factory

        mock_grid = MagicMock()
        mock_grid_cls.return_value = mock_grid

        mock_thread = Mock()

        def _run_serverapp_th_side_effect(*args: object, **kwargs: object) -> Mock:
            ctx_queue = kwargs["ctx_queue"]
            ctx_queue.put(context)
            return mock_thread

        mock_run_serverapp_th.side_effect = _run_serverapp_th_side_effect

        updated_context = _main_loop(
            num_supernodes=1,
            backend_name="ray",
            backend_config_stream="{}",
            app_dir=".",
            is_app=False,
            enable_tf_gpu_growth=False,
            run=run,
            exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
            server_app_context=context,
        )

        self.assertEqual(updated_context.run_id, run.run_id)
        mock_grid.set_run.assert_called_once_with(run)
        mock_thread.join.assert_called_once()

