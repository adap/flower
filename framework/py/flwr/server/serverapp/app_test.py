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
"""Tests for ServerApp process startup."""


from queue import Queue
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from flwr.common import Context, RecordDict
from flwr.common.typing import Fab, Run

from .app import run_serverapp


class TestRunServerApp(TestCase):
    """Tests for `run_serverapp`."""

    @patch("flwr.server.serverapp.app.flwr_exit")
    @patch("flwr.server.serverapp.app.run_")
    @patch("flwr.server.serverapp.app.HeartbeatSender")
    @patch("flwr.server.serverapp.app.event")
    @patch("flwr.server.serverapp.app.get_fused_config_from_dir", return_value={})
    @patch(
        "flwr.server.serverapp.app.get_project_config",
        return_value={
            "tool": {"flwr": {"app": {"components": {"serverapp": "mod:app"}}}}
        },
    )
    @patch("flwr.server.serverapp.app.get_project_dir", return_value=".")
    @patch(
        "flwr.server.serverapp.app.get_fab_metadata",
        return_value=("fab-id", "1.0.0"),
    )
    @patch("flwr.server.serverapp.app.install_from_fab")
    @patch("flwr.server.serverapp.app.start_log_uploader", return_value=None)
    @patch("flwr.server.serverapp.app.get_sha256_hash", return_value="hash")
    @patch("flwr.server.serverapp.app.fab_from_proto")
    @patch("flwr.server.serverapp.app.run_from_proto")
    @patch("flwr.server.serverapp.app.context_from_proto")
    @patch("flwr.server.serverapp.app.register_signal_handlers")
    @patch("flwr.server.serverapp.app.GrpcGrid")
    def test_run_serverapp_sets_run_from_pull_appinputs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_grid_cls: Mock,
        _mock_register_signal_handlers: Mock,
        mock_context_from_proto: Mock,
        mock_run_from_proto: Mock,
        mock_fab_from_proto: Mock,
        _mock_get_sha256_hash: Mock,
        _mock_start_log_uploader: Mock,
        _mock_install_from_fab: Mock,
        _mock_get_fab_metadata: Mock,
        _mock_get_project_dir: Mock,
        _mock_get_project_config: Mock,
        _mock_get_fused_config_from_dir: Mock,
        _mock_event: Mock,
        mock_heartbeat_sender_cls: Mock,
        mock_run_server_app: Mock,
        _mock_flwr_exit: Mock,
    ) -> None:
        """`run_serverapp` should call `grid.set_run` with a `Run` object."""
        run = Run.create_empty(run_id=42)
        fab = Fab(hash_str="fabhash", content=b"fab", verifications={})
        context = Context(
            run_id=42,
            node_id=0,
            node_config={},
            state=RecordDict(),
            run_config={},
        )
        mock_context_from_proto.return_value = context
        mock_run_from_proto.return_value = run
        mock_fab_from_proto.return_value = fab
        mock_run_server_app.return_value = context
        mock_heartbeat_sender = MagicMock()
        mock_heartbeat_sender_cls.return_value = mock_heartbeat_sender

        mock_stub = MagicMock()
        mock_stub.PullAppInputs.return_value = Mock(
            context=object(), run=object(), fab=object()
        )
        mock_stub.PushAppOutputs.return_value = Mock()
        mock_grid = MagicMock(**{"_stub": mock_stub})
        mock_grid_cls.return_value = mock_grid

        run_serverapp(
            serverappio_api_address="127.0.0.1:9091",
            log_queue=Queue(),
            token="token",
            certificates=None,
            parent_pid=None,
        )

        mock_grid.set_run.assert_called_once_with(run)
