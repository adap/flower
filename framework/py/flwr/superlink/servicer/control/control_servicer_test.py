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
"""Test the Control API servicer."""


import hashlib
import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import grpc
from parameterized import parameterized

from flwr.common import ConfigRecord, now
from flwr.common.constant import (
    NODE_NOT_FOUND_MESSAGE,
    PUBLIC_KEY_ALREADY_IN_USE_MESSAGE,
    PUBLIC_KEY_NOT_VALID,
    Status,
    SubStatus,
)
from flwr.common.typing import Run, RunStatus
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListNodesRequest,
    ListNodesResponse,
    ListRunsRequest,
    RegisterNodeRequest,
    StartRunRequest,
    StopRunRequest,
    StreamLogsRequest,
    StreamLogsResponse,
    UnregisterNodeRequest,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.control.control_account_auth_interceptor import (
    shared_account_info,
)

from .control_servicer import ControlServicer

FLWR_AID_MISMATCH_CASES = (
    # (context_flwr_aid, run_flwr_aid)
    ("user-123", "user-xyz"),
    ("user-234", ""),
    ("", "user-234"),
    ("user-345", None),
    (None, "user-456"),
    (None, None),
    ("", ""),
    ("", None),
    (None, ""),
)


class TestControlServicer(unittest.TestCase):
    """Test the Control API servicer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.store = Mock()
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager()
            ),
            ffs_factory=FfsFactory(self.tmp_dir.name),
            objectstore_factory=Mock(store=Mock(return_value=self.store)),
            is_simulation=False,
            authn_plugin=(authn_plugin := NoOpControlAuthnPlugin(Mock(), False)),
        )
        account_info = authn_plugin.validate_tokens_in_metadata([])[1]
        assert account_info is not None
        self.aid = account_info.flwr_aid
        shared_account_info.set(account_info)
        self.state = self.servicer.linkstate_factory.state()

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.tmp_dir.cleanup()

    def _create_dummy_run(self, flwr_aid: str | None) -> int:
        return self.state.create_run(
            "flwr/demo", "v0.0.1", "hash123", {}, "mock-fed", ConfigRecord(), flwr_aid
        )

    def test_start_run(self) -> None:
        """Test StartRun method of ControlServicer."""
        # Prepare
        fab_content = b"test FAB content 123456"
        fab_hash = hashlib.sha256(fab_content).hexdigest()
        fab_id = b"mock FAB ID"
        fab_version = b"mock FAB version"
        request = StartRunRequest()
        request.fab.hash_str = fab_hash
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION

        # Execute
        with patch(
            "flwr.superlink.servicer.control.control_servicer.get_fab_metadata"
        ) as mock_get_fab_metadata:
            mock_get_fab_metadata.return_value = (fab_id, fab_version)
            response = self.servicer.StartRun(request, Mock())
        run_info = self.state.get_run(response.run_id)

        # Assert
        assert run_info is not None
        self.assertEqual(run_info.fab_hash, fab_hash)
        self.assertEqual(run_info.fab_id, fab_id)
        self.assertEqual(run_info.fab_version, fab_version)

    def test_list_runs(self) -> None:
        """Test List method of ControlServicer with --runs option."""
        # Prepare
        run_ids = set()
        for _ in range(3):
            run_id = self._create_dummy_run(self.aid)
            run_ids.add(run_id)

        # Execute
        response = self.servicer.ListRuns(ListRunsRequest(), Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        self.assertLess(abs(retrieved_timestamp - now().timestamp()), 1e-3)
        self.assertEqual(set(response.run_dict.keys()), run_ids)

    def test_list_run_id(self) -> None:
        """Test List method of ControlServicer with --run-id option."""
        # Prepare
        for _ in range(3):
            run_id = self._create_dummy_run(self.aid)

        # Execute
        response = self.servicer.ListRuns(ListRunsRequest(run_id=run_id), Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        self.assertLess(abs(retrieved_timestamp - now().timestamp()), 1e-3)
        self.assertEqual(set(response.run_dict.keys()), {run_id})

    def test_stop_run(self) -> None:
        """Test StopRun method of ControlServicer."""
        # Prepare
        run_id = self._create_dummy_run(self.aid)
        expected_run_status = RunStatus(Status.FINISHED, SubStatus.STOPPED, "")

        # Execute
        response = self.servicer.StopRun(StopRunRequest(run_id=run_id), Mock())
        run_state = self.state.get_run(run_id)

        # Assert
        self.assertTrue(response.success)
        self.assertIsNotNone(run_state)
        if run_state is not None:
            self.assertEqual(run_state.status, expected_run_status)
        self.store.delete_objects_in_run.assert_called_once_with(run_id)

    @parameterized.expand(
        [
            (
                "",
                False,
                public_key_to_bytes(generate_key_pairs()[1]),
            ),  # PASSES, true EC keys used once
            (
                PUBLIC_KEY_ALREADY_IN_USE_MESSAGE,
                True,
                public_key_to_bytes(generate_key_pairs()[1]),
            ),  # FAILS, true EC keys but already in use
            (
                PUBLIC_KEY_NOT_VALID,
                False,
                os.urandom(32),
            ),  # FAILS, fake EC keys
        ]
    )  # type: ignore
    def test_create_node_cli(
        self, error_msg: str, pre_register_key: bool, pub_key: bytes
    ) -> None:
        """Test CreateNodeCli method of ControlServicer."""
        # Prepare
        if pre_register_key:
            self.state.create_node(
                owner_aid="fake_aid",
                owner_name="fake_name",
                public_key=pub_key,
                heartbeat_interval=10,
            )

        # Execute
        req = RegisterNodeRequest(public_key=pub_key)
        ctx = Mock()
        node_id = self.servicer.RegisterNode(req, ctx)
        if error_msg:
            ctx.abort.assert_called_once_with(
                grpc.StatusCode.FAILED_PRECONDITION, error_msg
            )
        else:
            assert node_id

    @parameterized.expand(
        [
            (True,),  # PASSES, uses registered node ID
            (False),  # FAILS, uses unregistered node ID
        ]
    )  # type: ignore
    def test_delete_node_cli(self, real_node_id: bool) -> None:
        """Test DeleteNodeCli method of ControlServicer."""
        # Prepare
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        node_id = self.state.create_node(
            owner_aid="fake_aid",
            owner_name="fake_name",
            public_key=pub_key,
            heartbeat_interval=10,
        )

        # Execute
        req = UnregisterNodeRequest(node_id=node_id if real_node_id else node_id + 1)
        ctx = Mock()
        self.servicer.UnregisterNode(req, ctx)
        if not real_node_id:
            ctx.abort.assert_called_once_with(
                grpc.StatusCode.NOT_FOUND, NODE_NOT_FOUND_MESSAGE
            )

    def test_create_delete_create_node_cli(self) -> None:
        """Test CreateNodeCli and DeleteNodeCli method of ControlServicer."""
        # Prepare
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        node_id = self.state.create_node(
            owner_aid="fake_aid",
            owner_name="fake_name",
            public_key=pub_key,
            heartbeat_interval=10,
        )

        # Execute
        # Unregister node
        self.servicer.UnregisterNode(UnregisterNodeRequest(node_id=node_id), Mock())

        # Try to add node with same public key again
        self.servicer.RegisterNode(RegisterNodeRequest(public_key=pub_key), Mock())

    @parameterized.expand(
        [
            ("fake_aid", True),  # One NodeId is retrieved
            ("another_fake_aid", False),  # Zero NodeId are retrieved
        ]
    )  # type: ignore
    def test_list_nodes_cli(self, flwr_aid_retrieving: str, expected: bool) -> None:
        """Test ListNodesCli method of ControlServicer."""
        # Prepare
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        node_id = self.state.create_node(
            owner_aid="fake_aid",
            owner_name="fake_name",
            public_key=pub_key,
            heartbeat_interval=10,
        )

        # Execute
        with patch(
            "flwr.superlink.servicer.control.control_servicer.shared_account_info",
            new=SimpleNamespace(
                get=lambda: SimpleNamespace(flwr_aid=flwr_aid_retrieving)
            ),
        ):
            res: ListNodesResponse = self.servicer.ListNodes(ListNodesRequest(), Mock())

        # Assert
        if expected:
            self.assertEqual(len(res.nodes_info), 1)
            self.assertEqual(res.nodes_info[0].node_id, node_id)
            self.assertEqual(res.nodes_info[0].owner_aid, "fake_aid")
            self.assertEqual(res.nodes_info[0].public_key, pub_key)
        else:
            self.assertEqual(len(res.nodes_info), 0)


class TestControlServicerAuth(unittest.TestCase):
    """Test ControlServicer methods with authentication plugin and flwr_aid checking."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager()
            ),
            ffs_factory=FfsFactory(self.tmp_dir.name),
            objectstore_factory=Mock(),
            is_simulation=False,
            authn_plugin=Mock(),
        )
        self.state = self.servicer.linkstate_factory.state()

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.tmp_dir.cleanup()

    def _create_dummy_run(self, flwr_aid: str | None) -> int:
        return self.state.create_run(
            "flwr/demo", "v0.0.1", "hash123", {}, "mock-fed", ConfigRecord(), flwr_aid
        )

    def make_context(self) -> MagicMock:
        """Create a mock context."""
        ctx = MagicMock(spec=grpc.ServicerContext)
        # abort should raise for testing error paths
        ctx.abort.side_effect = lambda code, msg: (_ for _ in ()).throw(
            RuntimeError(f"{code}:{msg}")
        )
        ctx.is_active.return_value = False
        return ctx

    # Test all invalid cases for StreamLogs with authentication
    @parameterized.expand(FLWR_AID_MISMATCH_CASES)  # type: ignore
    def test_streamlogs_auth_unsucessful(
        self, context_flwr_aid: str | None, run_flwr_aid: str | None
    ) -> None:
        """Test StreamLogs unsuccessful."""
        # Prepare
        run_id = self._create_dummy_run(run_flwr_aid)
        request = StreamLogsRequest(run_id=run_id, after_timestamp=0)
        ctx = self.make_context()

        # Execute & Assert
        with patch(
            "flwr.superlink.servicer.control.control_servicer.shared_account_info",
            new=SimpleNamespace(get=lambda: SimpleNamespace(flwr_aid=context_flwr_aid)),
        ):
            gen = self.servicer.StreamLogs(request, ctx)
            with self.assertRaises(RuntimeError) as cm:
                next(gen)
            self.assertIn("PERMISSION_DENIED", str(cm.exception))

    def test_streamlogs_auth_successful(self) -> None:
        """Test StreamLogs successful with matching flwr_aid."""
        # Prepare
        run_id = self._create_dummy_run("user-123")
        request = StreamLogsRequest(run_id=run_id, after_timestamp=0)
        ctx = self.make_context()
        ctx.is_active.return_value = True

        # Execute & Assert
        with (
            patch.object(
                self.state, "get_serverapp_log", new=lambda rid, ts: ("log1", 1.0)
            ),
            patch.object(
                self.state,
                "get_run_status",
                new=lambda ids: {
                    run_id: RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")
                },
            ),
            patch(
                "flwr.superlink.servicer.control.control_servicer.shared_account_info",
                new=SimpleNamespace(get=lambda: SimpleNamespace(flwr_aid="user-123")),
            ),
        ):
            msgs = list(self.servicer.StreamLogs(request, ctx))
            gen = self.servicer.StreamLogs(request, ctx)
            msgs = list(gen)
            self.assertEqual(len(msgs), 1)
            self.assertIsInstance(msgs[0], StreamLogsResponse)
            self.assertEqual(msgs[0].log_output, "log1")
            self.assertEqual(msgs[0].latest_timestamp, 1.0)

    # Test all invalid cases for StopRun with authentication
    @parameterized.expand(FLWR_AID_MISMATCH_CASES)  # type: ignore
    def test_stoprun_auth_unsuccessful(
        self, context_flwr_aid: str | None, run_flwr_aid: str | None
    ) -> None:
        """Test StopRun unsuccessful with missing or mismatched flwr_aid."""
        # Prepare
        run_id = self._create_dummy_run(run_flwr_aid)
        request = StopRunRequest(run_id=run_id)
        ctx = self.make_context()

        # Execute & Assert
        with patch(
            "flwr.superlink.servicer.control.control_servicer.shared_account_info",
            new=SimpleNamespace(get=lambda: SimpleNamespace(flwr_aid=context_flwr_aid)),
        ):
            with self.assertRaises(RuntimeError) as cm:
                self.servicer.StopRun(request, ctx)
            self.assertIn("PERMISSION_DENIED", str(cm.exception))

    def test_stoprun_auth_successful(self) -> None:
        """Test StopRun successful with matching flwr_aid."""
        # Prepare
        run_id = self._create_dummy_run("user-123")
        request = StopRunRequest(run_id=run_id)
        ctx = self.make_context()

        # Execute & Assert
        with patch(
            "flwr.superlink.servicer.control.control_servicer.shared_account_info",
            new=SimpleNamespace(get=lambda: SimpleNamespace(flwr_aid="user-123")),
        ):
            response = self.servicer.StopRun(request, ctx)
            self.assertTrue(response.success)
            run = self.state.get_run(run_id)
            self.assertEqual(cast(Run, run).status.status, Status.FINISHED)
            self.assertEqual(cast(Run, run).status.sub_status, SubStatus.STOPPED)

    # Test all invalid cases for ListRuns with authentication
    @parameterized.expand(FLWR_AID_MISMATCH_CASES)  # type: ignore
    def test_listruns_auth_unsuccessful(
        self, context_flwr_aid: str | None, run_flwr_aid: str | None
    ) -> None:
        """Test ListRuns unsuccessful with missing or mismatched flwr_aid."""
        # Prepare
        run_id = self._create_dummy_run(run_flwr_aid)
        request = ListRunsRequest(run_id=run_id)
        ctx = self.make_context()

        # Execute & Assert
        with patch(
            "flwr.superlink.servicer.control.control_servicer.shared_account_info",
            new=SimpleNamespace(get=lambda: SimpleNamespace(flwr_aid=context_flwr_aid)),
        ):
            with self.assertRaises(RuntimeError) as cm:
                self.servicer.ListRuns(request, ctx)
            self.assertIn("PERMISSION_DENIED", str(cm.exception))

    def test_listruns_auth_run_success(self) -> None:
        """Test ListRuns successful with matching flwr_aid."""
        # Prepare
        run_id = self._create_dummy_run("user-123")
        request = ListRunsRequest(run_id=run_id)
        ctx = self.make_context()

        # Execute & Assert
        with patch(
            "flwr.superlink.servicer.control.control_servicer.shared_account_info",
            new=SimpleNamespace(get=lambda: SimpleNamespace(flwr_aid="user-123")),
        ):
            response = self.servicer.ListRuns(request, ctx)
            self.assertEqual(set(response.run_dict.keys()), {run_id})
