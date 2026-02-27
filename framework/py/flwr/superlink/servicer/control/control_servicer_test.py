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
import json
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
    FEDERATION_NOT_SPECIFIED_MESSAGE,
    NODE_NOT_FOUND_MESSAGE,
    NOOP_ACCOUNT_NAME,
    PUBLIC_KEY_ALREADY_IN_USE_MESSAGE,
    PUBLIC_KEY_NOT_VALID,
    SUPERLINK_DOES_NOT_SUPPORT_FED_MANAGEMENT_MESSAGE,
    Status,
    SubStatus,
)
from flwr.common.typing import Run, RunStatus
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AddNodeToFederationRequest,
    AddNodeToFederationResponse,
    ArchiveFederationRequest,
    CreateFederationRequest,
    ListNodesRequest,
    ListNodesResponse,
    ListRunsRequest,
    RegisterNodeRequest,
    RemoveNodeFromFederationRequest,
    RemoveNodeFromFederationResponse,
    ShowFederationRequest,
    ShowFederationResponse,
    StartRunRequest,
    StopRunRequest,
    StreamLogsRequest,
    StreamLogsResponse,
    UnregisterNodeRequest,
)
from flwr.proto.federation_pb2 import Account, Member  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.control.control_account_auth_interceptor import (
    shared_account_info,
)

from .control_servicer import (
    ControlServicer,
    _format_verification,
    _validate_federation_and_node_in_request,
)

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
        objectstore_factory = Mock(store=Mock(return_value=self.store))
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
            ),
            ffs_factory=FfsFactory(self.tmp_dir.name),
            objectstore_factory=objectstore_factory,
            is_simulation=False,
            authn_plugin=(authn_plugin := NoOpControlAuthnPlugin(Mock(), False)),
        )
        account_info = authn_plugin.validate_tokens_in_metadata([])[1]
        assert account_info is not None
        assert account_info.flwr_aid is not None
        self.aid: str = account_info.flwr_aid
        shared_account_info.set(account_info)
        self.state = self.servicer.linkstate_factory.state()

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.tmp_dir.cleanup()

    def _create_dummy_run(self, flwr_aid: str | None) -> int:
        return self.state.create_run(
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            NOOP_FEDERATION,
            ConfigRecord(),
            flwr_aid,
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

    @parameterized.expand([(None,), (1,), (2,), (3,), (9,)])  # type: ignore
    def test_list_runs(self, limit: int | None) -> None:
        """Test List method of ControlServicer with --runs option."""
        # Prepare
        run_ids = [self._create_dummy_run(self.aid) for _ in range(3)]

        # Execute
        response = self.servicer.ListRuns(ListRunsRequest(limit=limit), Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        limit = limit or 999
        self.assertLess(abs(retrieved_timestamp - now().timestamp()), 1e-3)
        self.assertEqual(set(response.run_dict.keys()), set(run_ids[-limit:]))

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
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(flwr_aid=flwr_aid_retrieving),
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

    def test_show_federation(self) -> None:
        """Test ShowFederation method of ControlServicer."""
        # Prepare
        request = ShowFederationRequest(federation_name=NOOP_FEDERATION)

        # Execute
        response: ShowFederationResponse = self.servicer.ShowFederation(request, Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        self.assertLess(abs(retrieved_timestamp - now().timestamp()), 1e-3)
        self.assertEqual(response.federation.name, NOOP_FEDERATION)

    def test_create_federation_success(self) -> None:
        """Test CreateFederation succeeds when federation_manager.create_federation
        works."""
        # Prepare
        name = "test-federation"
        description = "A test federation"
        expected_name = f"@{NOOP_ACCOUNT_NAME}/{name}"
        request = CreateFederationRequest(
            federation_name=name,
            description=description,
        )
        mock_members = [
            Member(account=Account(id=self.aid), role="owner"),
        ]
        mock_federation = SimpleNamespace(
            name=expected_name,
            description=description,
            members=mock_members,
        )

        # Execute
        with patch.object(
            self.state.federation_manager,
            "create_federation",
            return_value=mock_federation,
        ) as mock_create:
            response = self.servicer.CreateFederation(request, Mock())

        # Assert
        mock_create.assert_called_once_with(
            name=expected_name,
            description=description,
            flwr_aid=self.aid,
        )
        self.assertEqual(response.federation.name, expected_name)
        self.assertEqual(response.federation.description, description)
        self.assertEqual(len(response.federation.members), 1)
        self.assertEqual(response.federation.members[0].account.id, self.aid)
        self.assertEqual(response.federation.members[0].role, "owner")

    def test_create_federation_fails_on_manager_error(self) -> None:
        """Test CreateFederation aborts when federation_manager.create_federation
        raises."""
        # Prepare
        name = "test-federation"
        description = "A test federation"
        request = CreateFederationRequest(
            federation_name=name,
            description=description,
        )
        mock_context = Mock()
        mock_context.abort.side_effect = grpc.RpcError()

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self.servicer.CreateFederation(request, mock_context)

        mock_context.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION,
            SUPERLINK_DOES_NOT_SUPPORT_FED_MANAGEMENT_MESSAGE,
        )

    def test_archive_federation_success(self) -> None:
        """Test ArchiveFederation succeeds when federation_manager.archive_federation
        works."""
        # Prepare
        request = ArchiveFederationRequest(federation_name="test-federation")

        # Execute
        with patch.object(
            self.state.federation_manager,
            "archive_federation",
            return_value=None,
        ) as mock_archive:
            response = self.servicer.ArchiveFederation(request, Mock())

        # Assert
        mock_archive.assert_called_once_with(
            flwr_aid=self.aid,
            name="test-federation",
        )
        self.assertIsNotNone(response)

    def test_archive_federation_fails_on_manager_error(self) -> None:
        """Test ArchiveFederation aborts when federation_manager.archive_federation
        raises."""
        # Prepare
        name = "test-federation"
        request = ArchiveFederationRequest(federation_name=name)
        mock_context = Mock()
        mock_context.abort.side_effect = grpc.RpcError()

        # Execute & Assert
        with self.assertRaises(grpc.RpcError):
            self.servicer.ArchiveFederation(request, mock_context)

        mock_context.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION,
            SUPERLINK_DOES_NOT_SUPPORT_FED_MANAGEMENT_MESSAGE,
        )


class TestControlServicerAuth(unittest.TestCase):
    """Test ControlServicer methods with authentication plugin and flwr_aid checking."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), Mock()
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
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            NOOP_FEDERATION,
            ConfigRecord(),
            flwr_aid,
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
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(flwr_aid=context_flwr_aid),
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
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(flwr_aid="user-123"),
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
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(flwr_aid=context_flwr_aid),
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
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(flwr_aid="user-123"),
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
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(flwr_aid=context_flwr_aid),
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
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(flwr_aid="user-123"),
        ):
            response = self.servicer.ListRuns(request, ctx)
            self.assertEqual(set(response.run_dict.keys()), {run_id})


class TestValidateFederationAndNodesInRequest(unittest.TestCase):
    """Tests for the _validate_federation_and_node_in_request helper."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        objectstore_factory = Mock(store=Mock(return_value=Mock()))
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
            ),
            ffs_factory=FfsFactory(self.tmp_dir.name),
            objectstore_factory=objectstore_factory,
            is_simulation=False,
            authn_plugin=(authn_plugin := NoOpControlAuthnPlugin(Mock(), False)),
        )
        account_info = authn_plugin.validate_tokens_in_metadata([])[1]
        assert account_info is not None
        assert account_info.flwr_aid is not None
        self.aid: str = account_info.flwr_aid
        shared_account_info.set(account_info)
        self.state = self.servicer.linkstate_factory.state()

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.tmp_dir.cleanup()

    def _make_context(self) -> MagicMock:
        """Create a mock gRPC context that raises on abort."""
        ctx = MagicMock(spec=grpc.ServicerContext)
        ctx.abort.side_effect = lambda code, msg: (_ for _ in ()).throw(
            RuntimeError(f"{code}:{msg}")
        )
        return ctx

    def _create_owned_node(self, owner_aid: str) -> int:
        """Create a node owned by the given flwr_aid."""
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        return self.state.create_node(
            owner_aid=owner_aid,
            owner_name="test_owner",
            public_key=pub_key,
            heartbeat_interval=10,
        )

    # --- _validate_federation_and_node_in_request tests ---

    def test_validate_aborts_when_federation_not_specified(self) -> None:
        """Test abort when federation name is empty."""
        ctx = self._make_context()
        with self.assertRaises(RuntimeError) as cm:
            _validate_federation_and_node_in_request(self.state, self.aid, "", 1, ctx)
        ctx.abort.assert_called_once()
        self.assertIn(FEDERATION_NOT_SPECIFIED_MESSAGE, str(cm.exception))

    def test_validate_aborts_when_federation_not_found(self) -> None:
        """Test abort when federation does not exist."""
        ctx = self._make_context()
        with self.assertRaises(RuntimeError) as cm:
            _validate_federation_and_node_in_request(
                self.state, self.aid, "nonexistent-fed", 1, ctx
            )
        ctx.abort.assert_called_once()
        self.assertIn("nonexistent-fed", str(cm.exception))

    def test_validate_aborts_when_not_a_member(self) -> None:
        """Test abort when flwr_aid is not a member of the federation."""
        ctx = self._make_context()
        with self.assertRaises(RuntimeError) as cm:
            _validate_federation_and_node_in_request(
                self.state, "wrong-aid", NOOP_FEDERATION, 1, ctx
            )
        ctx.abort.assert_called_once()
        self.assertIn("not a member", str(cm.exception))

    def test_validate_aborts_when_node_not_owned(self) -> None:
        """Test abort when a node is not owned by the requester."""
        # Create a node owned by someone else
        node_id = self._create_owned_node("other-aid")
        ctx = self._make_context()
        with self.assertRaises(RuntimeError) as cm:
            _validate_federation_and_node_in_request(
                self.state, self.aid, NOOP_FEDERATION, node_id, ctx
            )
        ctx.abort.assert_called_once()
        self.assertIn("not found or you are not its owner", str(cm.exception))

    def test_validate_aborts_when_node_does_not_exist(self) -> None:
        """Test abort when a node ID does not exist."""
        ctx = self._make_context()
        with self.assertRaises(RuntimeError) as cm:
            _validate_federation_and_node_in_request(
                self.state, self.aid, NOOP_FEDERATION, 999999, ctx
            )
        ctx.abort.assert_called_once()
        self.assertIn("not found or you are not its owner", str(cm.exception))

    # --- AddNodeToFederation / RemoveNodeFromFederation integration tests ---

    def test_add_node_to_federation_success(self) -> None:
        """Test AddNodeToFederation succeeds with valid inputs."""
        node_id = self._create_owned_node(self.aid)
        request = AddNodeToFederationRequest(
            federation_name=NOOP_FEDERATION, node_id=node_id
        )
        ctx = self._make_context()

        with patch.object(
            self.state.federation_manager,
            "add_supernode",
            return_value=None,
        ) as mock_add:
            response = self.servicer.AddNodeToFederation(request, ctx)

        mock_add.assert_called_once_with(
            flwr_aid=self.aid,
            federation=NOOP_FEDERATION,
            node_id=node_id,
        )
        self.assertIsInstance(response, AddNodeToFederationResponse)
        ctx.abort.assert_not_called()

    def test_add_node_to_federation_aborts_no_federation(self) -> None:
        """Test AddNodeToFederation aborts when no federation is specified."""
        request = AddNodeToFederationRequest(federation_name="", node_id=1)
        ctx = self._make_context()
        with self.assertRaises(RuntimeError) as cm:
            self.servicer.AddNodeToFederation(request, ctx)
        self.assertIn(FEDERATION_NOT_SPECIFIED_MESSAGE, str(cm.exception))

    def test_remove_node_from_federation_success(self) -> None:
        """Test RemoveNodeFromFederation succeeds with valid inputs."""
        node_id = self._create_owned_node(self.aid)
        request = RemoveNodeFromFederationRequest(
            federation_name=NOOP_FEDERATION, node_id=node_id
        )
        ctx = self._make_context()

        with patch.object(
            self.state.federation_manager,
            "remove_supernode",
            return_value=None,
        ) as mock_remove:
            response = self.servicer.RemoveNodeFromFederation(request, ctx)

        mock_remove.assert_called_once_with(
            flwr_aid=self.aid,
            federation=NOOP_FEDERATION,
            node_id=node_id,
        )
        self.assertIsInstance(response, RemoveNodeFromFederationResponse)
        ctx.abort.assert_not_called()

    def test_remove_node_from_federation_aborts_no_federation(self) -> None:
        """Test RemoveNodeFromFederation aborts when no federation is specified."""
        request = RemoveNodeFromFederationRequest(federation_name="", node_id=1)
        ctx = self._make_context()
        with self.assertRaises(RuntimeError) as cm:
            self.servicer.RemoveNodeFromFederation(request, ctx)
        self.assertIn(FEDERATION_NOT_SPECIFIED_MESSAGE, str(cm.exception))


def test_format_verification_compact() -> None:
    """One test covering both 'with entries' and 'None' input."""
    # Case 1: verifications list present
    verifications: list[dict[str, str]] = [
        {"public_key_id": "key1", "sig": "abc", "algo": "ed25519"},
        {"public_key_id": "key2", "sig": "def", "algo": "ed25519"},
    ]
    out: dict[str, str] = _format_verification(verifications)

    # Should mark valid
    assert out["valid_license"] == "Valid"
    # public_key_id -> JSON of remaining fields
    v1: dict[str, str] = json.loads(out["key1"])
    v2: dict[str, str] = json.loads(out["key2"])
    assert v1 == {"sig": "abc", "algo": "ed25519"}
    assert v2 == {"sig": "def", "algo": "ed25519"}
