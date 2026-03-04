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
"""Test the ClientAppIo API servicer."""

import tempfile
import unittest
from unittest.mock import Mock

import grpc
from parameterized import parameterized

from flwr.common import Context, typing
from flwr.common.constant import APP_TOKEN_HEADER
from flwr.common.message import make_message
from flwr.common.serde import fab_to_proto, message_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.proto import clientappio_pb2  # pylint:disable=E0611
from flwr.proto.appio_pb2 import (  # pylint:disable=E0611
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
    PullAppInputsResponse,
    PullAppMessagesResponse,
    PushAppMessagesResponse,
    PushAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint:disable=E0611
    SendAppHeartbeatRequest,
    SendAppHeartbeatResponse,
)
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint:disable=E0611
from flwr.proto.message_pb2 import (  # pylint:disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint:disable=E0611
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint:disable=E0611
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
    iterate_object_tree,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supernode.nodestate import NodeStateFactory
from flwr.supernode.runtime.run_clientapp import pull_appinputs, push_appoutputs
from flwr.supernode.start_client_internal import (
    CLIENTAPPIO_METHOD_REQUIRES_TOKEN,
    run_clientappio_api_grpc,
)

from .clientappio_servicer import ClientAppIoServicer

CLIENTAPPIO_TEST_ADDRESS = "127.0.0.1:19094"


def test_clientappio_token_policy_covers_all_proto_rpcs() -> None:
    """Ensure every ClientAppIo RPC has an explicit token policy decision."""
    # Reason: avoid auth gaps if a new RPC is added to ClientAppIo.
    # If this fails, update CLIENTAPPIO_METHOD_REQUIRES_TOKEN in
    # `py/flwr/supernode/start_client_internal.py`.
    service_name = "ClientAppIo"
    package_name = clientappio_pb2.DESCRIPTOR.package
    rpc_names = [
        method.name
        for method in clientappio_pb2.DESCRIPTOR.services_by_name[service_name].methods
    ]
    expected_methods = {
        f"/{package_name}.{service_name}/{rpc_name}" for rpc_name in rpc_names
    }
    configured_methods = set(CLIENTAPPIO_METHOD_REQUIRES_TOKEN)
    assert configured_methods == expected_methods, (
        "ClientAppIo token policy table is out of sync with proto RPCs. "
        "Update CLIENTAPPIO_METHOD_REQUIRES_TOKEN in "
        "`py/flwr/supernode/start_client_internal.py`."
    )


class TestClientAppIoServicer(unittest.TestCase):
    """Tests for `ClientAppIoServicer` class."""

    def setUp(self) -> None:
        """Initialize."""
        self.maker = RecordMaker()
        self.mock_stub = Mock()
        self.mock_state = Mock()
        mock_state_factory = Mock()
        mock_state_factory.state.return_value = self.mock_state
        self.servicer = ClientAppIoServicer(mock_state_factory, Mock(), Mock())

    def test_pull_clientapp_inputs(self) -> None:
        """Test pulling messages from SuperNode."""
        # Prepare
        mock_message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(3, 2, 1),
        )
        mock_fab = typing.Fab(
            hash_str="abc123#$%",
            content=b"\xf3\xf5\xf8\x98",
            verifications={"ab12#$%": "abc123#$%"},
        )
        mock_response = PullAppInputsResponse(
            context=ProtoContext(node_id=123),
            run=ProtoRun(run_id=61016, fab_id="mock/mock", fab_version="v1.0.0"),
            fab=fab_to_proto(mock_fab),
        )
        self.mock_stub.PullMessage.return_value = PullAppMessagesResponse(
            messages_list=[message_to_proto(mock_message)],
            message_object_trees=[get_object_tree(mock_message)],
        )
        # Create series of responses for PullObject
        # Adding responses for objects in a post-order traversal of object tree order
        all_objects = get_all_nested_objects(mock_message)
        all_objects[mock_message.object_id] = mock_message

        # Get the object tree and iterate in the correct order
        def pull_object_side_effect(request: PullObjectRequest) -> PullObjectResponse:
            obj_id = request.object_id
            return PullObjectResponse(
                object_found=True,
                object_available=True,
                object_content=all_objects[obj_id].deflate(),
            )

        self.mock_stub.PullObject.side_effect = pull_object_side_effect
        self.mock_stub.PullAppInputs.return_value = mock_response

        # Execute
        message, context, run, fab = pull_appinputs(self.mock_stub, token="abc")

        # Assert
        self.mock_stub.PullAppInputs.assert_called_once()
        req = self.mock_stub.PullAppInputs.call_args.args[0]
        self.assertEqual(req.token, "")
        self.assertEqual(len(message.content.array_records), 3)
        self.assertEqual(len(message.content.metric_records), 2)
        self.assertEqual(len(message.content.config_records), 1)
        self.assertEqual(context.node_id, 123)
        self.assertEqual(run.run_id, 61016)
        self.assertEqual(run.fab_id, "mock/mock")
        self.assertEqual(run.fab_version, "v1.0.0")
        if fab:
            self.assertEqual(fab.hash_str, mock_fab.hash_str)
            self.assertEqual(fab.content, mock_fab.content)

    def test_push_clientapp_outputs(self) -> None:
        """Test pushing messages to SuperNode."""
        # Prepare: Create Message and context
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(2, 2, 1),
        )
        context = Context(
            run_id=1,
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recorddict(2, 2, 1),
            run_config={"runconfig1": 6.1},
        )

        # Prepare: Mock PushAppOutputs RPC call
        mock_response = PushAppOutputsResponse()
        self.mock_stub.PushAppOutputs.return_value = mock_response

        # Prepare: Mock PushMessage RPC call
        object_tree = get_object_tree(message)
        all_obj_ids = [tree.object_id for tree in iterate_object_tree(object_tree)]
        self.mock_stub.PushMessage.return_value = PushAppMessagesResponse(
            message_ids=[message.object_id],
            objects_to_push=all_obj_ids,
        )

        # Prepare: Mock PushObject RPC calls
        pushed_obj_ids = set()

        def mock_push_object(request: PushObjectRequest) -> PushObjectResponse:
            """Mock PushObject RPC call."""
            pushed_obj_ids.add(request.object_id)
            return PushObjectResponse(stored=True)

        self.mock_stub.PushObject.side_effect = mock_push_object

        # Execute
        _ = push_appoutputs(
            stub=self.mock_stub, token="abc", message=message, context=context
        )

        # Assert
        self.mock_stub.PushAppOutputs.assert_called_once()
        self.mock_stub.PushMessage.assert_called_once()
        push_msg_req = self.mock_stub.PushMessage.call_args.args[0]
        push_out_req = self.mock_stub.PushAppOutputs.call_args.args[0]
        self.assertEqual(push_msg_req.run_id, context.run_id)
        self.assertEqual(push_msg_req.token, "")
        self.assertEqual(push_out_req.run_id, context.run_id)
        self.assertEqual(push_out_req.token, "")
        self.assertSetEqual(pushed_obj_ids, set(all_obj_ids))

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_send_app_heartbeat(self, success: bool) -> None:
        """Test sending an app heartbeat."""
        # Prepare
        token = "test-token"
        request = SendAppHeartbeatRequest(token=token)
        context = Mock(**{"_flwr_appio_authenticated_token": token})
        self.mock_state.acknowledge_app_heartbeat.return_value = success

        # Execute
        response = self.servicer.SendAppHeartbeat(request, context)

        # Assert
        self.assertIsInstance(response, SendAppHeartbeatResponse)
        self.assertEqual(response.success, success)
        self.mock_state.acknowledge_app_heartbeat.assert_called_once_with(token)


class TestClientAppIoGrpcTokenAuth(  # pylint: disable=too-many-instance-attributes
    unittest.TestCase
):
    """Integration tests for ClientAppIo token interceptor wiring."""

    def setUp(self) -> None:
        """Start ClientAppIo gRPC server and create stubs for test RPCs."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)

        self.objectstore_factory = ObjectStoreFactory()
        self.state_factory = NodeStateFactory(
            objectstore_factory=self.objectstore_factory
        )
        self.state = self.state_factory.state()
        self.ffs_factory = FfsFactory(self.temp_dir.name)

        self._server: grpc.Server = run_clientappio_api_grpc(
            CLIENTAPPIO_TEST_ADDRESS,
            self.state_factory,
            self.ffs_factory,
            self.objectstore_factory,
            None,
        )
        self._channel = grpc.insecure_channel(CLIENTAPPIO_TEST_ADDRESS)
        self._list_apps_to_launch = self._channel.unary_unary(
            "/flwr.proto.ClientAppIo/ListAppsToLaunch",
            request_serializer=ListAppsToLaunchRequest.SerializeToString,
            response_deserializer=ListAppsToLaunchResponse.FromString,
        )
        self._request_token = self._channel.unary_unary(
            "/flwr.proto.ClientAppIo/RequestToken",
            request_serializer=RequestTokenRequest.SerializeToString,
            response_deserializer=RequestTokenResponse.FromString,
        )
        self._get_run = self._channel.unary_unary(
            "/flwr.proto.ClientAppIo/GetRun",
            request_serializer=GetRunRequest.SerializeToString,
            response_deserializer=GetRunResponse.FromString,
        )
        self._send_app_heartbeat = self._channel.unary_unary(
            "/flwr.proto.ClientAppIo/SendAppHeartbeat",
            request_serializer=SendAppHeartbeatRequest.SerializeToString,
            response_deserializer=SendAppHeartbeatResponse.FromString,
        )
        self._push_object = self._channel.unary_unary(
            "/flwr.proto.ClientAppIo/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )

    def tearDown(self) -> None:
        """Stop ClientAppIo gRPC server."""
        self._server.stop(None)

    def test_superexec_methods_still_allow_unauthenticated_calls(self) -> None:
        """ListAppsToLaunch/RequestToken/GetRun remain callable without metadata."""
        list_res, list_call = self._list_apps_to_launch.with_call(
            ListAppsToLaunchRequest()
        )
        token_res, token_call = self._request_token.with_call(
            RequestTokenRequest(run_id=1)
        )
        run_res, run_call = self._get_run.with_call(GetRunRequest(run_id=1))

        self.assertIsInstance(list_res, ListAppsToLaunchResponse)
        self.assertEqual(list_call.code(), grpc.StatusCode.OK)
        self.assertIsInstance(token_res, RequestTokenResponse)
        self.assertTrue(token_res.token)
        self.assertEqual(token_call.code(), grpc.StatusCode.OK)
        self.assertIsInstance(run_res, GetRunResponse)
        self.assertEqual(run_call.code(), grpc.StatusCode.OK)

    def test_send_app_heartbeat_requires_token_metadata(self) -> None:
        """Token-protected methods reject calls without App token metadata."""
        with self.assertRaises(grpc.RpcError) as err:
            self._send_app_heartbeat.with_call(SendAppHeartbeatRequest(token="unused"))
        self.assertEqual(err.exception.code(), grpc.StatusCode.PERMISSION_DENIED)
        self.assertEqual(err.exception.details(), "Invalid token.")

    def test_send_app_heartbeat_accepts_valid_metadata_token(self) -> None:
        """Token-protected methods accept valid metadata token."""
        token = self.state.create_token(11)
        assert token is not None
        response, call = self._send_app_heartbeat.with_call(
            SendAppHeartbeatRequest(token="different-value"),
            metadata=((APP_TOKEN_HEADER, token),),
        )

        self.assertTrue(response.success)
        self.assertEqual(call.code(), grpc.StatusCode.OK)

    def test_token_metadata_run_id_mismatch_denied(self) -> None:
        """Token-protected run-scoped methods deny mismatched request.run_id."""
        token = self.state.create_token(11)
        assert token is not None
        with self.assertRaises(grpc.RpcError) as err:
            self._push_object.with_call(
                PushObjectRequest(run_id=99),
                metadata=((APP_TOKEN_HEADER, token),),
            )
        self.assertEqual(err.exception.code(), grpc.StatusCode.PERMISSION_DENIED)
        self.assertEqual(err.exception.details(), "Invalid token.")
