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
"""Control API server."""


from logging import INFO
from typing import Optional

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.auth_plugin import ControlAuthPlugin, ControlAuthzPlugin
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.proto.control_pb2_grpc import add_ControlServicer_to_server
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.license_plugin import LicensePlugin
from flwr.supercore.object_store import ObjectStoreFactory

from ...executor import Executor
from .control_event_log_interceptor import ControlEventLogInterceptor
from .control_license_interceptor import ControlLicenseInterceptor
from .control_servicer import ControlServicer
from .control_user_auth_interceptor import ControlUserAuthInterceptor

try:
    from flwr.ee import get_license_plugin
except ImportError:

    def get_license_plugin() -> Optional[LicensePlugin]:
        """Return the license plugin."""


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_control_api_grpc(
    address: str,
    executor: Executor,
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: Optional[tuple[bytes, bytes, bytes]],
    config: UserConfig,
    auth_plugin: Optional[ControlAuthPlugin] = None,
    authz_plugin: Optional[ControlAuthzPlugin] = None,
    event_log_plugin: Optional[EventLogWriterPlugin] = None,
) -> grpc.Server:
    """Run Control API (gRPC, request-response)."""
    executor.set_config(config)

    license_plugin: Optional[LicensePlugin] = get_license_plugin()
    if license_plugin and not license_plugin.check_license():
        flwr_exit(ExitCode.SUPERLINK_LICENSE_INVALID)

    control_servicer: grpc.Server = ControlServicer(
        linkstate_factory=state_factory,
        ffs_factory=ffs_factory,
        objectstore_factory=objectstore_factory,
        executor=executor,
        auth_plugin=auth_plugin,
    )
    interceptors: list[grpc.ServerInterceptor] = []
    if license_plugin is not None:
        interceptors.append(ControlLicenseInterceptor(license_plugin))
    if auth_plugin is not None and authz_plugin is not None:
        interceptors.append(ControlUserAuthInterceptor(auth_plugin, authz_plugin))
    # Event log interceptor must be added after user auth interceptor
    if event_log_plugin is not None:
        interceptors.append(ControlEventLogInterceptor(event_log_plugin))
        log(INFO, "Flower event logging enabled")
    control_add_servicer_to_server_fn = add_ControlServicer_to_server
    control_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(control_servicer, control_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors or None,
    )

    if auth_plugin is None:
        log(INFO, "Flower Deployment Engine: Starting Control API on %s", address)
    else:
        log(
            INFO,
            "Flower Deployment Engine: Starting Control API with user "
            "authentication on %s",
            address,
        )
    control_grpc_server.start()

    return control_grpc_server
