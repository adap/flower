# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Deployment engine executor."""

import subprocess
import sys
from logging import ERROR, INFO
from typing import Dict, Optional

from typing_extensions import override

from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.proto.driver_pb2 import CreateRunRequest  # pylint: disable=E0611
from flwr.proto.driver_pb2_grpc import DriverStub
from flwr.server.driver.grpc_driver import DEFAULT_SERVER_ADDRESS_DRIVER

from .executor import Executor, RunTracker


class DeploymentEngine(Executor):
    """Deployment engine executor."""

    def __init__(
        self,
        address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        root_certificates: Optional[bytes] = None,
    ) -> None:
        self.address = address
        self.root_certificates = root_certificates
        self.stub: Optional[DriverStub] = None

    def _connect(self) -> None:
        if self.stub is None:
            channel = create_channel(
                server_address=self.address,
                insecure=(self.root_certificates is None),
                root_certificates=self.root_certificates,
            )
            self.stub = DriverStub(channel)

    def _create_run(
        self,
        fab_id: str,
        fab_version: str,
        override_config: Dict[str, str],
    ) -> int:
        if self.stub is None:
            self._connect()

        assert self.stub is not None

        req = CreateRunRequest(
            fab_id=fab_id,
            fab_version=fab_version,
            override_config=override_config,
        )
        res = self.stub.CreateRun(request=req)
        return int(res.run_id)

    @override
    def start_run(
        self,
        fab_file: bytes,
        override_config: Dict[str, str],
        config: Optional[Dict[str, str]],
    ) -> Optional[RunTracker]:
        """Start run using the Flower Deployment Engine."""
        if config:
            superlink_address = config.get("superlink_address", None)
            if superlink_address:
                self.address = superlink_address

        try:
            # Install FAB to flwr dir
            fab_version, fab_id = get_fab_metadata(fab_file)
            fab_path = install_from_fab(fab_file, None, True)

            # Install FAB Python package
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", str(fab_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Call SuperLink to create run
            run_id: int = self._create_run(fab_id, fab_version, override_config)
            log(INFO, "Created run %s", str(run_id))

            # Start ServerApp
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    "flower-server-app",
                    "--run-id",
                    str(run_id),
                    "--insecure",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            log(INFO, "Started run %s", str(run_id))

            return RunTracker(
                run_id=run_id,
                proc=proc,
            )
        # pylint: disable-next=broad-except
        except Exception as e:
            log(ERROR, "Could not start run: %s", str(e))
            return None


executor = DeploymentEngine()
