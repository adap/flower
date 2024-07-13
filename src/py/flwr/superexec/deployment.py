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
from pathlib import Path
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
    """Deployment engine executor.

    Parameters
    ----------
    superlink: str (Default: "0.0.0.0:9091")
        Address of the `SuperLink` to connect to.
    root_certificates: Optional[str] (Default: None)
        Specifies the path to the PEM-encoded root certificate file for
        establishing secure HTTPS connections.
    flwr_dir: Optional[str] (Default: None)
        The path containing installed Flower Apps.
    """

    def __init__(
        self,
        superlink: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        root_certificates: Optional[str] = None,
        flwr_dir: Optional[str] = None,
    ) -> None:
        self.superlink = superlink
        if root_certificates is None:
            self.root_certificates = None
            self.root_certificates_bytes = None
        else:
            self.root_certificates = root_certificates
            self.root_certificates_bytes = Path(root_certificates).read_bytes()
        self.flwr_dir = flwr_dir
        self.stub: Optional[DriverStub] = None

    @override
    def set_config(
        self,
        config: Dict[str, str],
    ) -> None:
        """Set config arguments.

        Parameters
        ----------
        config : Dict[str, str]
            A dictionary for configuration values.
            Supported configuration key/value pairs:
            - "superlink": str
                The address of the SuperLink Driver API.
            - "root-certificates": str
                The path to the root certificates.
            - "flwr-dir": str
                The path to the Flower directory.
        """
        if not config:
            return
        if superlink_address := config.get("superlink"):
            self.superlink = superlink_address
        if root_certificates := config.get("root-certificates"):
            self.root_certificates = root_certificates
            self.root_certificates_bytes = Path(root_certificates).read_bytes()
        if flwr_dir := config.get("flwr-dir"):
            self.flwr_dir = flwr_dir

    def _connect(self) -> None:
        if self.stub is not None:
            return
        channel = create_channel(
            server_address=self.superlink,
            insecure=(self.root_certificates_bytes is None),
            root_certificates=self.root_certificates_bytes,
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
    ) -> Optional[RunTracker]:
        """Start run using the Flower Deployment Engine."""
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
                    f"--flwr-dir {self.flwr_dir}" if self.flwr_dir else "",
                    "--superlink",
                    self.superlink,
                    (
                        "--insecure"
                        if self.root_certificates is None
                        else f"--root-certificates {self.root_certificates}"
                    ),
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
