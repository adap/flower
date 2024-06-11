"""Deployment engine executor plugin."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.common.grpc import create_channel
from flwr.proto.driver_pb2 import CreateRunRequest  # pylint: disable=E0611
from flwr.proto.driver_pb2_grpc import DriverStub
from flwr.server.driver.grpc_driver import DEFAULT_SERVER_ADDRESS_DRIVER

from .executor import Executor, Run


class DeploymentEngine(Executor):
    """Deployment engine executor plugin."""

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

    def _create_run(self, fab_id: str, fab_version: str) -> int:
        if self.stub is None:
            self._connect()

        assert self.stub is not None

        req = CreateRunRequest(fab_id=fab_id, fab_version=fab_version)
        res = self.stub.CreateRun(request=req)
        return res.run_id

    def _install_fab(self, fab_file: bytes) -> Path:
        return install_from_fab(fab_file, None, True)

    def start_run(self, fab_file: bytes, ttl: Optional[float] = None) -> Run:
        """Start run using the Flower Deployment Engine."""
        _ = ttl
        fab_id, fab_version = get_fab_metadata(fab_file)

        run_id = self._create_run(fab_id, fab_version)

        fab_path = self._install_fab(fab_file)

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", str(fab_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return Run(
            run_id=run_id,
            proc=subprocess.Popen(
                [
                    "flower-server-app",
                    "build_demo.server:app",
                    "--run-id",
                    str(run_id),
                    "--insecure",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )


deployment_plugin = DeploymentEngine()
