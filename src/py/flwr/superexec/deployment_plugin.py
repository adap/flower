"""Deployment engine executor plugin."""

import subprocess
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
        self.run_id = None
        self.stub = None

    def _connect(self) -> None:
        if self.stub is None:
            channel = create_channel(
                server_address=self.address,
                insecure=(self.root_certificates is None),
                root_certificates=self.root_certificates,
            )
            self.stub = DriverStub(channel)

    def _create_run(self, fab_id: str, fab_version: str) -> None:
        if self.stub is None:
            self._connect()

        assert self.stub is not None

        if self.run_id is None:
            req = CreateRunRequest(fab_id=fab_id, fab_version=fab_version)
            res = self.stub.CreateRun(request=req)
            self.run_id = res.run_id

    def _install_fab(self, fab_file: bytes) -> None:
        install_from_fab(fab_file, None, True)

    def start_run(self, fab_file: bytes, ttl: Optional[float] = None) -> Run:
        """Echos success."""
        _ = ttl
        fab_id, fab_version = get_fab_metadata(fab_file)

        if self.run_id is None:
            self._create_run(fab_id, fab_version)

        self._install_fab(fab_file)

        assert self.run_id is not None

        return Run(
            run_id=self.run_id,
            proc=subprocess.Popen(
                [
                    "flower-server-app",
                    "build_demo.server:app",
                    "--run-id",
                    str(self.run_id),
                    "--insecure",
                ],
                # [
                #     "echo",
                #     str(self.run_id),
                # ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )


deployment_plugin = DeploymentEngine()
