"""Deployment engine executor plugin."""

import subprocess
from typing import Optional

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

    def _connect(self) -> None:
        if self.stub is None:
            channel = create_channel(
                server_address=self.address,
                insecure=(self.root_certificates is None),
                root_certificates=self.root_certificates,
            )
            self.stub = DriverStub(channel)

    def _create_run(self, fab_id, fab_version) -> int:
        if self.stub is None:
            self._connect()
        if self.run_id is None:
            req = CreateRunRequest(fab_id=fab_id, fab_version=fab_version)
            res = self.stub.CreateRun(request=req)
            self.run_id = res.run_id
        return self.run_id

    def start_run(self, fab_file: bytes, ttl: Optional[float] = None) -> Run:
        """Echos success."""
        _ = fab_file
        _ = ttl
        return Run(
            run_id=10,
            proc=subprocess.Popen(
                ["sh", "-c", "for i in {1..5}; do echo $i; sleep 2; done"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )


deployment_plugin = DeploymentEngine()
