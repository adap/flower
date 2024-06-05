from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel

from logging import DEBUG
from flwr.common.logger import log
from flwr.proto.exec_pb2 import StartRunRequest
from flwr.proto.exec_pb2_grpc import ExecStub


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


channel = create_channel(
    server_address="127.0.0.1:9093",
    insecure=True,
    root_certificates=None,
    max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    interceptors=None,
)
channel.subscribe(on_channel_state_change)
stub = ExecStub(channel)

in_file = open(
    "charles.build-demo.1-0-0.fab", "rb"
)  # opening for [r]eading as [b]inary
data = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
in_file.close()

req = StartRunRequest(fab_file=data)
res = stub.StartRun(req)
