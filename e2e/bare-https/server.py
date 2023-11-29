import flwr as fl
from pathlib import Path


hist = fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    credentials=(
        Path("credentials/ca.crt").read_bytes(),
        Path("credentials/server.pem").read_bytes(),
        Path("credentials/server.key").read_bytes(),
    ),
)

assert (
    hist.losses_distributed[-1][1] == 0
    or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98
)
