import flwr as fl
from pathlib import Path


# Start Flower server
hist = fl.driver.start_driver(
    server_address="127.0.0.1:9091",
    config=fl.server.ServerConfig(num_rounds=3),
    root_certificates=Path("credentials/ca.crt").read_bytes(),
)

assert hist.losses_distributed[-1][1] == 0
