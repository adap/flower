"""huggingface_example: A Flower / Hugging Face app."""

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

# if __name__ == "__main__":
# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

#    # Start server
#    fl.server.start_server(
#        server_address="0.0.0.0:8080",
#        config=fl.server.ServerConfig(num_rounds=3),
#        strategy=strategy,
#    )

config = ServerConfig(num_rounds=3)

app = ServerApp(
    config=config,
    strategy=strategy,
)
