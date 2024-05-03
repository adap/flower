import argparse
import flwr as fl
import logging
from strategy.strategy import FedCustom
from prometheus_client import start_http_server, Gauge

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a gauge to track the global model accuracy
accuracy_gauge = Gauge("model_accuracy", "Current accuracy of the global model")

# Define a gauge to track the global model loss
loss_gauge = Gauge("model_loss", "Current loss of the global model")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower Server")
parser.add_argument(
    "--number_of_rounds",
    type=int,
    default=100,
    help="Number of FL rounds (default: 100)",
)
args = parser.parse_args()


# Function to Start Federated Learning Server
def start_fl_server(strategy, rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"FL Server error: {e}", exc_info=True)


# Main Function
if __name__ == "__main__":
    # Start Prometheus Metrics Server
    start_http_server(8000)

    # Initialize Strategy Instance and Start FL Server
    strategy_instance = FedCustom(accuracy_gauge=accuracy_gauge, loss_gauge=loss_gauge)
    start_fl_server(strategy=strategy_instance, rounds=args.number_of_rounds)
