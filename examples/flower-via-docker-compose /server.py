import argparse
import flwr as fl
import logging
from strategy.strategy import FedCustom

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flower Server')
parser.add_argument('--number_of_rounds', type=int, default=10)
parser.add_argument('--total_clients', type=int, default=2)
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
    strategy_instance = FedCustom(total_clients = args.total_clients)
    start_fl_server(strategy_instance, args.number_of_rounds)