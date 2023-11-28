import argparse
import flwr as fl
import logging
from strategy.strategy import FedCustom
from prometheus_client import start_http_server, Gauge

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Prometheus Metrics
accuracy_gauge = Gauge('model_accuracy', 'Global model accuracy')
loss_gauge = Gauge('model_loss', 'Global model loss')

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
    
     # Start Prometheus Metrics Server
    start_http_server(8000) 
    
    # Initialize Strategy Instance and Start FL Server 
    strategy_instance = FedCustom(accuracy_gauge = accuracy_gauge, loss_gauge = loss_gauge, total_clients = args.total_clients)
    start_fl_server(strategy_instance, args.number_of_rounds)
 