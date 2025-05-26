"""Server app for serverless federated learning with PyTorch."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents


# Define metric aggregation function
def weighted_average(metrics):
    """Aggregate metrics using weighted average."""
    # This is a placeholder implementation
    # In a real scenario, you would aggregate metrics from multiple clients
    return {"accuracy": 0.0}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    return ServerAppComponents()
    
# Create ServerApp
app = ServerApp(server_fn=server_fn)

def main():
    app.run()
    print("Server initialized and waiting for clients...")

if __name__ == "__main__":
    main() 