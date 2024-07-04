from flwr.server import ServerApp, ServerConfig

from vertical_fl.strategy import Strategy
from vertical_fl.task import get_partitions_and_label

_, label = get_partitions_and_label()

# Start Flower server
app = ServerApp(
    config=ServerConfig(num_rounds=1000),
    strategy=Strategy(label),
)
