from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from task import IncomeClassifier, get_weights
from flwr.common import ndarrays_to_parameters

net = IncomeClassifier()
params = ndarrays_to_parameters(get_weights(net))
strategy = FedAvg(
    initial_parameters=params,
)

app = ServerApp(
    strategy=strategy,
    config=ServerConfig(num_rounds=5),
)
