from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from task import IncomeClassifier, get_weights
from flwr.common import ndarrays_to_parameters

net = IncomeClassifier()
params = ndarrays_to_parameters(get_weights(net))


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


strategy = FedAvg(
    initial_parameters=params,
    evaluate_metrics_aggregation_fn=weighted_average,
)
app = ServerApp(
    strategy=strategy,
    config=ServerConfig(num_rounds=5),
)
