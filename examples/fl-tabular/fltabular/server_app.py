"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context

from fltabular.task import IncomeClassifier, get_weights


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    net = IncomeClassifier()
    params = ndarrays_to_parameters(get_weights(net))

    strategy = FedAvg(
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
