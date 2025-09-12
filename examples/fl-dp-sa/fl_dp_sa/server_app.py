"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

from typing import List, Tuple, Dict
from datasets import load_dataset
from flwr.common import Context, Metrics, ndarrays_to_parameters
from fl_dp_sa.task import Net, test, set_weights, load_data
from torch.utils.data import DataLoader
from flwr.server import (
    Driver,
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from fl_dp_sa.strategy import histFedAvg
from fl_dp_sa.task import get_transform

from fl_dp_sa.task import Net, get_weights

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(testloader, device):
    def evaluate(server_round, parameters_nd, config):

        net = Net()
        set_weights(net, parameters_nd)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    w_acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_num_examples =sum(num_examples for num_examples, _ in metrics)

    return {"accuracy": sum(w_acc) / total_num_examples}


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    # Initialize global model
    model_weights = get_weights(Net())
    parameters = ndarrays_to_parameters(model_weights)

    testset=load_dataset("ylecun/mnist")["test"]
    testloader=DataLoader(testset.with_transform(get_transform()), batch_size=32)

    # Note: The fraction_fit value is configured based on the DP hyperparameter `num-sampled-clients`.
    strategy = histFedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.0,
        min_fit_clients=20,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        on_fit_config_fn=fit_round,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu")
    )

    noise_multiplier = context.run_config["noise-multiplier"]
    clipping_norm = context.run_config["clipping-norm"]

    # strategy = DifferentialPrivacyClientSideFixedClipping(
    #     strategy,
    #     noise_multiplier=noise_multiplier,
    #     clipping_norm=clipping_norm,
    #     num_sampled_clients=num_sampled_clients,
    # )

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            max_weight=5000
        )
    )

    # Execute
    workflow(grid, context)
