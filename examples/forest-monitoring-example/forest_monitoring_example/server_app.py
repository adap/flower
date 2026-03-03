"""forest-monitoring-example: A Flower / PyTorch app."""

import torch
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
import numpy as np
import math
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path

from forest_monitoring_example.task import load_model, get_weights

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.model = model
        self.eval_hist = []
        self.fit_hist  = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        print("Server round =", server_round)

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        aggregated_metrics = aggregated_metrics or {}

        # Convert `Parameters` to `List[np.ndarray]`
        aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # Convert `List[np.ndarray]` to PyTorch`state_dict`
        params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
                
        print("Return aggregated parameters and metrics")
        return aggregated_parameters, aggregated_metrics
    

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[float]:
                
        # Aggregate evaluation metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Save evaluation metrics of all rounds:
        self.global_metrics = aggregated_metrics  # Store final metrics

        return aggregated_loss, aggregated_metrics
    



def aggregate_from_sums(metrics):
    # metrics: List[Tuple[num_examples, metrics_dict]]
    n_total = sum(n for n, _ in metrics)
    sse_total = sum(m["sse"] for _, m in metrics)
    sum_y_total = sum(m["sum_y"] for _, m in metrics)
    sum_y2_total = sum(m["sum_y2"] for _, m in metrics)
    sum_pred_total = sum(m.get("sum_pred", 0.0) for _, m in metrics)

    mse = sse_total / n_total
    rmse = math.sqrt(mse)
    sst = sum_y2_total - (sum_y_total ** 2) / n_total
    r2 = float("nan") if sst <= 0 else 1.0 - (sse_total / sst)

    mean_pred = sum_pred_total / n_total if sum_pred_total != 0 else float("nan")
    rmse_pct = float("nan") if (not mean_pred or math.isnan(mean_pred)) else 100.0 * rmse / mean_pred

    return {"rmse": rmse, "rmse%": rmse_pct, "r2": r2, "mse_orig": mse}


def weighted_average_fit(metrics):
    """Compute weighted average for training metrics."""
    loss = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss": sum(loss) / sum(examples)} if examples else {}



# Function to configure the evaluation (evaluate) phase
def flwer_config(server_round: int, num_rounds: int = None):
    """Return evaluation configuration dict for each round."""
    config = {
        "current_round": server_round,
        "final_round": num_rounds,  # Mark the final round
    }
    return config



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # model config:
    feature_size = context.run_config["feature-size"]
    t_years = context.run_config["t-years"]
    out_conv1 = context.run_config["out-conv1"]
    out_conv2 = context.run_config["out-conv2"]
    kernel_time = context.run_config["kernel-time"]
    pool_time1 = context.run_config["pool-time1"]
    dropout_conv = context.run_config["dropout-conv"]
    adaptive_pool_time = context.run_config["adaptive-pool-time"]
    use_adaptive_pool = context.run_config["use-adaptive-pool"]


    # Initialize model parameters
    net = load_model(feature_size, t_years, out_conv1, out_conv2, kernel_time, pool_time1, dropout_conv, adaptive_pool_time, use_adaptive_pool)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = SaveModelStrategy(
        model = net.to(DEVICE),
        #proximal_mu=0.05,  # The mu parameter (usually between 0.01 and 1.0)
        fit_metrics_aggregation_fn=weighted_average_fit,  # Aggregate training metrics
        evaluate_metrics_aggregation_fn=aggregate_from_sums,
        num_rounds=num_rounds,
        min_available_clients=context.run_config["min-available-clients"],  # Set the expected number of clients. At least 2 clients must be available. # TODO: put in context.run_config[]
        min_fit_clients=context.run_config["min-fit-clients"],  # # At least 2 clients must participate.  # TODO: put in context.run_config[]
        fraction_fit=context.run_config["fraction-fit"],  # Use all available clients
        fraction_evaluate=context.run_config["fraction-evaluate"],
        initial_parameters=parameters,
        on_fit_config_fn=lambda round: flwer_config(round, num_rounds),  # these are used internally in flower
        on_evaluate_config_fn=lambda round: flwer_config(round, num_rounds),  # these are used internally in flower
        )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
