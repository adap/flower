from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvg
import wandb


class VisualiseFedAvg(FedAvg):
    """Adapted FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]

        for fit_metric in fit_metrics:
            data = {}

            for key, value in fit_metric[1].items():
                if key in ["cid", "gen_image", "true_image", "latent_rep"]:
                    continue
                data = {
                    f"train_{key}_{fit_metric[1]['cid']}": value,
                }

            data[f"train_num_examples_{fit_metric[1]['cid']}"] = fit_metric[0]
            data[f"train_true_image_{fit_metric[1]['cid']}"] = wandb.Image(
                fit_metric[1]["true_image"]
            )
            data[f"train_gen_image_{fit_metric[1]['cid']}"] = wandb.Image(
                fit_metric[1]["gen_image"]
            )
            data[f"train_latent_rep_{fit_metric[1]['cid']}"] = fit_metric[1][
                "latent_rep"
            ]

            wandb.log(data=data, step=server_round)
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]

        for eval_metric in eval_metrics:
            data = {}
            for key, value in eval_metric[1].items():
                if key in ["cid"]:
                    continue
                data = {
                    f"eval_{key}_{eval_metric[1]['cid']}": value,
                }
            data[f"eval_num_examples_{eval_metric[1]['cid']}"] = eval_metric[0]

            wandb.log(data=data, step=server_round)
        data_agg = {
            f"eval_{key}_aggregated": value for key, value in metrics_aggregated.items()
        }
        wandb.log(
            data={
                "eval_loss_aggregated": loss_aggregated,
                **data_agg,
            },
            step=server_round,
        )
        return loss_aggregated, metrics_aggregated
