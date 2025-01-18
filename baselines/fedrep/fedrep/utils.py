"""fedrep: A Flower Baseline."""

from typing import Callable, Type, Union

from flwr.common import Context, Parameters
from flwr.server.strategy import FedAvg

from .constants import Algorithm, ModelDatasetName
from .models import (
    CNNCifar10,
    CNNCifar10ModelManager,
    CNNCifar10ModelSplit,
    CNNCifar100,
    CNNCifar100ModelManager,
    CNNCifar100ModelSplit,
)
from .strategy import FedRep


def get_create_model_fn(
    context: Context,
) -> tuple[
    Union[Callable[[], CNNCifar10], Callable[[], CNNCifar100]],
    Union[Type[CNNCifar10ModelSplit], Type[CNNCifar100ModelSplit]],
]:
    """Get create model function."""
    model_name = str(context.run_config["model-name"])
    if model_name == ModelDatasetName.CNN_CIFAR_10.value:
        split = CNNCifar10ModelSplit

        def create_model() -> CNNCifar10:  # type: ignore
            """Create initial CNNCifar10 model."""
            return CNNCifar10()

    elif model_name == ModelDatasetName.CNN_CIFAR_100.value:
        split = CNNCifar100ModelSplit

        def create_model() -> CNNCifar100:  # type: ignore
            """Create initial CNNCifar100 model."""
            return CNNCifar100()

    else:
        raise NotImplementedError(f"Not a recognized model name {model_name}.")
    return create_model, split


def get_model_manager_class(
    context: Context,
) -> Union[Type[CNNCifar10ModelManager], Type[CNNCifar100ModelManager]]:
    """Depending on the model name type return the corresponding model manager."""
    model_name = str(context.run_config["model-name"])
    if model_name.lower() == ModelDatasetName.CNN_CIFAR_10.value:
        model_manager_class = CNNCifar10ModelManager
    elif model_name.lower() == ModelDatasetName.CNN_CIFAR_100.value:
        model_manager_class = CNNCifar100ModelManager  # type: ignore
    else:
        raise NotImplementedError(
            f"Model {model_name} not implemented, please check model name."
        )
    return model_manager_class


def get_server_strategy(
    context: Context, params: Parameters, eval_fn: Callable
) -> Union[FedAvg, FedRep]:
    """Define server strategy based on input algorithm."""
    algorithm = str(context.run_config["algorithm"]).lower()
    if algorithm == Algorithm.FEDAVG.value:
        strategy = FedAvg
    elif algorithm == Algorithm.FEDREP.value:
        strategy = FedRep
    else:
        raise RuntimeError(f"Unknown algorithm {algorithm}.")

    # Read strategy config
    fraction_fit = float(context.run_config["fraction-fit"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    min_available_clients = int(context.run_config["min-available-clients"])

    strategy = strategy(
        fraction_fit=float(fraction_fit),
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=eval_fn,
    )  # type: ignore
    return strategy  # type: ignore
