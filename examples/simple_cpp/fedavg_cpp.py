from collections import OrderedDict
from io import BytesIO
from typing import Callable, Dict, List, Tuple

import numpy as np

from flwr.server.strategy import FedAvg
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
from typing import Optional
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from numpy import bytes_, numarray


class FedAvgLibCpp(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        self.internal_model = None

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        aggregated_weights = aggregate(weights_results)
        parameters_results = weights_to_parameters(
            aggregated_weights,
            tensor_type="libtorch",
            model=self.internal_model,
        )
        return (parameters_results, {})

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        list_with_one_client = client_manager.sample(num_clients=1, min_num_clients=1)
        initial_parameters_res = list_with_one_client[0].get_parameters()
        initial_parameters = initial_parameters_res.parameters

        if initial_parameters.tensor_type == "libtorch":
            self.internal_model = bytes_to_libtorch(initial_parameters.tensors[0])

        return initial_parameters


## Serialization methods


def parameters_to_weights(parameters: Parameters) -> Weights:
    """Convert parameters object to libtorch weights.
    There is only one 'layer'."""
    #print(parameters.tensor_type)
    if parameters.tensor_type == "libtorch":
        #print(parameters.tensors[0])
        model = bytes_to_libtorch(parameters.tensors[0])
        return libtorch_to_weights(model)


def bytes_to_libtorch(serialized_model: bytes) -> torch.nn.Module:
    model = torch.load(BytesIO(serialized_model))
    return model


def libtorch_to_weights(model: torch.nn.Module) -> Weights:
    #for _, val in model.state_dict().items():
        #print(val.cpu().numpy())
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
    #p = libtorch_to_parameter(model)
    #return [bytes_to_ndarray(tensor) for tensor in p.tensors]

def weights_to_parameters(
    weights: Weights, tensor_type: str, model: torch.nn.Module
) -> Parameters:
    if tensor_type == "libtorch":
        model = weights_to_libtorch(weights, model)
        #print(model)
        return libtorch_to_parameter(model)


def weights_to_libtorch(weights: Weights, model: torch.nn.Module) -> torch.nn.Module:
    #print(type(weights))
    #print(model)
    #print("hello")
    #tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    #parameters = Parameters(tensors=tensors, tensor_type="numpy.ndarray")
    params_dict = zip(model.state_dict().keys(), weights)
    #print(list(params_dict))
    
    state_dict = OrderedDict({k: torch.Tensor(np.array(v)) for k, v in params_dict})
    #print(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model


def libtorch_to_parameter(model: torch.nn.Module) -> Parameters:
    bytes_io = BytesIO()
    model.save(bytes_io)
    return Parameters(tensors=bytes_io.getvalue(), tensor_type="libtorch")
