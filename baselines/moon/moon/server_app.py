"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.server.strategy import FedAvg
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from moon.models import init_net, test
from moon.dataset_preparation import get_dataset, get_data_transforms, get_transforms_apply_fn


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    dataset_name: str, 
    model_name: str,
    model_output_dim: int,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        net = init_net(dataset_name, model_name, model_output_dim)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)
        accuracy, loss = test(net, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:

    dataset_name = context.run_config['dataset-name']
    fds = get_dataset(dataset_name=dataset_name,
                      dirichlet_alpha=context.run_config['dirichlet-alpha'],
                      partition_by=context.run_config['dataset-partition-by'],
                      num_partitions=1) # TODO: a temporary fix
    global_test_set = fds.load_split("test")

    _, test_transforms = get_data_transforms(dataset_name=dataset_name)

    transforms_fn = get_transforms_apply_fn(test_transforms)
    testloader = DataLoader(global_test_set.with_transform(transforms_fn),
                            batch_size=context.run_config['batch-size'])

    evaluate_fn = gen_evaluate_fn(testloader,
                                  device=context.run_config['server-device'],
                                  dataset_name=dataset_name,
                                  model_name=context.run_config["model-name"],
                                  model_output_dim=context.run_config["model-output-dim"])

    
    strategy = FedAvg(
        # Clients in MOON do not perform federated evaluation
        # (see the client's evaluate())
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=0.0,
        evaluate_fn=evaluate_fn,
    )

    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)



app = ServerApp(server_fn=server_fn)
