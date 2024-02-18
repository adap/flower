from typing import Dict, Tuple
import time

import flwr as fl
import wandb
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common.message import Message
from flwr.common.constant import MESSAGE_TYPE_FIT, MESSAGE_TYPE_EVALUATE
from flwr.common.context import Context
from flwr.common import NDArrays, Parameters, Scalar

from task import (
    Net,
    DEVICE,
    load_data,
    get_parameters,
    set_parameters,
    train,
    test,
)


# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(net)

    def fit(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        return get_parameters(net), len(trainloader.dataset), results

    def evaluate(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str) -> fl.client.Client:
    return FlowerClient().to_client()


def get_wandb_mod(name: str) -> Mod:
    def wandb_mod(fwd: Message, context: Context, app: ClientAppCallable) -> Message:
        start_time = None

        project_name = name
        run_id = fwd.metadata.run_id
        group_id = fwd.metadata.group_id
        group_name = f"Workload ID: {run_id}"

        client_id = str(fwd.metadata.node_id)
        run_name = f"Client ID: {client_id}"

        time_diff = None

        config = fwd.content.configs
        if "round" in config:
            round = config["round"]
        else:
            round = group_id
        if "project" in config:
            project_name = str(config["project"])
        if "group" in config:
            group_name = str(config["group"])

        wandb.init(
            project=project_name,
            group=group_name,
            name=run_name,
            id=f"{run_id}{client_id}",
            resume="allow",
            reinit=True,
        )

        start_time = time.time()

        bwd = app(fwd, context)

        if bwd.metadata.message_type == (MESSAGE_TYPE_FIT or MESSAGE_TYPE_EVALUATE):
            time_diff = time.time() - start_time

            results_to_log = {}

            metrics = bwd.content.metrics
            msg_type = bwd.metadata.message_type

            if "loss" in metrics:
                results_to_log[f"{msg_type}_loss"] = metrics["loss"]
            if "accuracy" in metrics:
                results_to_log[f"{msg_type}_accuracy"] = metrics["accuracy"]
            if time_diff is not None:
                results_to_log[f"{msg_type}_time"] = time_diff

            wandb.log(results_to_log, step=int(round))

        return bwd

    return wandb_mod


# Run via `flower-client-app client:app`
app = fl.client.ClientApp(
    client_fn=client_fn, mods=[get_wandb_mod("MT PyTorch Callable")]
)
