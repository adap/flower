import os
import datetime
import time
from typing import Dict

import flwr as fl
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common.context import Context
from flwr.common.message import Message
import tensorflow as tf
from flwr.common import NDArrays, Scalar
from flwr.common.constant import TASK_TYPE_FIT, TASK_TYPE_EVALUATE

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

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        return get_parameters(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    return FlowerClient().to_client()


def get_tensorboard_mod(logdir) -> Mod:
    os.makedirs(logdir, exist_ok=True)

    # To allow multiple runs and group those we will create a subdir
    # in the logdir which is named as number of directories in logdir + 1
    run_id = str(
        len(
            [
                name
                for name in os.listdir(logdir)
                if os.path.isdir(os.path.join(logdir, name))
            ]
        )
    )
    run_id = run_id + "-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    logdir_run = os.path.join(logdir, run_id)

    def tensorboard_mod(
        fwd: Message, context: Context, app: ClientAppCallable
    ) -> Message:
        group_id = fwd.metadata.group_id

        client_id = str(fwd.message)

        config = fwd.message.configs
        if "round" in config:
            round = str(config["round"])
        else:
            round = group_id

        start_time = time.time()

        bwd = app(fwd, context)

        time_diff = time.time() - start_time

        if bwd.metadata.task_type == (TASK_TYPE_FIT or TASK_TYPE_EVALUATE):
            writer = tf.summary.create_file_writer(os.path.join(logdir_run, client_id))

            metrics = bwd.message.metrics
            task_type = bwd.metadata.task_type

            # Write aggregated loss
            with writer.as_default(step=round):  # pylint: disable=not-context-manager
                tf.summary.scalar(f"{task_type}_time", time_diff, step=round)
                if "accuracy" in metrics:
                    tf.summary.scalar(
                        f"{task_type}_accuracy",
                        metrics["accuracy"],
                        step=round,
                    )
                if "loss" in metrics:
                    tf.summary.scalar(
                        f"{task_type}_loss",
                        metrics["loss"],
                        step=round,
                    )
                writer.flush()

        return bwd

    return tensorboard_mod


# To run this: `flower-client client:app`
app = fl.client.ClientApp(
    client_fn=client_fn, mods=[get_tensorboard_mod(".runs_history")]
)


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:9092",  # "0.0.0.0:9093" for REST
        client_fn=client_fn,
        transport="grpc-rere",  # "rest" for REST
    )
