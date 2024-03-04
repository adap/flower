import datetime
import os
import time

import flwr as fl
import tensorflow as tf
import wandb
from flwr.common import ConfigsRecord
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common.context import Context
from flwr.common.message import Message
from flwr.common.constant import MESSAGE_TYPE_FIT

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
    def get_parameters(self, config):
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


def get_wandb_mod(name: str) -> Mod:
    def wandb_mod(msg: Message, context: Context, app: ClientAppCallable) -> Message:
        round = int(msg.metadata.group_id)

        run_id = msg.metadata.run_id
        group_name = f"Workload ID: {run_id}"

        client_id = str(msg.metadata.dst_node_id)
        run_name = f"Client ID: {client_id}"

        wandb.init(
            project=name,
            group=group_name,
            name=run_name,
            id=f"{run_id}{client_id}",
            resume="allow",
            reinit=True,
        )

        start_time = time.time()

        bwd = app(msg, context)

        msg_type = bwd.metadata.message_type

        if msg_type == MESSAGE_TYPE_FIT:

            time_diff = time.time() - start_time

            metrics = bwd.content.configs_records

            results_to_log = dict(
                metrics.get(f"{msg_type}res.metrics", ConfigsRecord())
            )

            results_to_log[f"{msg_type}_time"] = time_diff

            wandb.log(results_to_log, step=int(round), commit=True)

        return bwd

    return wandb_mod


def get_tensorboard_mod(logdir) -> Mod:
    os.makedirs(logdir, exist_ok=True)

    def tensorboard_mod(
        msg: Message, context: Context, app: ClientAppCallable
    ) -> Message:
        logdir_run = os.path.join(logdir, msg.metadata.run_id)

        client_id = str(msg.metadata.dst_node_id)

        round = int(msg.metadata.group_id)

        start_time = time.time()

        bwd = app(msg, context)

        time_diff = time.time() - start_time

        if bwd.metadata.message_type == MESSAGE_TYPE_FIT:
            writer = tf.summary.create_file_writer(os.path.join(logdir_run, client_id))

            metrics = dict(
                bwd.content.configs_records.get("fitres.metrics", ConfigsRecord())
            )
            # Write aggregated loss
            with writer.as_default(step=round):  # pylint: disable=not-context-manager
                tf.summary.scalar(f"fit_time", time_diff, step=round)
                for metric in metrics:
                    tf.summary.scalar(
                        f"{metric}",
                        metrics[metric],
                        step=round,
                    )
                writer.flush()

        return bwd

    return tensorboard_mod


# Run via `flower-client-app client:wandb_app`
wandb_app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[get_wandb_mod("Custom mods example")],
)

# Run via `flower-client-app client:tb_app`
tb_app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[get_tensorboard_mod(".runs_history/")],
)
