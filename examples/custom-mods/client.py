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
        server_round = int(msg.metadata.group_id)

        run_id = msg.metadata.run_id
        group_name = f"Run ID: {run_id}"

        node_id = str(msg.metadata.dst_node_id)
        run_name = f"Node ID: {node_id}"

        wandb.init(
            project=name,
            group=group_name,
            name=run_name,
            id=f"{run_id}{node_id}",
            resume="allow",
            reinit=True,
        )

        start_time = time.time()

        reply = app(msg, context)

        time_diff = time.time() - start_time

        if reply.metadata.message_type == MESSAGE_TYPE_FIT and reply.has_content():

            metrics = reply.content.configs_records

            results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))

            results_to_log["fit_time"] = time_diff

            wandb.log(results_to_log, step=int(server_round), commit=True)

        return reply

    return wandb_mod


def get_tensorboard_mod(logdir) -> Mod:
    os.makedirs(logdir, exist_ok=True)

    def tensorboard_mod(
        msg: Message, context: Context, app: ClientAppCallable
    ) -> Message:
        logdir_run = os.path.join(logdir, str(msg.metadata.run_id))

        node_id = str(msg.metadata.dst_node_id)

        server_round = int(msg.metadata.group_id)

        start_time = time.time()

        reply = app(msg, context)

        time_diff = time.time() - start_time

        if reply.metadata.message_type == MESSAGE_TYPE_FIT and reply.has_content():
            writer = tf.summary.create_file_writer(os.path.join(logdir_run, node_id))

            metrics = dict(
                reply.content.configs_records.get("fitres.metrics", ConfigsRecord())
            )

            with writer.as_default(step=server_round):
                tf.summary.scalar(f"fit_time", time_diff, step=server_round)
                for metric in metrics:
                    tf.summary.scalar(
                        f"{metric}",
                        metrics[metric],
                        step=server_round,
                    )
                writer.flush()

        return reply

    return tensorboard_mod


# Run via `flower-client-app client:wandb_app`
wandb_app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[
        get_wandb_mod("Custom mods example"),
    ],
)

# Run via `flower-client-app client:tb_app`
tb_app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[
        get_tensorboard_mod(".runs_history/"),
    ],
)
