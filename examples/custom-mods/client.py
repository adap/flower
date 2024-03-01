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
    def wandb_mod(fwd: Message, context: Context, app: ClientAppCallable) -> Message:
        start_time = None

        project_name = name
        run_id = fwd.metadata.run_id
        round = int(fwd.metadata.group_id)
        group_name = f"Workload ID: {run_id}"

        client_id = str(fwd.metadata.dst_node_id)
        run_name = f"Client ID: {client_id}"

        time_diff = None

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

        msg_type = bwd.metadata.message_type

        if msg_type == MESSAGE_TYPE_FIT:

            time_diff = time.time() - start_time

            metrics = bwd.content.configs_records

            results_to_log = dict(
                metrics.get(f"{msg_type}res.metrics", ConfigsRecord())
            )

            if time_diff is not None:
                results_to_log[f"{msg_type}_time"] = time_diff

            wandb.log(results_to_log, step=int(round), commit=True)

        return bwd

    return wandb_mod


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
        client_id = str(fwd.metadata.dst_node_id)

        round = int(fwd.metadata.group_id)

        start_time = time.time()

        bwd = app(fwd, context)

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
