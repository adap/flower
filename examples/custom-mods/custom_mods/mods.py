import os
import time
import wandb
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common import ConfigsRecord
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message

from torch.utils.tensorboard import SummaryWriter


def get_wandb_mod(name: str) -> Mod:
    def wandb_mod(msg: Message, context: Context, app: ClientAppCallable) -> Message:
        """Flower Mod that logs the metrics dictionary returned by the client's fit
        function to Weights & Biases."""
        server_round = int(msg.metadata.group_id)

        if server_round == 1 and msg.metadata.message_type == MessageType.TRAIN:
            run_id = msg.metadata.run_id
            group_name = f"Run ID: {run_id}"

            node_id = str(msg.metadata.dst_node_id)
            run_name = f"Node ID: {node_id}"

            wandb.init(
                project=name,
                group=group_name,
                name=run_name,
                id=f"{run_id}_{node_id}",
                resume="allow",
                reinit=True,
            )
            # We'll use a custom metric as x-axis step
            # This is needed if not all clients participate in all rounds
            # W&B doesn't allow logging at step 1,2,4 (i.e. skipping 3)
            wandb.define_metric("server-round")

        start_time = time.time()

        reply = app(msg, context)

        time_diff = time.time() - start_time

        # if the `ClientApp` just processed a "fit" message, let's log some metrics to W&B
        if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():
            metrics = reply.content.configs_records

            results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))
            results_to_log["fit_time"] = time_diff

            # Ensure all metrics to log use the custom step
            wandb.define_metric("*", step_metric="server-round")
            results_to_log["server-round"] = server_round
            # Log as usual
            wandb.log(results_to_log, commit=True)

        return reply

    return wandb_mod


def get_tensorboard_mod(logdir) -> Mod:
    os.makedirs(logdir, exist_ok=True)

    def tensorboard_mod(
        msg: Message, context: Context, app: ClientAppCallable
    ) -> Message:
        """Flower Mod that logs the metrics dictionary returned by the client's fit
        function to TensorBoard."""
        logdir_run = os.path.join(logdir, str(msg.metadata.run_id))

        node_id = str(msg.metadata.dst_node_id)

        server_round = int(msg.metadata.group_id)

        start_time = time.time()

        reply = app(msg, context)

        time_diff = time.time() - start_time

        # if the `ClientApp` just processed a "fit" message, let's log some metrics to TensorBoard
        if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():
            writer = SummaryWriter(os.path.join(logdir_run, node_id))

            metrics = dict(
                reply.content.configs_records.get("fitres.metrics", ConfigsRecord())
            )

            writer.add_scalar(f"fit_time", time_diff, global_step=server_round)
            for metric in metrics:
                writer.add_scalar(
                    f"{metric}",
                    metrics[metric],
                    global_step=server_round,
                )
            writer.flush()

        return reply

    return tensorboard_mod
