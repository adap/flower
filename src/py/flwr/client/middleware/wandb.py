import time
from collections.abc import Callable
from datetime import datetime

import wandb

from flwr.client.typing import Bwd, Fwd
from flwr.common.serde import client_message_from_proto, server_message_from_proto

from .typing import App


def test_middleware(fwd: Fwd, app: App) -> Bwd:
    print("before")

    bwd = app(fwd)

    print("after")

    return bwd


def get_wandb_middleware(
    project_name: str, client_id: str
) -> Callable[[Fwd, App], Bwd]:
    now = datetime.now().strftime("%b%d_%H_%M_%S")
    wandb_group = f"exp_{now}"
    wandb.init(project=project_name, group=wandb_group, name=f"client-{client_id}")

    def wandb_middleware(fwd: Fwd, app: App) -> Bwd:
        start_time = None
        round = ""

        server_message = server_message_from_proto(
            fwd.task_ins.task.legacy_server_message
        )
        if server_message.fit_ins:
            config = server_message.fit_ins.config
            if "round" in config:
                round = f"_rnd-{config['round']}"
            start_time = time.time()
        if server_message.evaluate_ins:
            config = server_message.evaluate_ins.config
            if "round" in config:
                round = f"_rnd-{config['round']}"

        bwd = app(fwd)

        results_to_log = {}

        if len(round) > 0:
            results_to_log["round"] = round

        client_message = client_message_from_proto(
            bwd.task_res.task.legacy_client_message
        )

        if client_message.evaluate_res:
            results_to_log["evaluate_loss"] = client_message.evaluate_res.loss
            if "accuracy" in client_message.evaluate_res.metrics:
                results_to_log["accuracy"] = client_message.evaluate_res.metrics[
                    "accuracy"
                ]

        if client_message.fit_res:
            if start_time is not None:
                results_to_log["fit_time"] = time.time() - start_time
            if "loss" in client_message.fit_res.metrics:
                results_to_log["loss"] = client_message.fit_res.metrics["loss"]
            if "accuracy" in client_message.fit_res.metrics:
                results_to_log["accuracy"] = client_message.fit_res.metrics["accuracy"]

        wandb.log(results_to_log)

        return bwd

    return wandb_middleware
