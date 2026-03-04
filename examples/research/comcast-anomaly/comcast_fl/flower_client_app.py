"""Flower ClientApp entrypoint for Comcast anomaly FL."""

from __future__ import annotations

from flwr.clientapp import ClientApp

from .app_state import get_active_experiment_from_context
from .flower_logic import client_evaluate_handler, client_train_handler

app = ClientApp()


@app.train()
def train(msg, context):
    msg_cfg = msg.content.get("config", {})
    cfg, domain = get_active_experiment_from_context(context.run_config, msg_cfg)
    return client_train_handler(msg=msg, context=context, cfg=cfg, domain=domain)


@app.evaluate()
def evaluate(msg, context):
    msg_cfg = msg.content.get("config", {})
    cfg, domain = get_active_experiment_from_context(context.run_config, msg_cfg)
    return client_evaluate_handler(msg=msg, context=context, cfg=cfg, domain=domain)
