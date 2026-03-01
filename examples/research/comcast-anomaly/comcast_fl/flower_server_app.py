"""Flower ServerApp entrypoint for Comcast anomaly FL."""

from __future__ import annotations

from flwr.serverapp import ServerApp

from .app_state import get_active_experiment_from_context
from .flower_logic import server_main

app = ServerApp()


@app.main()
def main(grid, context) -> None:
    cfg, domain = get_active_experiment_from_context(context.run_config)
    server_main(grid=grid, context=context, cfg=cfg, domain=domain)
