"""Flower launcher app for orchestrating FedOps client processes."""

import os
import subprocess
import sys
import time
from typing import Optional

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

app = ServerApp()


def _terminate_process(proc: subprocess.Popen, name: str, timeout_sec: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    print(f"[launcher] stopping {name} (pid={proc.pid})")
    proc.terminate()
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"[launcher] force-killing {name} (pid={proc.pid})")
        proc.kill()
        proc.wait(timeout=5.0)


def _read_str(config: Context, key: str, default: str) -> str:
    value: Optional[object] = config.run_config.get(key)
    if value is None:
        return default
    return str(value)


@app.main()
def main(grid: Grid, context: Context) -> None:
    del grid

    task_id = _read_str(context, "task_id", "task_id")

    manager_cmd = [sys.executable, "-m", "fedopsmnist.client_manager_main"]
    client_cmd = [sys.executable, "-m", "fedopsmnist.client_main"]
    client_env = os.environ.copy()
    client_env["FEDOPS_TASK_ID"] = task_id

    print(f"[launcher] task_id={task_id}")
    print(f"[launcher] manager cmd: {' '.join(manager_cmd)}")
    print(f"[launcher] client cmd: {' '.join(client_cmd)}")

    manager_proc = subprocess.Popen(manager_cmd)
    # Manager API needs to be up before client starts polling.
    time.sleep(2)
    client_proc = subprocess.Popen(client_cmd, env=client_env)

    try:
        while True:
            manager_code = manager_proc.poll()
            client_code = client_proc.poll()

            if manager_code is not None:
                raise RuntimeError(f"client_manager_main exited with code {manager_code}")
            if client_code is not None:
                raise RuntimeError(f"client_main exited with code {client_code}")

            time.sleep(1)
    except KeyboardInterrupt:
        print("[launcher] interrupted by user")
    except RuntimeError as err:
        print(f"[launcher] {err}")
        raise
    finally:
        _terminate_process(client_proc, "client_main")
        _terminate_process(manager_proc, "client_manager_main")
