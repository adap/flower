#!/usr/bin/env python3
"""Run Comcast FL live UI server and optionally launch an FL run."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comcast_ui.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Comcast FL live UI")
    parser.add_argument("--run-name", required=True, help="Run name under artifacts root")
    parser.add_argument("--run-root", default="artifacts/fl", help="Artifacts root directory")
    parser.add_argument("--domains", default="downstream_rxmer,upstream_return", help="Comma-separated domain list")
    parser.add_argument("--host", default="127.0.0.1", help="UI host")
    parser.add_argument("--port", type=int, default=8050, help="UI port")
    parser.add_argument("--poll-sec", type=float, default=1.0, help="File polling interval")
    parser.add_argument("--supernode-poll-sec", type=float, default=2.0, help="Supernode heartbeat polling interval")
    parser.add_argument("--superlink-connection", default=None, help="Optional explicit SuperLink connection name")
    parser.add_argument("--flwr-home", default=None, help="Optional explicit FLWR_HOME for connection resolution")
    parser.add_argument("--launch-fl", action="store_true", help="Launch FL run as a child process")
    parser.add_argument("--fl-config", default=None, help="Config file for child FL run")
    parser.add_argument("--fl-mode", choices=["simulation", "deployment"], default=None, help="Optional FL mode override")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (ROOT / run_root).resolve()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    app = create_app(
        run_root=run_root,
        run_name=args.run_name,
        domains=domains,
        poll_interval_sec=args.poll_sec,
        supernode_poll_interval_sec=args.supernode_poll_sec,
        superlink_connection=args.superlink_connection,
        flwr_home=args.flwr_home,
    )

    child: subprocess.Popen[str] | None = None
    try:
        if args.launch_fl:
            if not args.fl_config:
                raise ValueError("--fl-config is required when --launch-fl is set")
            cmd = [sys.executable, str(ROOT / "scripts" / "run_comcast_fl.py"), "--config", args.fl_config]
            if args.fl_mode:
                cmd.extend(["--mode", args.fl_mode])
            child = subprocess.Popen(cmd, cwd=str(ROOT), text=True)
            print(f"Started FL child process pid={child.pid}")

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    raise SystemExit(main())
