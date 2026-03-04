#!/usr/bin/env python3
"""Run Comcast anomaly FL experiments from a YAML/JSON config file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comcast_fl import (
    apply_mode_override,
    load_experiment_config,
    run_domain_deployment,
    run_domain_simulation,
    start_managed_azure_runtime,
    start_managed_local_runtime,
    stop_managed_azure_runtime,
    stop_managed_local_runtime,
    write_run_summary,
)
from comcast_fl.ui_hooks import UiHookSink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Comcast anomaly FL experiment")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON experiment config")
    parser.add_argument(
        "--mode",
        choices=["simulation", "deployment"],
        default=None,
        help="Optional mode override",
    )
    return parser.parse_args()


def run_experiment(cfg, hook_sink: UiHookSink | None = None) -> dict[str, Any]:
    root_dir = Path(cfg.artifacts.root_dir)
    if not root_dir.is_absolute():
        cfg.artifacts.root_dir = str((ROOT / root_dir).resolve())

    print("Effective config:")
    print(json.dumps(cfg.to_dict(), indent=2))

    runtime = None
    if cfg.mode == "deployment" and cfg.deployment.launch_mode == "managed_local":
        runtime = start_managed_local_runtime(cfg, hook_sink=hook_sink)
        print(
            "Managed local runtime started:",
            f"control_api={runtime.control_api_addr}, fleet_api={runtime.fleet_api_addr}",
        )
    elif cfg.mode == "deployment" and cfg.deployment.launch_mode == "managed_azure_ssh":
        runtime = start_managed_azure_runtime(cfg, hook_sink=hook_sink)
        print(
            "Managed Azure runtime started:",
            f"control_api={runtime.control_api_addr}, fleet_api={runtime.fleet_api_addr}",
        )

    results: dict[str, dict] = {}
    try:
        for domain in cfg.domains:
            print(f"\n=== Running domain={domain} mode={cfg.mode} ===")
            if cfg.mode == "simulation":
                out = run_domain_simulation(domain=domain, cfg=cfg, hook_sink=hook_sink)
            else:
                out = run_domain_deployment(domain=domain, cfg=cfg, runtime=runtime, hook_sink=hook_sink)
            results[domain] = out
            macro = out["domain_metrics"]["gated_metrics"]["macro_f1"]
            print(f"Domain {domain} completed. gated_macro_f1={macro:.4f}")
    finally:
        if runtime is not None:
            if cfg.deployment.launch_mode == "managed_local":
                stop_managed_local_runtime(
                    runtime,
                    shutdown_grace_sec=int(cfg.deployment.shutdown_grace_sec),
                    hook_sink=hook_sink,
                    run_name=cfg.artifacts.run_name,
                )
            elif cfg.deployment.launch_mode == "managed_azure_ssh":
                stop_managed_azure_runtime(runtime)

    summary_path = Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / "summary.json"
    write_run_summary(results=results, out_path=str(summary_path), cfg=cfg)

    print("\nRun finished.")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Comparison CSV: {(summary_path.parent / 'comparison.csv').resolve()}")
    return {
        "results": results,
        "summary_path": str(summary_path.resolve()),
        "comparison_csv": str((summary_path.parent / "comparison.csv").resolve()),
    }


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    cfg = apply_mode_override(cfg, args.mode)
    _ = run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
