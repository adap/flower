#!/usr/bin/env python3
"""Run Comcast anomaly FL experiments from a YAML/JSON config file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comcast_fl import (
    apply_mode_override,
    load_experiment_config,
    run_domain_deployment,
    run_domain_simulation,
    write_run_summary,
)


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


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    cfg = apply_mode_override(cfg, args.mode)
    root_dir = Path(cfg.artifacts.root_dir)
    if not root_dir.is_absolute():
        cfg.artifacts.root_dir = str((ROOT / root_dir).resolve())

    print("Effective config:")
    print(json.dumps(cfg.to_dict(), indent=2))

    results: dict[str, dict] = {}
    for domain in cfg.domains:
        print(f"\n=== Running domain={domain} mode={cfg.mode} ===")
        if cfg.mode == "simulation":
            out = run_domain_simulation(domain=domain, cfg=cfg)
        else:
            out = run_domain_deployment(domain=domain, cfg=cfg)
        results[domain] = out
        macro = out["domain_metrics"]["gated_metrics"]["macro_f1"]
        print(f"Domain {domain} completed. gated_macro_f1={macro:.4f}")

    summary_path = Path(cfg.artifacts.root_dir) / cfg.artifacts.run_name / "summary.json"
    write_run_summary(results=results, out_path=str(summary_path), cfg=cfg)

    print("\nRun finished.")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Comparison CSV: {(summary_path.parent / 'comparison.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
