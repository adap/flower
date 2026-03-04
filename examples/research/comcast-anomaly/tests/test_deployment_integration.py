from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


BASE = Path(__file__).resolve().parents[1]
RUN_INTEGRATION = os.environ.get("RUN_COMCAST_DEPLOYMENT_INTEGRATION") == "1"
HAS_BINARIES = shutil.which("flower-superlink") is not None and shutil.which("flower-supernode") is not None


@pytest.mark.skipif(not RUN_INTEGRATION or not HAS_BINARIES, reason="Set RUN_COMCAST_DEPLOYMENT_INTEGRATION=1 with Flower binaries installed")
def test_managed_local_deployment_single_domain_smoke(tmp_path: Path) -> None:
    cfg = {
        "schema_version": "1.0",
        "mode": "deployment",
        "domains": ["downstream_rxmer"],
        "seed": 7,
        "federation": {
            "num_clients": 2,
            "num_rounds": 1,
            "fraction_train": 1.0,
            "fraction_evaluate": 1.0,
            "min_train_nodes": 2,
            "min_evaluate_nodes": 2,
        },
        "local_training": {
            "local_epochs": 1,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        "data": {
            "samples_per_client_train": 120,
            "samples_per_client_val": 40,
            "samples_per_client_test": 40,
        },
        "non_iid": {"global": 0.3},
        "unknown_gate": {"enabled": True, "threshold_grid_size": 51, "unknown_class_index": 6},
        "artifacts": {"root_dir": str(tmp_path / "artifacts"), "run_name": "deploy_single_domain"},
        "deployment": {
            "launch_mode": "managed_local",
            "connection_name": "comcast-local-it",
            "run_timeout_sec": 300,
            "poll_interval_sec": 2.0,
            "startup_timeout_sec": 20,
            "shutdown_grace_sec": 5,
            "local_num_supernodes": 2,
            "local_insecure": True,
            "local_database": ":flwr-in-memory:",
            "local_runtime_dir": str(tmp_path / "artifacts" / "deploy_single_domain" / "deployment_runtime"),
            "superlink": None,
            "federation": None,
            "stream_logs": False,
        },
    }
    cfg_path = tmp_path / "deploy_local_single.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(BASE / "scripts" / "run_comcast_fl.py"), "--config", str(cfg_path), "--mode", "deployment"],
        cwd=str(BASE),
        text=True,
        capture_output=True,
        check=False,
        timeout=420,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    run_root = Path(cfg["artifacts"]["root_dir"]) / cfg["artifacts"]["run_name"]
    assert (run_root / "downstream_rxmer" / "metrics.json").exists()
    assert (run_root / "downstream_rxmer" / "checkpoint_best.pt").exists()
    assert (run_root / "summary.json").exists()

    runtime_state = json.loads((run_root / "deployment_runtime" / "runtime_state.json").read_text(encoding="utf-8"))
    assert runtime_state["teardown"]["all_exited"] is True


@pytest.mark.skipif(not RUN_INTEGRATION or not HAS_BINARIES, reason="Set RUN_COMCAST_DEPLOYMENT_INTEGRATION=1 with Flower binaries installed")
def test_managed_local_deployment_two_domain_e2e(tmp_path: Path) -> None:
    cfg = {
        "schema_version": "1.0",
        "mode": "deployment",
        "domains": ["downstream_rxmer", "upstream_return"],
        "seed": 8,
        "federation": {
            "num_clients": 2,
            "num_rounds": 1,
            "fraction_train": 1.0,
            "fraction_evaluate": 1.0,
            "min_train_nodes": 2,
            "min_evaluate_nodes": 2,
        },
        "local_training": {
            "local_epochs": 1,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        "data": {
            "samples_per_client_train": 120,
            "samples_per_client_val": 40,
            "samples_per_client_test": 40,
        },
        "non_iid": {"global": 0.3},
        "unknown_gate": {"enabled": True, "threshold_grid_size": 51, "unknown_class_index": 6},
        "artifacts": {"root_dir": str(tmp_path / "artifacts"), "run_name": "deploy_two_domain"},
        "deployment": {
            "launch_mode": "managed_local",
            "connection_name": "comcast-local-it-2",
            "run_timeout_sec": 300,
            "poll_interval_sec": 2.0,
            "startup_timeout_sec": 20,
            "shutdown_grace_sec": 5,
            "local_num_supernodes": 2,
            "local_insecure": True,
            "local_database": ":flwr-in-memory:",
            "local_runtime_dir": str(tmp_path / "artifacts" / "deploy_two_domain" / "deployment_runtime"),
            "superlink": None,
            "federation": None,
            "stream_logs": False,
        },
    }
    cfg_path = tmp_path / "deploy_local_two.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(BASE / "scripts" / "run_comcast_fl.py"), "--config", str(cfg_path), "--mode", "deployment"],
        cwd=str(BASE),
        text=True,
        capture_output=True,
        check=False,
        timeout=540,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    run_root = Path(cfg["artifacts"]["root_dir"]) / cfg["artifacts"]["run_name"]
    assert (run_root / "downstream_rxmer" / "metrics.json").exists()
    assert (run_root / "upstream_return" / "metrics.json").exists()
    assert (run_root / "summary.json").exists()
    assert (run_root / "comparison.csv").exists()

    runtime_state = json.loads((run_root / "deployment_runtime" / "runtime_state.json").read_text(encoding="utf-8"))
    assert runtime_state["teardown"]["all_exited"] is True
