from __future__ import annotations

from pathlib import Path

from comcast_fl.adapters import write_run_summary
from comcast_fl.config import load_experiment_config


BASE = Path(__file__).resolve().parents[1]


def test_write_run_summary(tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg.artifacts.root_dir = str(tmp_path)
    cfg.artifacts.run_name = "unit"

    fake_dm = {
        "unknown_threshold": 0.3,
        "gated_metrics": {
            "macro_f1": 0.5,
            "event_peak_macro_f1": 0.4,
            "impulse_f1": 0.2,
            "unknown_f1": 0.3,
            "anomaly_auroc": 0.7,
        },
        "edge": {
            "params": 100,
            "p95_latency_ms_cpu_proxy": 1.0,
            "quantization_ready": True,
            "pass_edge_gate": True,
        },
    }

    results = {d: {"domain_metrics": fake_dm} for d in cfg.domains}
    out_path = tmp_path / "unit" / "summary.json"
    write_run_summary(results, str(out_path), cfg)

    assert out_path.exists()
    assert (tmp_path / "unit" / "comparison.csv").exists()
