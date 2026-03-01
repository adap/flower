from __future__ import annotations

from pathlib import Path

import numpy as np

from comcast_fl.config import load_experiment_config, resolve_non_iid
from comcast_fl.federated_core import build_client_bundle
from comcast_fl.synthetic import make_client_profile


BASE = Path(__file__).resolve().parents[1]


def test_yaml_json_load_equivalence() -> None:
    cfg_yaml = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg_json = load_experiment_config(str(BASE / "configs" / "smoke.json"))
    assert cfg_yaml.mode == "simulation"
    assert cfg_json.mode == "simulation"
    assert cfg_yaml.federation.num_clients == cfg_json.federation.num_clients == 4


def test_non_iid_axis_override_precedence() -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg.non_iid.global_ = 0.2
    cfg.non_iid.class_skew = 0.8
    sev = resolve_non_iid(cfg)
    assert sev["class_skew"] == 0.8
    assert sev["regime_skew"] == 0.2


def test_client_bundle_deterministic() -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    b1 = build_client_bundle("0", "downstream_rxmer", cfg)
    b2 = build_client_bundle("0", "downstream_rxmer", cfg)
    assert np.allclose(b1["raw"]["train"]["X_seq"], b2["raw"]["train"]["X_seq"])
    assert np.array_equal(b1["raw"]["train"]["y"], b2["raw"]["train"]["y"])


def test_non_iid_severity_increases_profile_dispersion() -> None:
    cfg_low = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg_high = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg_low.non_iid.global_ = 0.1
    cfg_high.non_iid.global_ = 0.9

    def profile_dispersion(cfg) -> float:
        rows = []
        for cid in range(12):
            p = make_client_profile(str(cid), "upstream_return", cfg)
            rows.append(p.class_probs_by_regime.reshape(-1))
        arr = np.stack(rows, axis=0)
        return float(arr.std(axis=0).mean())

    assert profile_dispersion(cfg_high) > profile_dispersion(cfg_low)
