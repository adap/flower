from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from comcast_fl.config import (
    AzureSshConfig,
    AzureVmSpec,
    SupernodeAuthConfig,
    TlsConfig,
    apply_mode_override,
    load_experiment_config,
    resolve_non_iid,
)
from comcast_fl.federated_core import build_client_bundle
from comcast_fl.synthetic import make_client_profile


BASE = Path(__file__).resolve().parents[1]


def test_yaml_json_load_equivalence() -> None:
    cfg_yaml = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg_json = load_experiment_config(str(BASE / "configs" / "smoke.json"))
    assert cfg_yaml.mode == "simulation"
    assert cfg_json.mode == "simulation"
    assert cfg_yaml.federation.num_clients == cfg_json.federation.num_clients


def test_non_iid_axis_override_precedence() -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg.non_iid.global_ = 0.2
    cfg.non_iid.class_skew = 0.8
    sev = resolve_non_iid(cfg)
    assert sev["class_skew"] == 0.8
    assert sev["regime_skew"] == 0.2


def test_external_deployment_requires_superlink() -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg.mode = "deployment"
    cfg.deployment.launch_mode = "external"
    cfg.deployment.superlink = None
    with pytest.raises(ValueError, match="deployment.superlink"):
        apply_mode_override(cfg, "deployment")


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


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return str(path.resolve())


def _set_valid_azure_deployment(cfg, tmp_path: Path) -> None:
    ssh_key = _touch(tmp_path / "ssh" / "id_ed25519")
    ca = _touch(tmp_path / "certs" / "ca.crt")
    cert = _touch(tmp_path / "certs" / "server.pem")
    key = _touch(tmp_path / "certs" / "server.key")
    priv = [_touch(tmp_path / "keys" / f"sn{i}_private.pem") for i in range(2)]
    pub = [_touch(tmp_path / "keys" / f"sn{i}_public.pem") for i in range(2)]

    cfg.mode = "deployment"
    cfg.federation.num_clients = 2
    cfg.deployment.launch_mode = "managed_azure_ssh"
    cfg.deployment.azure_ssh = AzureSshConfig(
        vms=[
            AzureVmSpec(
                name="vm-a",
                host="10.0.0.4",
                ssh_port=22,
                ssh_user="azureuser",
                ssh_key_path=ssh_key,
                supernodes_on_vm=1,
                roles=["superlink", "control", "worker"],
            ),
            AzureVmSpec(
                name="vm-b",
                host="10.0.0.5",
                ssh_port=22,
                ssh_user="azureuser",
                ssh_key_path=ssh_key,
                supernodes_on_vm=1,
                roles=["worker"],
            ),
        ],
        total_supernodes=2,
        control_vm="vm-a",
        superlink_vm="vm-a",
    )
    cfg.deployment.tls = TlsConfig(
        ca_cert_local_path=ca,
        server_cert_local_path=cert,
        server_key_local_path=key,
    )
    cfg.deployment.supernode_auth = SupernodeAuthConfig(
        enabled=True,
        private_key_local_paths=priv,
        public_key_local_paths=pub,
    )


def test_managed_azure_requires_tls(tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    _set_valid_azure_deployment(cfg, tmp_path=tmp_path)
    cfg.deployment.tls = None
    with pytest.raises(ValueError, match="deployment.tls"):
        apply_mode_override(cfg, "deployment")


def test_managed_azure_validates_supernode_sum(tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    _set_valid_azure_deployment(cfg, tmp_path=tmp_path)
    assert cfg.deployment.azure_ssh is not None
    cfg.deployment.azure_ssh.total_supernodes = 3
    with pytest.raises(ValueError, match="sum\\(azure_ssh.vms\\[\\*\\].supernodes_on_vm\\)"):
        apply_mode_override(cfg, "deployment")


def test_managed_azure_valid_config_passes(tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    _set_valid_azure_deployment(cfg, tmp_path=tmp_path)
    out = apply_mode_override(cfg, "deployment")
    assert out.deployment.launch_mode == "managed_azure_ssh"


def test_managed_azure_rejects_public_ip_by_default(tmp_path: Path) -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    _set_valid_azure_deployment(cfg, tmp_path=tmp_path)
    assert cfg.deployment.azure_ssh is not None
    cfg.deployment.azure_ssh.vms[0].host = "52.10.10.10"
    with pytest.raises(ValueError, match="Public host IP not allowed"):
        apply_mode_override(cfg, "deployment")


def test_run_name_validation_rejects_path_segments() -> None:
    cfg = load_experiment_config(str(BASE / "configs" / "smoke.yaml"))
    cfg.artifacts.run_name = "../bad"
    with pytest.raises(ValueError, match="artifacts.run_name"):
        apply_mode_override(cfg, cfg.mode)
