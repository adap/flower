"""Config schema and validation for Comcast FL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ipaddress
import json
import math
from pathlib import Path
import re
from typing import Any

from .constants import (
    DEFAULT_CLASS_PROBS_BY_DOMAIN,
    DEFAULT_REGIME_MIX,
    V2_SIGNAL_DOMAINS,
)


@dataclass(slots=True)
class FederationConfig:
    num_clients: int = 10
    num_rounds: int = 3
    fraction_train: float = 1.0
    fraction_evaluate: float = 1.0
    min_train_nodes: int = 2
    min_evaluate_nodes: int = 2


@dataclass(slots=True)
class LocalTrainingConfig:
    local_epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(slots=True)
class DataConfig:
    samples_per_client_train: int = 1200
    samples_per_client_val: int = 300
    samples_per_client_test: int = 300
    regime_mix: list[float] = field(default_factory=lambda: list(DEFAULT_REGIME_MIX))
    class_priors_by_domain: dict[str, list[list[float]]] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_PROBS_BY_DOMAIN)
    )


@dataclass(slots=True)
class NonIIDConfig:
    global_: float = 0.5
    class_skew: float | None = None
    regime_skew: float | None = None
    context_skew: float | None = None
    template_skew: float | None = None


@dataclass(slots=True)
class UnknownGateConfig:
    enabled: bool = True
    threshold_grid_size: int = 201
    unknown_class_index: int = 6


@dataclass(slots=True)
class ArtifactsConfig:
    root_dir: str = "artifacts/fl"
    run_name: str = "default_run"


@dataclass(slots=True)
class AzureVmSpec:
    name: str
    host: str
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_key_path: str = ""
    supernodes_on_vm: int = 0
    roles: list[str] = field(default_factory=lambda: ["worker"])


@dataclass(slots=True)
class AzureSshConfig:
    vms: list[AzureVmSpec] = field(default_factory=list)
    total_supernodes: int = 0
    control_vm: str = ""
    superlink_vm: str = ""
    superlink_bind_host: str | None = None
    allow_public_ips: bool = False
    remote_workspace_dir: str = "/opt/comcast-anomaly"
    remote_python: str = "python3"
    remote_venv_dir: str | None = None
    ssh_connect_timeout_sec: int = 10
    startup_timeout_sec: int = 90
    poll_interval_sec: float = 2.0
    domain_run_timeout_sec: int = 3600
    teardown_grace_sec: int = 10
    sync_mode: str = "scp_tar"


@dataclass(slots=True)
class TlsConfig:
    ca_cert_local_path: str
    server_cert_local_path: str
    server_key_local_path: str
    remote_cert_dir: str = "secrets/certificates"


@dataclass(slots=True)
class SupernodeAuthConfig:
    enabled: bool = True
    private_key_local_paths: list[str] = field(default_factory=list)
    public_key_local_paths: list[str] = field(default_factory=list)
    remote_key_dir: str = "secrets/keys"


@dataclass(slots=True)
class DeploymentConfig:
    launch_mode: str = "managed_local"
    connection_name: str = "comcast-local"
    run_timeout_sec: int = 900
    poll_interval_sec: float = 2.0
    startup_timeout_sec: int = 20
    shutdown_grace_sec: int = 5
    local_num_supernodes: int | None = None
    local_insecure: bool = True
    local_database: str = ":flwr-in-memory:"
    local_runtime_dir: str | None = None
    superlink: str | None = None
    federation: str | None = None
    stream_logs: bool = True
    azure_ssh: AzureSshConfig | None = None
    tls: TlsConfig | None = None
    supernode_auth: SupernodeAuthConfig | None = None


@dataclass(slots=True)
class ExperimentConfig:
    schema_version: str = "1.0"
    mode: str = "simulation"
    domains: list[str] = field(default_factory=lambda: list(V2_SIGNAL_DOMAINS))
    seed: int = 42
    federation: FederationConfig = field(default_factory=FederationConfig)
    local_training: LocalTrainingConfig = field(default_factory=LocalTrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    non_iid: NonIIDConfig = field(default_factory=NonIIDConfig)
    unknown_gate: UnknownGateConfig = field(default_factory=UnknownGateConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def to_dict(self) -> dict[str, Any]:
        obj = asdict(self)
        obj["non_iid"]["global"] = obj["non_iid"].pop("global_")
        return obj


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _normalize(v: list[float]) -> list[float]:
    s = float(sum(v))
    _require(s > 0.0, "Probability vector sum must be > 0")
    return [float(x) / s for x in v]


def _read_bool(raw: Any, key: str) -> bool:
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"{key} must be a boolean")


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required for YAML configs. Install with `pip install pyyaml`."
            ) from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("YAML config must deserialize to a mapping")
        return data
    raise ValueError(f"Unsupported config extension: {suffix}")


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    federation = FederationConfig(**raw.get("federation", {}))
    local_training = LocalTrainingConfig(**raw.get("local_training", {}))

    data_raw = raw.get("data", {})
    regime_mix = data_raw.get("regime_mix", DEFAULT_REGIME_MIX)
    class_priors = data_raw.get("class_priors_by_domain", DEFAULT_CLASS_PROBS_BY_DOMAIN)
    data = DataConfig(
        samples_per_client_train=int(data_raw.get("samples_per_client_train", 1200)),
        samples_per_client_val=int(data_raw.get("samples_per_client_val", 300)),
        samples_per_client_test=int(data_raw.get("samples_per_client_test", 300)),
        regime_mix=[float(x) for x in regime_mix],
        class_priors_by_domain=class_priors,
    )

    non_iid_raw = raw.get("non_iid", {})
    non_iid = NonIIDConfig(
        global_=float(non_iid_raw.get("global", non_iid_raw.get("global_", 0.5))),
        class_skew=non_iid_raw.get("class_skew"),
        regime_skew=non_iid_raw.get("regime_skew"),
        context_skew=non_iid_raw.get("context_skew"),
        template_skew=non_iid_raw.get("template_skew"),
    )

    unknown_gate = UnknownGateConfig(**raw.get("unknown_gate", {}))
    artifacts = ArtifactsConfig(**raw.get("artifacts", {}))
    dep_raw = raw.get("deployment", {})
    azure_ssh_raw = dep_raw.get("azure_ssh")
    tls_raw = dep_raw.get("tls")
    supernode_auth_raw = dep_raw.get("supernode_auth")

    azure_ssh_cfg: AzureSshConfig | None = None
    if azure_ssh_raw is not None:
        allow_public_ips_raw = azure_ssh_raw.get("allow_public_ips", False)
        vms = [AzureVmSpec(**vm) for vm in azure_ssh_raw.get("vms", [])]
        azure_ssh_cfg = AzureSshConfig(
            vms=vms,
            total_supernodes=int(azure_ssh_raw.get("total_supernodes", 0)),
            control_vm=str(azure_ssh_raw.get("control_vm", "")),
            superlink_vm=str(azure_ssh_raw.get("superlink_vm", "")),
            superlink_bind_host=(
                str(azure_ssh_raw["superlink_bind_host"])
                if azure_ssh_raw.get("superlink_bind_host") is not None
                else None
            ),
            allow_public_ips=_read_bool(
                allow_public_ips_raw,
                "deployment.azure_ssh.allow_public_ips",
            ),
            remote_workspace_dir=str(azure_ssh_raw.get("remote_workspace_dir", "/opt/comcast-anomaly")),
            remote_python=str(azure_ssh_raw.get("remote_python", "python3")),
            remote_venv_dir=(
                str(azure_ssh_raw["remote_venv_dir"])
                if azure_ssh_raw.get("remote_venv_dir") is not None
                else None
            ),
            ssh_connect_timeout_sec=int(azure_ssh_raw.get("ssh_connect_timeout_sec", 10)),
            startup_timeout_sec=int(azure_ssh_raw.get("startup_timeout_sec", 90)),
            poll_interval_sec=float(azure_ssh_raw.get("poll_interval_sec", 2.0)),
            domain_run_timeout_sec=int(azure_ssh_raw.get("domain_run_timeout_sec", 3600)),
            teardown_grace_sec=int(azure_ssh_raw.get("teardown_grace_sec", 10)),
            sync_mode=str(azure_ssh_raw.get("sync_mode", "scp_tar")),
        )

    tls_cfg: TlsConfig | None = None
    if tls_raw is not None:
        tls_cfg = TlsConfig(
            ca_cert_local_path=str(tls_raw["ca_cert_local_path"]),
            server_cert_local_path=str(tls_raw["server_cert_local_path"]),
            server_key_local_path=str(tls_raw["server_key_local_path"]),
            remote_cert_dir=str(tls_raw.get("remote_cert_dir", "secrets/certificates")),
        )

    supernode_auth_cfg: SupernodeAuthConfig | None = None
    if supernode_auth_raw is not None:
        supernode_auth_cfg = SupernodeAuthConfig(
            enabled=bool(supernode_auth_raw.get("enabled", True)),
            private_key_local_paths=[str(p) for p in supernode_auth_raw.get("private_key_local_paths", [])],
            public_key_local_paths=[str(p) for p in supernode_auth_raw.get("public_key_local_paths", [])],
            remote_key_dir=str(supernode_auth_raw.get("remote_key_dir", "secrets/keys")),
        )

    deployment = DeploymentConfig(
        launch_mode=str(dep_raw.get("launch_mode", "managed_local")),
        connection_name=str(dep_raw.get("connection_name", "comcast-local")),
        run_timeout_sec=int(dep_raw.get("run_timeout_sec", 900)),
        poll_interval_sec=float(dep_raw.get("poll_interval_sec", 2.0)),
        startup_timeout_sec=int(dep_raw.get("startup_timeout_sec", 20)),
        shutdown_grace_sec=int(dep_raw.get("shutdown_grace_sec", 5)),
        local_num_supernodes=(
            int(dep_raw["local_num_supernodes"])
            if dep_raw.get("local_num_supernodes") is not None
            else None
        ),
        local_insecure=bool(dep_raw.get("local_insecure", True)),
        local_database=str(dep_raw.get("local_database", ":flwr-in-memory:")),
        local_runtime_dir=(
            str(dep_raw["local_runtime_dir"])
            if dep_raw.get("local_runtime_dir") is not None
            else None
        ),
        superlink=(str(dep_raw["superlink"]) if dep_raw.get("superlink") is not None else None),
        federation=(str(dep_raw["federation"]) if dep_raw.get("federation") is not None else None),
        stream_logs=bool(dep_raw.get("stream_logs", True)),
        azure_ssh=azure_ssh_cfg,
        tls=tls_cfg,
        supernode_auth=supernode_auth_cfg,
    )

    cfg = ExperimentConfig(
        schema_version=str(raw.get("schema_version", "1.0")),
        mode=str(raw.get("mode", "simulation")),
        domains=[str(x) for x in raw.get("domains", V2_SIGNAL_DOMAINS)],
        seed=int(raw.get("seed", 42)),
        federation=federation,
        local_training=local_training,
        data=data,
        non_iid=non_iid,
        unknown_gate=unknown_gate,
        artifacts=artifacts,
        deployment=deployment,
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    _require(cfg.mode in {"simulation", "deployment"}, "mode must be simulation or deployment")
    _require(len(cfg.domains) > 0, "domains must be non-empty")
    for d in cfg.domains:
        _require(d in V2_SIGNAL_DOMAINS, f"Unsupported domain: {d}")

    f = cfg.federation
    _require(f.num_clients > 0, "federation.num_clients must be > 0")
    _require(f.num_rounds > 0, "federation.num_rounds must be > 0")
    _require(0.0 < f.fraction_train <= 1.0, "fraction_train must be in (0,1]")
    _require(0.0 <= f.fraction_evaluate <= 1.0, "fraction_evaluate must be in [0,1]")
    _require(1 <= f.min_train_nodes <= f.num_clients, "min_train_nodes out of range")
    _require(0 <= f.min_evaluate_nodes <= f.num_clients, "min_evaluate_nodes out of range")
    _require(
        math.ceil(f.fraction_train * f.num_clients) >= f.min_train_nodes,
        "Impossible federation config: ceil(fraction_train*num_clients) < min_train_nodes",
    )
    if f.fraction_evaluate > 0.0:
        _require(
            math.ceil(f.fraction_evaluate * f.num_clients) >= f.min_evaluate_nodes,
            "Impossible federation config: ceil(fraction_evaluate*num_clients) < min_evaluate_nodes",
        )

    lt = cfg.local_training
    _require(lt.local_epochs > 0, "local_epochs must be > 0")
    _require(lt.batch_size > 0, "batch_size must be > 0")
    _require(lt.lr > 0.0, "lr must be > 0")
    _require(lt.weight_decay >= 0.0, "weight_decay must be >= 0")

    d = cfg.data
    _require(d.samples_per_client_train > 0, "samples_per_client_train must be > 0")
    _require(d.samples_per_client_val > 0, "samples_per_client_val must be > 0")
    _require(d.samples_per_client_test > 0, "samples_per_client_test must be > 0")
    _require(len(d.regime_mix) == 3, "data.regime_mix must have length 3")
    _ = _normalize([float(x) for x in d.regime_mix])

    for dom in cfg.domains:
        probs = d.class_priors_by_domain.get(dom)
        _require(probs is not None, f"Missing class_priors_by_domain for domain {dom}")
        _require(len(probs) == 3, f"class priors for {dom} must have 3 regime rows")
        for row in probs:
            _require(len(row) == 7, f"class prior row for {dom} must have 7 classes")
            _ = _normalize([float(x) for x in row])

    _require(cfg.unknown_gate.threshold_grid_size >= 2, "threshold_grid_size must be >= 2")
    _require(0 <= cfg.unknown_gate.unknown_class_index <= 6, "unknown_class_index must be 0..6")
    _require(
        re.fullmatch(r"[A-Za-z0-9._-]{1,64}", cfg.artifacts.run_name) is not None,
        "artifacts.run_name must match [A-Za-z0-9._-]{1,64}",
    )

    s = resolve_non_iid(cfg)
    for k, v in s.items():
        _require(0.0 <= v <= 1.0, f"non_iid {k} must be in [0,1]")

    dep = cfg.deployment
    _require(
        dep.launch_mode in {"managed_local", "external", "managed_azure_ssh"},
        "deployment.launch_mode must be managed_local, external, or managed_azure_ssh",
    )
    _require(dep.connection_name.strip() != "", "deployment.connection_name must be non-empty")
    _require(
        re.fullmatch(r"[A-Za-z0-9._-]{1,64}", dep.connection_name) is not None,
        "deployment.connection_name must match [A-Za-z0-9._-]{1,64}",
    )
    _require(dep.run_timeout_sec > 0, "deployment.run_timeout_sec must be > 0")
    _require(dep.poll_interval_sec > 0.0, "deployment.poll_interval_sec must be > 0")
    _require(dep.startup_timeout_sec > 0, "deployment.startup_timeout_sec must be > 0")
    _require(dep.shutdown_grace_sec > 0, "deployment.shutdown_grace_sec must be > 0")
    if dep.local_num_supernodes is not None:
        _require(dep.local_num_supernodes > 0, "deployment.local_num_supernodes must be > 0 when set")
    _require(dep.local_database.strip() != "", "deployment.local_database must be non-empty")
    if dep.launch_mode == "managed_local":
        _require(dep.local_insecure is True, "deployment.local_insecure must be true in this phase")
    if cfg.mode == "deployment" and dep.launch_mode == "external":
        _require(dep.superlink is not None and dep.superlink.strip() != "", "deployment.superlink must be set in external deployment mode")
    if dep.launch_mode == "managed_azure_ssh":
        _require(dep.azure_ssh is not None, "deployment.azure_ssh must be set in managed_azure_ssh mode")
        _require(dep.tls is not None, "deployment.tls must be set in managed_azure_ssh mode")
        _require(
            dep.supernode_auth is not None,
            "deployment.supernode_auth must be set in managed_azure_ssh mode",
        )

        azure = dep.azure_ssh
        tls = dep.tls
        auth = dep.supernode_auth
        assert azure is not None
        assert tls is not None
        assert auth is not None

        _require(azure.total_supernodes > 0, "deployment.azure_ssh.total_supernodes must be > 0")
        _require(
            isinstance(azure.allow_public_ips, bool),
            "deployment.azure_ssh.allow_public_ips must be a boolean",
        )
        _require(len(azure.vms) > 0, "deployment.azure_ssh.vms must be non-empty")
        _require(azure.ssh_connect_timeout_sec > 0, "deployment.azure_ssh.ssh_connect_timeout_sec must be > 0")
        _require(azure.startup_timeout_sec > 0, "deployment.azure_ssh.startup_timeout_sec must be > 0")
        _require(azure.poll_interval_sec > 0.0, "deployment.azure_ssh.poll_interval_sec must be > 0")
        _require(azure.domain_run_timeout_sec > 0, "deployment.azure_ssh.domain_run_timeout_sec must be > 0")
        _require(azure.teardown_grace_sec > 0, "deployment.azure_ssh.teardown_grace_sec must be > 0")
        _require(azure.sync_mode == "scp_tar", "deployment.azure_ssh.sync_mode must be scp_tar")
        _require(
            azure.remote_workspace_dir.strip() != "",
            "deployment.azure_ssh.remote_workspace_dir must be non-empty",
        )
        _require(
            azure.remote_workspace_dir.startswith("/") and azure.remote_workspace_dir != "/",
            "deployment.azure_ssh.remote_workspace_dir must be an absolute non-root path",
        )
        _require(
            ".." not in azure.remote_workspace_dir.split("/"),
            "deployment.azure_ssh.remote_workspace_dir must not contain '..'",
        )
        if azure.remote_venv_dir is not None:
            _require(
                azure.remote_venv_dir.strip() != "",
                "deployment.azure_ssh.remote_venv_dir must be non-empty when set",
            )
            _require(
                azure.remote_venv_dir.startswith("/") and azure.remote_venv_dir != "/",
                "deployment.azure_ssh.remote_venv_dir must be an absolute non-root path",
            )
            _require(
                ".." not in azure.remote_venv_dir.split("/"),
                "deployment.azure_ssh.remote_venv_dir must not contain '..'",
            )

        names = [vm.name for vm in azure.vms]
        _require(len(set(names)) == len(names), "deployment.azure_ssh.vms names must be unique")
        _require(
            azure.control_vm in names,
            "deployment.azure_ssh.control_vm must match one of deployment.azure_ssh.vms[*].name",
        )
        _require(
            azure.superlink_vm in names,
            "deployment.azure_ssh.superlink_vm must match one of deployment.azure_ssh.vms[*].name",
        )

        total_from_vms = 0
        allowed_roles = {"superlink", "control", "worker"}
        for vm in azure.vms:
            _require(vm.name.strip() != "", "deployment.azure_ssh.vms[*].name must be non-empty")
            _require(vm.host.strip() != "", "deployment.azure_ssh.vms[*].host must be non-empty")
            _require(vm.ssh_port > 0, "deployment.azure_ssh.vms[*].ssh_port must be > 0")
            _require(vm.ssh_user.strip() != "", "deployment.azure_ssh.vms[*].ssh_user must be non-empty")
            _require(vm.supernodes_on_vm >= 0, "deployment.azure_ssh.vms[*].supernodes_on_vm must be >= 0")
            _require(
                vm.ssh_key_path.strip() != "",
                "deployment.azure_ssh.vms[*].ssh_key_path must be non-empty",
            )
            key_path = Path(vm.ssh_key_path)
            _require(key_path.is_absolute(), f"SSH key path must be absolute: {key_path}")
            _require(key_path.exists(), f"SSH key path does not exist: {key_path}")
            if not azure.allow_public_ips:
                try:
                    host_ip = ipaddress.ip_address(vm.host)
                except ValueError as exc:
                    raise ValueError(
                        f"deployment.azure_ssh.vms[*].host must be an IP literal when allow_public_ips=false: {vm.host}"
                    ) from exc
                _require(
                    host_ip.is_private,
                    f"Public host IP not allowed when allow_public_ips=false: {vm.host}",
                )

            vm_roles = set(vm.roles)
            _require(vm_roles <= allowed_roles, f"Invalid VM roles for {vm.name}: {sorted(vm_roles - allowed_roles)}")
            total_from_vms += int(vm.supernodes_on_vm)

        if azure.superlink_bind_host is not None:
            _require(
                azure.superlink_bind_host.strip() != "",
                "deployment.azure_ssh.superlink_bind_host must be non-empty when set",
            )
            if not azure.allow_public_ips:
                try:
                    bind_ip = ipaddress.ip_address(azure.superlink_bind_host)
                except ValueError as exc:
                    raise ValueError(
                        "deployment.azure_ssh.superlink_bind_host must be an IP literal when allow_public_ips=false"
                    ) from exc
                _require(
                    bind_ip.is_private,
                    "deployment.azure_ssh.superlink_bind_host must be private when allow_public_ips=false",
                )

        _require(
            total_from_vms == azure.total_supernodes,
            "sum(azure_ssh.vms[*].supernodes_on_vm) must equal deployment.azure_ssh.total_supernodes",
        )
        _require(
            azure.total_supernodes == cfg.federation.num_clients,
            "deployment.azure_ssh.total_supernodes must equal federation.num_clients",
        )

        _require(auth.enabled is True, "deployment.supernode_auth.enabled must be true in managed_azure_ssh mode")
        _require(
            len(auth.private_key_local_paths) == azure.total_supernodes,
            "deployment.supernode_auth.private_key_local_paths length must equal total_supernodes",
        )
        _require(
            len(auth.public_key_local_paths) == azure.total_supernodes,
            "deployment.supernode_auth.public_key_local_paths length must equal total_supernodes",
        )
        for p in [tls.ca_cert_local_path, tls.server_cert_local_path, tls.server_key_local_path]:
            cert_path = Path(p)
            _require(cert_path.is_absolute(), f"TLS path must be absolute: {cert_path}")
            _require(cert_path.exists(), f"TLS path does not exist: {cert_path}")
        for p in auth.private_key_local_paths + auth.public_key_local_paths:
            key_path = Path(p)
            _require(key_path.is_absolute(), f"SuperNode auth key path must be absolute: {key_path}")
            _require(key_path.exists(), f"SuperNode auth key path does not exist: {key_path}")
        _require(
            not Path(tls.remote_cert_dir).is_absolute(),
            "deployment.tls.remote_cert_dir must be relative",
        )
        _require(
            ".." not in Path(tls.remote_cert_dir).parts,
            "deployment.tls.remote_cert_dir must not contain '..'",
        )
        _require(
            not Path(auth.remote_key_dir).is_absolute(),
            "deployment.supernode_auth.remote_key_dir must be relative",
        )
        _require(
            ".." not in Path(auth.remote_key_dir).parts,
            "deployment.supernode_auth.remote_key_dir must not contain '..'",
        )


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment config from YAML or JSON."""
    return _build_config(_load_raw(Path(path)))


def resolve_non_iid(cfg: ExperimentConfig) -> dict[str, float]:
    """Resolve non-IID severities by applying axis override over global."""

    def _val(axis: float | None) -> float:
        return float(axis) if axis is not None else float(cfg.non_iid.global_)

    return {
        "class_skew": _val(cfg.non_iid.class_skew),
        "regime_skew": _val(cfg.non_iid.regime_skew),
        "context_skew": _val(cfg.non_iid.context_skew),
        "template_skew": _val(cfg.non_iid.template_skew),
    }


def apply_mode_override(cfg: ExperimentConfig, mode: str | None) -> ExperimentConfig:
    if mode is None:
        return cfg
    cfg.mode = mode
    _validate_config(cfg)
    return cfg


def dumps_config_json(cfg: ExperimentConfig) -> str:
    return json.dumps(cfg.to_dict(), separators=(",", ":"))


def loads_config_json(raw: str) -> ExperimentConfig:
    return _build_config(json.loads(raw))
