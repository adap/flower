"""Config schema and validation for Comcast FL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
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
class DeploymentConfig:
    superlink: str | None = None
    federation: str | None = None
    stream_logs: bool = True


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
    deployment = DeploymentConfig(**raw.get("deployment", {}))

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

    s = resolve_non_iid(cfg)
    for k, v in s.items():
        _require(0.0 <= v <= 1.0, f"non_iid {k} must be in [0,1]")


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
