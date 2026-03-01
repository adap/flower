"""Runtime state for sharing active experiment/domain across app wrappers."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig, dumps_config_json, loads_config_json

_ACTIVE_CONFIG: ExperimentConfig | None = None
_ACTIVE_DOMAIN: str | None = None


def set_active_experiment(cfg: ExperimentConfig, domain: str) -> None:
    global _ACTIVE_CONFIG, _ACTIVE_DOMAIN
    _ACTIVE_CONFIG = cfg
    _ACTIVE_DOMAIN = domain


def get_active_experiment_from_context(run_config: dict[str, Any]) -> tuple[ExperimentConfig, str]:
    global _ACTIVE_CONFIG, _ACTIVE_DOMAIN

    if _ACTIVE_CONFIG is not None and _ACTIVE_DOMAIN is not None:
        return _ACTIVE_CONFIG, _ACTIVE_DOMAIN

    raw = run_config.get("experiment-config-json", "")
    if not raw:
        raise RuntimeError(
            "No active experiment config found. For deployment mode, pass "
            "`experiment-config-json` through Flower run-config."
        )

    cfg = loads_config_json(str(raw))
    domain = str(run_config.get("domain", cfg.domains[0]))

    _ACTIVE_CONFIG = cfg
    _ACTIVE_DOMAIN = domain
    return cfg, domain


def build_run_config_payload(cfg: ExperimentConfig, domain: str) -> dict[str, str]:
    return {
        "experiment-config-json": dumps_config_json(cfg),
        "domain": domain,
        "run-name": cfg.artifacts.run_name,
        "output-root": cfg.artifacts.root_dir,
    }
