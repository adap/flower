"""Runtime state for sharing active experiment/domain across app wrappers."""

from __future__ import annotations

from typing import Any, Mapping

from .config import ExperimentConfig, dumps_config_json, loads_config_json

_ACTIVE_CONFIG: ExperimentConfig | None = None
_ACTIVE_DOMAIN: str | None = None


def set_active_experiment(cfg: ExperimentConfig, domain: str) -> None:
    global _ACTIVE_CONFIG, _ACTIVE_DOMAIN
    _ACTIVE_CONFIG = cfg
    _ACTIVE_DOMAIN = domain


def _mapping_get_str(mapping: Mapping[str, Any] | None, key: str, default: str = "") -> str:
    if mapping is None:
        return default
    value = mapping.get(key, default)
    return str(value) if value is not None else default


def get_active_experiment_from_context(
    run_config: Mapping[str, Any],
    config_fallback: Mapping[str, Any] | None = None,
) -> tuple[ExperimentConfig, str]:
    global _ACTIVE_CONFIG, _ACTIVE_DOMAIN

    # Prefer explicit payload from run/message config over stale in-process globals.
    raw = _mapping_get_str(run_config, "experiment-config-json")
    if not raw:
        raw = _mapping_get_str(config_fallback, "experiment-config-json")

    if raw:
        cfg = loads_config_json(raw)
        domain = _mapping_get_str(run_config, "domain")
        if not domain:
            domain = _mapping_get_str(config_fallback, "domain", cfg.domains[0])
        _ACTIVE_CONFIG = cfg
        _ACTIVE_DOMAIN = domain
        return cfg, domain

    if _ACTIVE_CONFIG is not None and _ACTIVE_DOMAIN is not None:
        return _ACTIVE_CONFIG, _ACTIVE_DOMAIN

    raise RuntimeError(
        "No active experiment config found. Pass `experiment-config-json` with "
        "run-config (deployment) or message config (simulation)."
    )


def build_run_config_payload(cfg: ExperimentConfig, domain: str) -> dict[str, str]:
    return {
        "experiment-config-json": dumps_config_json(cfg),
        "domain": domain,
        "run-name": cfg.artifacts.run_name,
        "output-root": cfg.artifacts.root_dir,
    }
