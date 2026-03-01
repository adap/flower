"""Utility helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def stable_int_seed(*parts: str | int) -> int:
    s = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_builtin(v) for v in x]
    if isinstance(x, tuple):
        return [to_builtin(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x
