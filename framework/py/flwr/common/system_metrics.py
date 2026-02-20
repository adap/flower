# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""System metrics helpers for profiling."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover - not available on all platforms
    resource = None


BLOCK_SIZE_BYTES = 512


@dataclass(frozen=True)
class DiskIoSnapshot:
    """Disk IO snapshot in MB."""

    read_mb: float | None
    write_mb: float | None
    source: str | None


def _bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024**2)


def _blocks_to_mb(blocks: int | float) -> float:
    return (float(blocks) * BLOCK_SIZE_BYTES) / (1024**2)


def read_disk_io_mb(proc: "psutil.Process | None") -> DiskIoSnapshot:
    """Return disk IO snapshot (MB) and its source."""
    # Per-process IO
    if proc is not None and hasattr(proc, "io_counters"):
        try:
            counters = proc.io_counters()
            if counters is not None:
                return DiskIoSnapshot(
                    read_mb=_bytes_to_mb(counters.read_bytes),
                    write_mb=_bytes_to_mb(counters.write_bytes),
                    source="process",
                )
        except Exception:
            pass

    # System-wide IO
    if psutil is not None:
        try:
            counters = psutil.disk_io_counters()
            if counters is not None:
                return DiskIoSnapshot(
                    read_mb=_bytes_to_mb(counters.read_bytes),
                    write_mb=_bytes_to_mb(counters.write_bytes),
                    source="system",
                )
        except Exception:
            pass

    # Resource fallback (per-process blocks, approximate)
    if resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return DiskIoSnapshot(
                read_mb=_blocks_to_mb(usage.ru_inblock),
                write_mb=_blocks_to_mb(usage.ru_oublock),
                source="resource",
            )
        except Exception:
            pass

    return DiskIoSnapshot(read_mb=None, write_mb=None, source=None)
