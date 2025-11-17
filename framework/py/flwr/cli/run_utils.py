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
"""Flower command line interface utils."""


from dataclasses import dataclass
from datetime import datetime, timedelta

from flwr.common.date import format_timedelta, isoformat8601_utc
from flwr.common.typing import Run


@dataclass
class RunRow:  # pylint: disable=too-many-instance-attributes
    """Represents a single run's data for display."""

    run_id: int
    federation: str
    fab_id: str
    fab_version: str
    fab_hash: str
    status_text: str
    elapsed: str
    pending_at: str
    starting_at: str
    running_at: str
    finished_at: str


def format_runs(run_dict: dict[int, Run], now_isoformat: str) -> list[RunRow]:
    """Format runs to a list of RunRow objects."""

    def _format_datetime(dt: datetime | None) -> str:
        return isoformat8601_utc(dt).replace("T", " ") if dt else "N/A"

    run_list: list[RunRow] = []

    # Add rows
    for run in sorted(
        run_dict.values(), key=lambda x: datetime.fromisoformat(x.pending_at)
    ):
        # Combine status and sub-status into a single string
        if run.status.sub_status == "":
            status_text = run.status.status
        else:
            status_text = f"{run.status.status}:{run.status.sub_status}"

        # Convert isoformat to datetime
        pending_at = datetime.fromisoformat(run.pending_at) if run.pending_at else None
        starting_at = (
            datetime.fromisoformat(run.starting_at) if run.starting_at else None
        )
        running_at = datetime.fromisoformat(run.running_at) if run.running_at else None
        finished_at = (
            datetime.fromisoformat(run.finished_at) if run.finished_at else None
        )

        # Calculate elapsed time
        elapsed_time = timedelta()
        if running_at:
            if finished_at:
                end_time = finished_at
            else:
                end_time = datetime.fromisoformat(now_isoformat)
            elapsed_time = end_time - running_at

        row = RunRow(
            run_id=run.run_id,
            federation=run.federation,
            fab_id=run.fab_id,
            fab_version=run.fab_version,
            fab_hash=run.fab_hash,
            status_text=status_text,
            elapsed=format_timedelta(elapsed_time),
            pending_at=_format_datetime(pending_at),
            starting_at=_format_datetime(starting_at),
            running_at=_format_datetime(running_at),
            finished_at=_format_datetime(finished_at),
        )
        run_list.append(row)
    return run_list
