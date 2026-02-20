# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface `federation` utilities."""

import click


def parse_node_ids(raw: str) -> list[int]:
    """Parse a comma-separated string of node IDs into a list of ints."""
    try:
        return [int(nid.strip()) for nid in raw.split(",") if nid.strip()]
    except ValueError as exc:
        raise click.BadParameter(
            f"Invalid node IDs '{raw}'. Expected comma-separated integers "
            "(e.g. 124 or 124,125,126)."
        ) from exc
