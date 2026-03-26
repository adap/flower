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
"""Store per-run federation config as JSON.

Revision ID: f19d0b5e7c2a
Revises: c8f4f6e2c1ad
Create Date: 2026-03-26 14:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "f19d0b5e7c2a"
down_revision: str | Sequence[str] | None = "c8f4f6e2c1ad"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("run", sa.Column("federation_config", sa.String(), nullable=True))
    op.drop_column("run", "federation_options")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "run", sa.Column("federation_options", sa.LargeBinary(), nullable=True)
    )
    op.drop_column("run", "federation_config")
