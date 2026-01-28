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
"""Helpers for running and validating Alembic migrations."""


from sqlalchemy import MetaData

from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata


def get_combined_metadata() -> MetaData:
    """Combine all Flower state metadata objects into a single MetaData instance.

    This ensures Alembic can track all tables across CoreState, LinkState, and
    ObjectStore.

    Returns
    -------
    MetaData
        Combined SQLAlchemy MetaData with all Flower state tables.
    """
    # Start with linkstate tables
    metadata = create_linkstate_metadata()

    # Add corestate tables
    corestate_metadata = create_corestate_metadata()
    for table in corestate_metadata.tables.values():
        table.to_metadata(metadata)

    # Add objectstore tables
    objectstore_metadata = create_objectstore_metadata()
    for table in objectstore_metadata.tables.values():
        table.to_metadata(metadata)

    return metadata
