from sqlalchemy.orm import declarative_base

from flwr.supercore.state.schema.corestate_tables import (  # noqa: F401
    corestate_metadata,
)

Base = declarative_base(metadata=corestate_metadata)
