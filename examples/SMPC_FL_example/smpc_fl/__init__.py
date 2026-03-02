"""SMPC Federated Learning Package."""

from .client_app import app as client_app
from .server_app import app as server_app

# Import gRPC modules if available
try:
    from . import smpc_pb2
    from . import smpc_pb2_grpc
except ImportError:
    pass

__version__ = "1.0.0"
__all__ = ["client_app", "server_app"]
