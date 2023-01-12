import sys

# pylint: disable=import-error, no-name-in-module
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
# pylint: enable=import-error, no-name-in-module


def version() -> str:
    try:
        flwr_version: str = importlib_metadata.version("flwr")
        return flwr_version
    except Exception:
        flwr_nightly_version: str = importlib_metadata.version("flwr-nightly")
        return flwr_nightly_version
