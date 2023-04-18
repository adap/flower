"""Flower package version helper."""
import sys
from typing import Tuple

# pylint: disable=import-error, no-name-in-module
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
# pylint: enable=import-error, no-name-in-module


def _check_package(name: str) -> Tuple[str, str]:
    version: str = importlib_metadata.version(name)
    return name, version


def _version() -> Tuple[str, str]:
    """Read and return Flower package name and version.

    Returns
    -------
    package_name, package_version : Tuple[str, str]
    """
    for name in ["flwr", "flwr-nightly"]:
        try:
            return _check_package(name)
        except importlib_metadata.PackageNotFoundError:
            pass

    return ("unknown", "unknown")


package_name, package_version = _version()
