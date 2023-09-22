"""Flower package version helper."""


import importlib.metadata as importlib_metadata
from typing import Tuple


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
