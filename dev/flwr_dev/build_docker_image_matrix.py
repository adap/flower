"""Build all Docker images."""

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Annotated, Any, Callable, Optional

import typer


class _DistroName(str, Enum):
    ALPINE = "alpine"
    UBUNTU = "ubuntu"


@dataclass
class _Distro:
    name: "_DistroName"
    version: str


LATEST_SUPPORTED_PYTHON_VERSION = "3.11"
SUPPORTED_PYTHON_VERSIONS = [
    "3.9",
    "3.10",
    LATEST_SUPPORTED_PYTHON_VERSION,
]

DOCKERFILE_ROOT = "src/docker"


@dataclass
class _BaseImage:
    distro: _Distro
    python_version: str
    namespace_repository: str
    file_dir: str
    tag: str
    flwr_version: str


def _new_base_image(
    flwr_version: str, python_version: str, distro: _Distro
) -> dict[str, Any]:
    return _BaseImage(
        distro,
        python_version,
        "flwr/base",
        f"{DOCKERFILE_ROOT}/base/{distro.name.value}",
        f"{flwr_version}-py{python_version}-{distro.name.value}{distro.version}",
        flwr_version,
    )


def _generate_base_images(
    flwr_version: str, python_versions: list[str], distros: list[dict[str, str]]
) -> list[dict[str, Any]]:
    return [
        _new_base_image(flwr_version, python_version, distro)
        for distro in distros
        for python_version in python_versions
    ]


@dataclass
class _BinaryImage:
    namespace_repository: str
    file_dir: str
    base_image: str
    tags: list[str]


def _new_binary_image(
    name: str,
    base_image: _BaseImage,
    tags_fn: Optional[Callable],
) -> dict[str, Any]:
    tags = []
    if tags_fn is not None:
        tags += tags_fn(base_image) or []

    return _BinaryImage(
        f"flwr/{name}",
        f"{DOCKERFILE_ROOT}/{name}",
        base_image.tag,
        "\n".join(tags),
    )


def _generate_binary_images(
    name: str,
    base_images: list[_BaseImage],
    tags_fn: Optional[Callable] = None,
    filter_func: Optional[Callable] = None,
) -> list[dict[str, Any]]:
    filter_func = filter_func or (lambda _: True)

    return [
        _new_binary_image(name, image, tags_fn)
        for image in base_images
        if filter_func(image)
    ]


def _tag_latest_alpine_with_flwr_version(image: _BaseImage) -> list[str]:
    if (
        image.distro.name == _DistroName.ALPINE
        and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
    ):
        return [image.tag, image.flwr_version]
    return [image.tag]


def _tag_latest_ubuntu_with_flwr_version(image: _BaseImage) -> list[str]:
    if (
        image.distro.name == _DistroName.UBUNTU
        and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
    ):
        return [image.tag, image.flwr_version]
    return [image.tag]


def build_images(
    flwr_version: Annotated[
        str, typer.Argument(help="Version of Flower to build the Docker images for")
    ]
):
    """Generate all Docker images for a given version."""
    # ubuntu base images for each supported python version
    ubuntu_base_images = _generate_base_images(
        flwr_version,
        SUPPORTED_PYTHON_VERSIONS,
        [_Distro(_DistroName.UBUNTU, "22.04")],
    )
    # alpine base images for the latest supported python version
    alpine_base_images = _generate_base_images(
        flwr_version,
        [LATEST_SUPPORTED_PYTHON_VERSION],
        [_Distro(_DistroName.ALPINE, "3.19")],
    )

    base_images = ubuntu_base_images + alpine_base_images

    binary_images = (
        # ubuntu and alpine images for the latest supported python version
        _generate_binary_images(
            "superlink",
            base_images,
            _tag_latest_alpine_with_flwr_version,
            lambda image: image.python_version == LATEST_SUPPORTED_PYTHON_VERSION,
        )
        # ubuntu images for each supported python version
        + _generate_binary_images(
            "supernode",
            base_images,
            _tag_latest_alpine_with_flwr_version,
            lambda image: image.distro.name == _DistroName.UBUNTU
            or (
                image.distro.name == _DistroName.ALPINE
                and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
            ),
        )
        # ubuntu images for each supported python version
        + _generate_binary_images(
            "serverapp",
            base_images,
            _tag_latest_ubuntu_with_flwr_version,
            lambda image: image.distro.name == _DistroName.UBUNTU,
        )
        # ubuntu images for each supported python version
        + _generate_binary_images(
            "superexec",
            base_images,
            _tag_latest_ubuntu_with_flwr_version,
            lambda image: image.distro.name == _DistroName.UBUNTU,
        )
        # ubuntu images for each supported python version
        + _generate_binary_images(
            "clientapp",
            base_images,
            _tag_latest_ubuntu_with_flwr_version,
            lambda image: image.distro.name == _DistroName.UBUNTU,
        )
    )

    print(
        json.dumps(
            {
                "base": {"images": [asdict(image) for image in base_images]},
                "binary": {"images": [asdict(image) for image in binary_images]},
            }
        )
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Generate Github Docker workflow matrix"
    )
    arg_parser.add_argument("--flwr-version", type=str, required=True)
    args = arg_parser.parse_args()

    build_images(args.flwr_version)
