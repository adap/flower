"""
Usage: python dev/build-docker-image-matrix.py --flwr-version <flower version e.g. 1.9.0>
"""

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union


class reversor:
    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return other.obj == self.obj

    def __lt__(self, other):
        return other.obj < self.obj


class DistroName(str, Enum):
    ALPINE = "alpine"
    UBUNTU = "ubuntu"


@dataclass
class Distro:
    name: "DistroName"
    version: str


LATEST_SUPPORTED_PYTHON_VERSION = "3.11"
SUPPORTED_PYTHON_VERSIONS = [
    "3.8",
    "3.9",
    "3.10",
    LATEST_SUPPORTED_PYTHON_VERSION,
]

DOCKERFILE_ROOT = "src/docker"


@dataclass
class BaseImage:
    distro: Distro
    python_version: str
    namespace_repository: str
    file_dir: str
    readme_path: str
    tags: str
    flwr_version: str


def new_base_image(flwr_version: str, python_version: str, distro: Distro) -> BaseImage:
    return BaseImage(
        distro,
        python_version,
        "flwr/base",
        f"{DOCKERFILE_ROOT}/base/{distro.name.value}",
        f"{DOCKERFILE_ROOT}/base/README.md",
        f"{flwr_version}-py{python_version}-{distro.name.value}{distro.version}",
        flwr_version,
    )


def generate_base_images(
    flwr_version: str, python_versions: List[str], distros: List[Dict[str, str]]
) -> List[BaseImage]:
    return [
        new_base_image(flwr_version, python_version, distro)
        for distro in distros
        for python_version in python_versions
    ]


@dataclass
class BinaryImage:
    namespace_repository: str
    file_dir: str
    readme_path: str
    base_image: str
    distro: Distro
    python_version: str
    tags: str


def new_binary_image(
    name: str,
    base_image: BaseImage,
    tags_fn: Optional[Callable],
) -> BinaryImage:
    tags = []
    if tags_fn is not None:
        tags += tags_fn(base_image) or []

    return BinaryImage(
        f"flwr/{name}",
        f"{DOCKERFILE_ROOT}/{name}",
        f"{DOCKERFILE_ROOT}/{name}/README.md",
        base_image.tags,
        base_image.distro,
        base_image.python_version,
        "\n".join(tags),
    )


def generate_binary_images(
    name: str,
    base_images: List[BaseImage],
    tags_fn: Optional[Callable] = None,
    filter: Optional[Callable] = None,
) -> List[BinaryImage]:
    filter = filter or (lambda _: True)

    return [
        new_binary_image(name, image, tags_fn) for image in base_images if filter(image)
    ]


def tag_latest_alpine_with_flwr_version(image: BaseImage) -> List[str]:
    if (
        image.distro.name == DistroName.ALPINE
        and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
    ):
        return [image.tags, image.flwr_version]
    else:
        return [image.tags]


def tag_latest_ubuntu_with_flwr_version(image: BaseImage) -> List[str]:
    if (
        image.distro.name == DistroName.UBUNTU
        and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
    ):
        return [image.tags, image.flwr_version]
    else:
        return [image.tags]


def create_readme_updates(
    images: List[Union[BinaryImage, BinaryImage]]
) -> Dict[str, str]:
    def version_to_tuple(version: str) -> Tuple[int, ...]:
        return tuple(map(int, version.split(".")))

    def create_list_item(tags: str) -> str:
        tags = tags.split("\n")
        tags.sort(key=lambda tag: len(tag))
        tags = "`, `".join(tags)
        return f"- `{tags}`"

    images.sort(
        key=lambda image: (
            image.distro.name,
            reversor(version_to_tuple(image.python_version)),
        )
    )

    list_items = list(map(lambda image: create_list_item(image.tags), images))

    return {
        "path": images[0].readme_path,
        "update": "\\n".join(list_items),
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Generate Github Docker workflow matrix"
    )
    arg_parser.add_argument("--flwr-version", type=str, required=True)
    args = arg_parser.parse_args()

    flwr_version = args.flwr_version

    # ubuntu base images for each supported python version
    ubuntu_base_images = generate_base_images(
        flwr_version,
        SUPPORTED_PYTHON_VERSIONS,
        [Distro(DistroName.UBUNTU, "22.04")],
    )
    # alpine base images for the latest supported python version
    alpine_base_images = generate_base_images(
        flwr_version,
        [LATEST_SUPPORTED_PYTHON_VERSION],
        [Distro(DistroName.ALPINE, "3.19")],
    )

    base_images = ubuntu_base_images + alpine_base_images

    binary_images = [
        # ubuntu and alpine images for the latest supported python version
        generate_binary_images(
            "superlink",
            base_images,
            tag_latest_alpine_with_flwr_version,
            lambda image: image.python_version == LATEST_SUPPORTED_PYTHON_VERSION,
        ),
        # ubuntu images for each supported python version
        generate_binary_images(
            "supernode",
            base_images,
            tag_latest_alpine_with_flwr_version,
            lambda image: image.distro.name == DistroName.UBUNTU
            or (
                image.distro.name == DistroName.ALPINE
                and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
            ),
        ),
        # ubuntu images for each supported python version
        generate_binary_images(
            "serverapp",
            base_images,
            tag_latest_ubuntu_with_flwr_version,
            lambda image: image.distro.name == DistroName.UBUNTU,
        ),
        # ubuntu images for each supported python version
        generate_binary_images(
            "superexec",
            base_images,
            tag_latest_ubuntu_with_flwr_version,
            lambda image: image.distro.name == DistroName.UBUNTU,
        ),
        # ubuntu images for each supported python version
        generate_binary_images(
            "clientapp",
            base_images,
            tag_latest_ubuntu_with_flwr_version,
            lambda image: image.distro.name == DistroName.UBUNTU,
        ),
    ]

    readme_updates = [
        create_readme_updates(image) for image in [base_images] + binary_images
    ]

    print(
        json.dumps(
            {
                "base": {"images": list(map(asdict, base_images))},
                "binary": {
                    "images": list(map(asdict, chain.from_iterable(binary_images))),
                },
                "readme_updates": readme_updates,
            }
        )
    )
