"""
Usage: python dev/build-docker-image-matrix.py --flwr-version <flower version e.g. 1.13.0>

Images are built in three workflows: stable, nightly, and unstable (main).
Each builds for `amd64` and `arm64`.

1. **Ubuntu Images**:
   - Used for images where dependencies might be installed by users.
   - Ubuntu uses `glibc`, compatible with most ML frameworks.

2. **Alpine Images**:
   - Used only for minimal images (e.g., SuperLink) where no extra dependencies are expected.
   - Limited use due to dependency (in particular ML frameworks) compilation complexity with `musl`.

Workflow Details:
- **Stable Release**: Triggered on new releases. Builds full matrix (all Python versions, Ubuntu and Alpine).
- **Nightly Release**: Daily trigger. Builds full matrix (latest Python, Ubuntu only).
- **Unstable**: Triggered on main branch commits. Builds simplified matrix (latest Python, Ubuntu only).
"""

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class DistroName(str, Enum):
    ALPINE = "alpine"
    UBUNTU = "ubuntu"


@dataclass
class Distro:
    name: "DistroName"
    version: str


LATEST_SUPPORTED_PYTHON_VERSION = "3.11"
SUPPORTED_PYTHON_VERSIONS = [
    "3.9",
    "3.10",
    LATEST_SUPPORTED_PYTHON_VERSION,
]

DOCKERFILE_ROOT = "src/docker"


@dataclass
class Variant:
    distro: Distro
    file_dir_fn: Callable[[Distro, dict], str]
    tag_fn: Callable[[Distro, str, str, dict], str]
    build_args_fn: Callable[[Distro, str, str, dict], str]
    extras: Optional[Any] = None

    def file_dir(self) -> str:
        return self.file_dir_fn(self.distro, self.extras)

    def tag(self, flwr_version: str, python_version: str) -> str:
        return self.tag_fn(self.distro, flwr_version, python_version, self.extras)

    def build_args(self, flwr_version: str, python_version: str) -> str:
        return self.build_args_fn(
            self.distro, flwr_version, python_version, self.extras
        )


@dataclass
class CpuVariant:
    pass


CPU_BUILD_ARGS = """PYTHON_VERSION={python_version}
FLWR_VERSION={flwr_version}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""


@dataclass
class CudaVariant:
    version: str


CUDA_BUILD_ARGS = (
    CPU_BUILD_ARGS
    + """CUDA_VERSION={cuda_version}
"""
)


LATEST_SUPPORTED_CUDA_VERSIONS = "12.4.1"
CUDA_VERSIONS_CONFIG = [
    ("11.2.2", "20.04"),
    ("11.8.0", "22.04"),
    ("12.1.0", "22.04"),
    ("12.3.2", "22.04"),
    (LATEST_SUPPORTED_CUDA_VERSIONS, "22.04"),
]


def remove_patch_version(version: str) -> str:
    return ".".join(version.split(".")[0:2])


# ubuntu base images for each supported python version
UBUNTU_VARIANTS = [
    Variant(
        Distro(DistroName.UBUNTU, "24.04"),
        lambda distro, _: f"{DOCKERFILE_ROOT}/base/{distro.name.value}",
        lambda distro, flwr_version, python_version, _: f"{flwr_version}-py{python_version}-{distro.name.value}{distro.version}",
        lambda distro, flwr_version, python_version, _: CPU_BUILD_ARGS.format(
            python_version=python_version,
            flwr_version=flwr_version,
            distro_name=distro.name,
            distro_version=distro.version,
        ),
        CpuVariant(),
    )
]


# alpine base images for the latest supported python version
ALPINE_VARIANTS = [
    Variant(
        Distro(DistroName.ALPINE, "3.19"),
        lambda distro, _: f"{DOCKERFILE_ROOT}/base/{distro.name.value}",
        lambda distro, flwr_version, python_version, _: f"{flwr_version}-py{python_version}-{distro.name.value}{distro.version}",
        lambda distro, flwr_version, python_version, _: CPU_BUILD_ARGS.format(
            python_version=python_version,
            flwr_version=flwr_version,
            distro_name=distro.name,
            distro_version=distro.version,
        ),
        CpuVariant(),
    )
]

# ubuntu cuda base images for each supported python and cuda version
CUDA_VARIANTS = [
    Variant(
        Distro(DistroName.UBUNTU, ubuntu_version),
        lambda distro, _: f"{DOCKERFILE_ROOT}/base/{distro.name.value}-cuda",
        lambda distro, flwr_version, python_version, extras: f"{flwr_version}-py{python_version}-cu{remove_patch_version(extras.version)}-{distro.name.value}{distro.version}",
        lambda distro, flwr_version, python_version, extras: CUDA_BUILD_ARGS.format(
            python_version=python_version,
            flwr_version=flwr_version,
            distro_name=distro.name,
            distro_version=distro.version,
            cuda_version=extras.version,
        ),
        CudaVariant(version=cuda_version),
    )
    for (cuda_version, ubuntu_version) in CUDA_VERSIONS_CONFIG
]


@dataclass
class BaseImage:
    variant: Variant
    python_version: str
    namespace_repository: str
    file_dir: str
    tag: str
    flwr_version: str
    build_args: str


def new_base_image(
    flwr_version: str, python_version: str, variant: Variant
) -> Dict[str, Any]:
    return BaseImage(
        variant,
        python_version,
        "flwr/base",
        variant.file_dir(),
        variant.tag(flwr_version, python_version),
        flwr_version,
        variant.build_args(flwr_version, python_version),
    )


def generate_base_images(
    flwr_version: str, python_versions: List[str], variants: List[Variant]
) -> List[Dict[str, Any]]:
    return [
        new_base_image(flwr_version, python_version, variant)
        for variant in variants
        for python_version in python_versions
    ]


@dataclass
class BinaryImage:
    namespace_repository: str
    file_dir: str
    base_image: str
    tags: List[str]


def new_binary_image(
    name: str,
    base_image: BaseImage,
    tags_fn: Optional[Callable],
) -> Dict[str, Any]:
    tags = []
    if tags_fn is not None:
        tags += tags_fn(base_image) or []

    return BinaryImage(
        f"flwr/{name}",
        f"{DOCKERFILE_ROOT}/{name}",
        base_image.tag,
        "\n".join(tags),
    )


def generate_binary_images(
    name: str,
    base_images: List[BaseImage],
    tags_fn: Optional[Callable] = None,
    filter: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    filter = filter or (lambda _: True)

    return [
        new_binary_image(name, image, tags_fn) for image in base_images if filter(image)
    ]


def tag_latest_alpine_with_flwr_version(image: BaseImage) -> List[str]:
    if (
        image.variant.distro.name == DistroName.ALPINE
        and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
    ):
        return [image.tag, image.flwr_version]
    else:
        return [image.tag]


def tag_latest_ubuntu_with_flwr_version(image: BaseImage) -> List[str]:
    if (
        image.variant.distro.name == DistroName.UBUNTU
        and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
        and isinstance(image.variant.extras, CpuVariant)
    ):
        return [image.tag, image.flwr_version]
    else:
        return [image.tag]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Generate Github Docker workflow matrix"
    )
    arg_parser.add_argument("--flwr-version", type=str, required=True)
    args = arg_parser.parse_args()

    flwr_version = args.flwr_version

    ubuntu_base_images = generate_base_images(
        flwr_version,
        SUPPORTED_PYTHON_VERSIONS,
        UBUNTU_VARIANTS,
    )

    alpine_base_images = generate_base_images(
        flwr_version,
        [LATEST_SUPPORTED_PYTHON_VERSION],
        ALPINE_VARIANTS,
    )

    cuda_base_images = generate_base_images(
        flwr_version,
        SUPPORTED_PYTHON_VERSIONS,
        CUDA_VARIANTS,
    )

    base_images = ubuntu_base_images + alpine_base_images

    binary_images = (
        # ubuntu and alpine images for the latest supported python version
        generate_binary_images(
            "superlink",
            base_images,
            tag_latest_alpine_with_flwr_version,
            lambda image: image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
            and isinstance(image.variant.extras, CpuVariant),
        )
        # ubuntu images for each supported python version
        + generate_binary_images(
            "supernode",
            base_images,
            tag_latest_alpine_with_flwr_version,
            lambda image: (
                image.variant.distro.name == DistroName.UBUNTU
                and isinstance(image.variant.extras, CpuVariant),
            )
            or (
                image.variant.distro.name == DistroName.ALPINE
                and image.python_version == LATEST_SUPPORTED_PYTHON_VERSION
            ),
        )
        # ubuntu images for each supported python version
        + generate_binary_images(
            "serverapp",
            base_images,
            tag_latest_ubuntu_with_flwr_version,
            lambda image: image.variant.distro.name == DistroName.UBUNTU,
        )
        # ubuntu images for each supported python version
        + generate_binary_images(
            "clientapp",
            base_images,
            tag_latest_ubuntu_with_flwr_version,
            lambda image: image.variant.distro.name == DistroName.UBUNTU,
        )
    )

    print(
        json.dumps(
            {
                "base": {
                    "images": list(
                        map(
                            lambda image: asdict(
                                image,
                                dict_factory=lambda x: {
                                    k: v
                                    for (k, v) in x
                                    if v is not None and callable(v) is False
                                },
                            ),
                            base_images,
                        )
                    )
                },
                "binary": {
                    "images": list(map(lambda image: asdict(image), binary_images))
                },
            }
        )
    )
