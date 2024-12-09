"""Build Docker image matrix.

Usage:
python dev/build-docker-image-matrix.py --flwr-version <flower version e.g. 1.13.0>

Images are built in three workflows: stable, nightly, and unstable (main).
Each builds for `amd64` and `arm64`.

1. **Ubuntu Images**:
   - Used for images where dependencies might be installed by users.
   - Ubuntu uses `glibc`, compatible with most ML frameworks.

2. **Alpine Images**:
   - Used only for minimal images (e.g., SuperLink)
   where no extra dependencies are expected.
   - Limited use due to dependency (in particular ML frameworks)
   compilation complexity with `musl`.

Workflow Details:
- **Stable Release**: Triggered on new releases.
  Builds full matrix (all Python versions, Ubuntu and Alpine).
- **Nightly Release**: Daily trigger.
  Builds full matrix (latest Python, Ubuntu only).
- **Unstable**: Triggered on main branch commits.
  Builds simplified matrix (latest Python, Ubuntu only).
"""

import json
from dataclasses import asdict, dataclass, field

try:
    from enum import StrEnum

    class _DistroName(StrEnum):
        ALPINE = "alpine"
        UBUNTU = "ubuntu"

except ImportError:
    from enum import Enum

    class _DistroName(str, Enum):
        ALPINE = "alpine"
        UBUNTU = "ubuntu"


from typing import Any, Callable, Optional

import typer

# when we switch to Python 3.11 in the ci, we need to change the DistroName to:
# class DistroName(StrEnum):
#     ALPINE = "alpine"
#     UBUNTU = "ubuntu"
# assert sys.version_info < (3, 11), "Script requires Python 3.9 or lower."


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
class _Variant:
    distro: _Distro
    extras: Optional[Any] = None


@dataclass
class _CpuVariant:
    pass


@dataclass
class _CudaVariant:
    version: str


CUDA_VERSIONS_CONFIG = [
    ("11.2.2", "20.04"),
    ("11.8.0", "22.04"),
    ("12.1.0", "22.04"),
    ("12.3.2", "22.04"),
]
LATEST_SUPPORTED_CUDA_VERSION = _Variant(
    _Distro(_DistroName.UBUNTU, "22.04"),
    _CudaVariant(version="12.4.1"),
)

# ubuntu base image
UBUNTU_VARIANT = _Variant(
    _Distro(_DistroName.UBUNTU, "24.04"),
    _CpuVariant(),
)


# alpine base image
ALPINE_VARIANT = _Variant(
    _Distro(_DistroName.ALPINE, "3.19"),
    _CpuVariant(),
)


# ubuntu cuda base images
CUDA_VARIANTS = [
    _Variant(
        _Distro(_DistroName.UBUNTU, ubuntu_version),
        _CudaVariant(version=cuda_version),
    )
    for (cuda_version, ubuntu_version) in CUDA_VERSIONS_CONFIG
] + [LATEST_SUPPORTED_CUDA_VERSION]


def _remove_patch_version(version: str) -> str:
    return ".".join(version.split(".")[0:2])


@dataclass
class _BaseImageBuilder:  # pylint: disable=too-many-instance-attributes
    file_dir_fn: Callable[[Any], str]
    tags_fn: Callable[[Any], list[str]]
    build_args_fn: Callable[[Any], str]
    build_args: Any
    tags: list[str] = field(init=False)
    file_dir: str = field(init=False)
    tags_encoded: str = field(init=False)
    build_args_encoded: str = field(init=False)


@dataclass
class _BaseImage(_BaseImageBuilder):
    namespace_repository: str = "flwr/base"

    @property
    def file_dir(self) -> str:
        """File directory."""
        return self.file_dir_fn(self.build_args)

    @property
    def tags(self) -> list[str]:
        """Get list of tags."""
        return self.tags_fn(self.build_args)

    @property
    def tags_encoded(self) -> str:
        """Encoded tags."""
        return "\n".join(self.tags)

    @property
    def build_args_encoded(self) -> str:
        """Build arguments."""
        return self.build_args_fn(self.build_args)


@dataclass
class _BinaryImage:
    namespace_repository: str
    file_dir: str
    base_image: str
    tags_encoded: str


def _new_binary_image(
    name: str,
    base_image: _BaseImage,
    tags_fn: Optional[Callable],
) -> _BinaryImage:
    tags = []
    if tags_fn is not None:
        tags += tags_fn(base_image) or []

    return _BinaryImage(
        namespace_repository=f"flwr/{name}",
        file_dir=f"{DOCKERFILE_ROOT}/{name}",
        base_image=base_image.tags[0],
        tags_encoded="\n".join(tags),
    )


def _generate_binary_images(
    name: str,
    base_images: list[_BaseImage],
    tags_fn: Optional[Callable] = None,
    filter_func: Optional[Callable] = None,
) -> list[_BinaryImage]:
    filter_func = filter_func or (lambda _: True)

    return [
        _new_binary_image(name, image, tags_fn)
        for image in base_images
        if filter_func(image)
    ]


def _tag_latest_alpine_with_flwr_version(image: _BaseImage) -> list[str]:
    if (
        image.build_args.variant.distro.name == _DistroName.ALPINE
        and image.build_args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
    ):
        return image.tags + [image.build_args.flwr_version]
    return image.tags


def _tag_latest_ubuntu_with_flwr_version(image: _BaseImage) -> list[str]:
    if (
        image.build_args.variant.distro.name == _DistroName.UBUNTU
        and image.build_args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
        and isinstance(image.build_args.variant.extras, _CpuVariant)
    ):
        return image.tags + [image.build_args.flwr_version]
    return image.tags


#
# Build matrix for stable releases
#
def _build_stable_matrix(
    flwr_version: str,
) -> tuple[list[_BaseImage], list[_BinaryImage]]:
    @dataclass
    class _StableBaseImageBuildArgs:
        variant: _Variant
        python_version: str
        flwr_version: str

    cpu_build_args = """PYTHON_VERSION={python_version}
FLWR_VERSION={flwr_version}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""

    cpu_build_args_variants = [
        _StableBaseImageBuildArgs(UBUNTU_VARIANT, python_version, flwr_version)
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ] + [
        _StableBaseImageBuildArgs(
            ALPINE_VARIANT, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version
        )
    ]

    cpu_base_images = [
        _BaseImage(
            file_dir_fn=lambda args: (
                f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}"
            ),
            tags_fn=lambda args: [
                f"{args.flwr_version}-py{args.python_version}-"
                f"{args.variant.distro.name.value}{args.variant.distro.version}"
            ],
            build_args_fn=lambda args: cpu_build_args.format(
                python_version=args.python_version,
                flwr_version=args.flwr_version,
                distro_name=args.variant.distro.name,
                distro_version=args.variant.distro.version,
            ),
            build_args=build_args_variant,
        )
        for build_args_variant in cpu_build_args_variants
    ]

    cuda_build_args_variants = [
        _StableBaseImageBuildArgs(variant, python_version, flwr_version)
        for variant in CUDA_VARIANTS
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ]

    cuda_build_args = cpu_build_args + """CUDA_VERSION={cuda_version}"""

    _cuda_base_image = [
        _BaseImage(
            file_dir_fn=lambda args: (
                f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}-cuda"
            ),
            tags_fn=lambda args: [
                f"{args.flwr_version}-py{args.python_version}-"
                f"cu{_remove_patch_version(args.variant.extras.version)}-"
                f"{args.variant.distro.name.value}{args.variant.distro.version}",
            ],
            build_args_fn=lambda args: cuda_build_args.format(
                python_version=args.python_version,
                flwr_version=args.flwr_version,
                distro_name=args.variant.distro.name,
                distro_version=args.variant.distro.version,
                cuda_version=args.variant.extras.version,
            ),
            build_args=build_args_variant,
        )
        for build_args_variant in cuda_build_args_variants
    ]

    # base_images = cpu_base_images + cuda_base_image
    base_images = cpu_base_images

    binary_images = (
        # ubuntu and alpine images for the latest supported python version
        _generate_binary_images(
            "superlink",
            base_images,
            _tag_latest_alpine_with_flwr_version,
            lambda image: image.build_args.python_version
            == LATEST_SUPPORTED_PYTHON_VERSION
            and isinstance(image.build_args.variant.extras, _CpuVariant),
        )
        # ubuntu images for each supported python version
        + _generate_binary_images(
            "supernode",
            base_images,
            _tag_latest_alpine_with_flwr_version,
            lambda image: (
                image.build_args.variant.distro.name == _DistroName.UBUNTU
                and isinstance(image.build_args.variant.extras, _CpuVariant)
            )
            or (
                image.build_args.variant.distro.name == _DistroName.ALPINE
                and image.build_args.python_version == LATEST_SUPPORTED_PYTHON_VERSION
            ),
        )
        # ubuntu images for each supported python version
        + _generate_binary_images(
            "serverapp",
            base_images,
            _tag_latest_ubuntu_with_flwr_version,
            lambda image: image.build_args.variant.distro.name == _DistroName.UBUNTU,
        )
        # ubuntu images for each supported python version
        + _generate_binary_images(
            "clientapp",
            base_images,
            _tag_latest_ubuntu_with_flwr_version,
            lambda image: image.build_args.variant.distro.name == _DistroName.UBUNTU,
        )
    )

    return base_images, binary_images


#
# Build matrix for unstable releases
#
def _build_unstable_matrix(
    flwr_version_ref: str,
) -> tuple[list[_BaseImage], list[_BinaryImage]]:
    @dataclass
    class _UnstableBaseImageBuildArgs:
        variant: _Variant
        python_version: str
        flwr_version_ref: str

    cpu_ubuntu_build_args_variant = _UnstableBaseImageBuildArgs(
        UBUNTU_VARIANT, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version_ref
    )

    cpu_build_args = """PYTHON_VERSION={python_version}
FLWR_VERSION_REF={flwr_version_ref}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""

    cpu_base_image = _BaseImage(
        file_dir_fn=lambda args: (
            f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}"
        ),
        tags_fn=lambda _: ["unstable"],
        build_args_fn=lambda args: cpu_build_args.format(
            python_version=args.python_version,
            flwr_version_ref=args.flwr_version_ref,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
        ),
        build_args=cpu_ubuntu_build_args_variant,
    )

    cuda_build_args_variant = _UnstableBaseImageBuildArgs(
        LATEST_SUPPORTED_CUDA_VERSION, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version_ref
    )

    cuda_build_args = cpu_build_args + """CUDA_VERSION={cuda_version}"""

    _cuda_base_image = _BaseImage(
        file_dir_fn=lambda args: (
            f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}-cuda"
        ),
        tags_fn=lambda _: ["unstable-cuda"],
        build_args_fn=lambda args: cuda_build_args.format(
            python_version=args.python_version,
            flwr_version_ref=args.flwr_version_ref,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
            cuda_version=args.variant.extras.version,
        ),
        build_args=cuda_build_args_variant,
    )

    # base_images = [cpu_base_image, cuda_base_image]
    base_images = [cpu_base_image]

    binary_images = (
        _generate_binary_images(
            "superlink",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, _CpuVariant),
        )
        + _generate_binary_images(
            "supernode",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, _CpuVariant),
        )
        + _generate_binary_images("serverapp", base_images, lambda image: image.tags)
        + _generate_binary_images("clientapp", base_images, lambda image: image.tags)
    )

    return base_images, binary_images


#
# Build matrix for nightly releases
#
def _build_nightly_matrix(
    flwr_version: str, flwr_package: str
) -> tuple[list[_BaseImage], list[_BinaryImage]]:
    @dataclass
    class _NightlyBaseImageBuildArgs:
        variant: _Variant
        python_version: str
        flwr_version: str
        flwr_package: str

    cpu_ubuntu_build_args_variant = _NightlyBaseImageBuildArgs(
        UBUNTU_VARIANT, LATEST_SUPPORTED_PYTHON_VERSION, flwr_version, flwr_package
    )

    cpu_build_args = """PYTHON_VERSION={python_version}
FLWR_VERSION={flwr_version}
FLWR_PACKAGE={flwr_package}
DISTRO={distro_name}
DISTRO_VERSION={distro_version}
"""

    cpu_base_image = _BaseImage(
        file_dir_fn=lambda args: (
            f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}"
        ),
        tags_fn=lambda args: [args.flwr_version, "nightly"],
        build_args_fn=lambda args: cpu_build_args.format(
            python_version=args.python_version,
            flwr_version=args.flwr_version,
            flwr_package=args.flwr_package,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
        ),
        build_args=cpu_ubuntu_build_args_variant,
    )

    cuda_build_args_variant = _NightlyBaseImageBuildArgs(
        LATEST_SUPPORTED_CUDA_VERSION,
        LATEST_SUPPORTED_PYTHON_VERSION,
        flwr_version,
        flwr_package,
    )

    cuda_build_args = cpu_build_args + """CUDA_VERSION={cuda_version}"""

    _cuda_base_image = _BaseImage(
        file_dir_fn=lambda args: (
            f"{DOCKERFILE_ROOT}/base/{args.variant.distro.name.value}-cuda"
        ),
        tags_fn=lambda args: [f"{args.flwr_version}-cuda", "nightly-cuda"],
        build_args_fn=lambda args: cuda_build_args.format(
            python_version=args.python_version,
            flwr_version=args.flwr_version,
            flwr_package=args.flwr_package,
            distro_name=args.variant.distro.name,
            distro_version=args.variant.distro.version,
            cuda_version=args.variant.extras.version,
        ),
        build_args=cuda_build_args_variant,
    )

    # base_images = [cpu_base_image, cuda_base_image]
    base_images = [cpu_base_image]

    binary_images = (
        _generate_binary_images(
            "superlink",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, _CpuVariant),
        )
        + _generate_binary_images(
            "supernode",
            base_images,
            lambda image: image.tags,
            lambda image: isinstance(image.build_args.variant.extras, _CpuVariant),
        )
        + _generate_binary_images("serverapp", base_images, lambda image: image.tags)
        + _generate_binary_images("clientapp", base_images, lambda image: image.tags)
    )

    return base_images, binary_images


def build_images(
    flwr_version: str = typer.Option(..., help="The Flower version"),
    flwr_package: str = typer.Option("flwr", help="The Flower package"),
    matrix: str = typer.Option(
        "stable",
        help="The workflow matrix type",
        case_sensitive=False,
    ),
):
    """Build updated docker images."""
    if matrix == "stable":
        base_images, binary_images = _build_stable_matrix(flwr_version)
    elif matrix == "nightly":
        base_images, binary_images = _build_nightly_matrix(flwr_version, flwr_package)
    else:
        base_images, binary_images = _build_unstable_matrix(flwr_version)

    print(
        json.dumps(
            {
                "base": {
                    "images": [
                        asdict(
                            image,
                            dict_factory=lambda x: {
                                k: v
                                for (k, v) in x
                                if v is not None and callable(v) is False
                            },
                        )
                        for image in base_images
                    ]
                },
                "binary": {"images": [asdict(image) for image in binary_images]},
            }
        )
    )


if __name__ == "__main__":
    typer.run(build_images)
