
## Building the base Docker images

We provide the base images used in the Dockerfile of the parent directory (i.e. `jafermarq/jetsonfederated_cpu` and `jafermarq/jetsonfederated_gpu`). To make the process of running the demo as seamsless as possible (i.e. without long Docker build times) we have pre-built these images and uploaded them to dockerhub. In that way, the Dockerfile in the parent directory only requires adding a couple of python scripts to the image. If you want to build these images by yourself, you can do so by running the `build.sh` script in each directory. Note that building these images might take around one hour, depending on your system's specs.

These images target a `aarch64` machine (e.g. RPi) but you'd probably will be building these images on a `x86_64` machine. To achieve this you'll need `qemu`. You should enable this before building the images by doing:

```bash
$ docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

More details can be found in the [`qemu-user-static`](https://github.com/multiarch/qemu-user-static) repository.
