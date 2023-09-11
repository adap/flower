# Build Docker images

This document describes how to build and publish the Docker images residing in `src/docker`.

## Images

### base

This is the default base image on which all others build. It is not published.

### server

The default Flower server image which is published on Docker Hub. The image can be used similarly to the Flower server and can be run standalone.

## Setup

Building the images requires that Docker is installed on the machine executing the build.

This guide is for contributors and developers who want to build the images themself. Regular users can just pull the images from Docker Hub. Installing Docker can be achived by following the [Get Docker](https://docs.docker.com/get-docker/) documentation.

## Test

Start with running the tests to ensure the images are correctly building and working. Running the test for a particular image can be done the following way.

```bash
FLWR_VERSION=1.5.0 BUILD_TARGET=server ./test.sh
```

## Build

Build a particular image for a specific Flower version using the following command.

```bash
FLWR_VERSION=1.5.0 BUILD_TARGET=server ./build.sh
```

## Publish

After the image was build it can be published using the following command given the right access rights are present.

```bash
FLWR_VERSION=1.5.0 BUILD_TARGET=server ./publish.sh
```

The default behaviour here is that an image will be published to Docker Hub under the names:

- `flwr/[BUILD-TARGET]:[FLWR_VERSION]`
- `flwr/[BUILD-TARGET]:latest`

Beware that the latest tag will be bound to the latest published image.

## Run

After the image is build locally it can be run using the following command. In case it was not locally build the Docker runtime will try to pull it from Docker Hub.

```bash
docker run -p 9091:9091 -p 9092:9092 flwr/server --rest
```

You can pass any option you would pass to flwr-server here after the image name as seen in the example. This works because the flwr-server is defined as an entrypoint in the image.
