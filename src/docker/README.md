# Flower on Docker

This directory contains the Flower Docker images for various purposes.

## Images

### Base

This is the default base image on which all others build.

### Server

Default Flower server image.

## How to use

This guide is for contributors and developers who want to build the images themself.
Regular users can just pull the images from Docker hub.

### Run all tests

```bash
./test.sh
```

### Production

Build server image for specific Flower version.

```bash
FLWR_VERSION=1.4.0 BUILD_TARGET=server ./build.sh
```
