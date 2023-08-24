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

### Test

```bash
# ./build build my_version_tag
./build test test
```

### Production

```bash
# ./build build my_version_tag
./build build 0.1.0
```
