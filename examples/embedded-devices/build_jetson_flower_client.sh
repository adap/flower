#!/bin/bash

if [ -z "${CI}" ]; then
    BUILDKIT=1
else
    BUILDKIT=0
fi

# This script build a docker image that's ready to run your flower client.
# Depending on your choice of ML framework (TF or PyTorch), the appropiate
# base image from NVIDIA will be pulled. This ensures you get the best
# performance out of your Jetson device.

BASE_PYTORCH=nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3
BASE_TF=nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3
EXTRA=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--pytorch)
      BASE_IMAGE=$BASE_PYTORCH
      shift
      ;;
    -t|--tensorflow)
      BASE_IMAGE=$BASE_TF
      shift
      ;;
    -r|--no-cache)
      EXTRA="--no-cache"
      shift
      ;;
    -*|--*)
      echo "Unknown option $1 (pass either --pytorch or --tensorflow)"
      exit 1
      ;;
  esac
done

DOCKER_BUILDKIT=${BUILDKIT} docker build $EXTRA \
                                        --build-arg BASE_IMAGE=$BASE_IMAGE \
                                        . \
                                        -t flower_client:latest
