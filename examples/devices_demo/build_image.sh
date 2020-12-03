#!/bin/bash

if [ -z "${CI}" ]; then
    BUILDKIT=1
else
    BUILDKIT=0
fi

# TODO: should we do a `docker pull` here ? 

DOCKER_BUILDKIT=${BUILDKIT} docker build $@ . -t flower_client:latest

