#!/bin/bash

./../../src/docker/build.sh
docker build --platform linux/amd64 -t flwr_waterlily .
