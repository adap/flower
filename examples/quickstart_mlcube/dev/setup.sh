#!/bin/bash

set -e

# Cleanup
echo "Cleanup mlcube and clients directory"
sudo rm -rf mlcube clients

# Clone mlcube
echo "Clone mlcube from GitHub"
git clone -b initial_checkpoint https://github.com/msheller/mlcube_examples.git && \
cp -r mlcube_examples/mnist_openfl mlcube && \
rm -rf mlcube_examples

# Build mlcube
pushd mlcube
poetry run mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml
popd
