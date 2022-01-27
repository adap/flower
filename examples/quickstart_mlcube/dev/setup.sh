#!/bin/bash

set -e

# Cleanup
echo "Cleanup mlcube and clients directory"
sudo rm -rf mlcube clients

# Clone mlcube
echo "Clone mlcube from GitHub"
git clone https://github.com/mlcommons/mlcube_examples.git 
cp -r mlcube_examples/mnist_fl/tensorflow mlcube
rm -rf mlcube_examples

# Build mlcube
pushd mlcube
poetry run mlcube_docker configure --mlcube=. 
mlcube run --task download
popd
