#!/bin/bash

set -e

# Cleanup
echo "Cleanup mlcube and clients directory"
sudo rm -rf mlcube clients

# Clone mlcube
echo "Clone mlcube from GitHub"
git clone https://github.com/danieljanes/mlcube_examples.git -b rename-fl-example mlcube_examples
cp -r mlcube_examples/mnist_fl/tensorflow mlcube
rm -rf mlcube_examples

#git clone -b initial_checkpoint https://github.com/msheller/mlcube_examples.git && \
#cp -r mlcube_examples/mnist_openfl mlcube && \
#rm -rf mlcube_examples

# Build mlcube
pushd mlcube
poetry run mlcube_docker configure --mlcube=. 
mlcube run --task download
popd
