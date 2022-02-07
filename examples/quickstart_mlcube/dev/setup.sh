#!/bin/bash

set -e

# Cleanup
echo "Cleanup mlcube and clients directory"
rm -rf mlcube

# Clone mlcube
echo "Clone mlcube from GitHub"
git clone https://github.com/mlcommons/mlcube_examples.git 
cp -r mlcube_examples/mnist_fl/tensorflow mlcube
rm -rf mlcube_examples

# Download dataset
cd mlcube
mlcube configure -Pdocker.build_strategy=auto
