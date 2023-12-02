#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Cleanup
echo "Cleanup mlcube and clients directory"
rm -rf mlcube

# Clone mlcube
echo "Clone mlcube from GitHub"
git clone https://github.com/mlcommons/mlcube_examples.git
cp -r mlcube_examples/mnist_fl/tensorflow mlcube
rm -rf mlcube_examples
cp dev/{Dockerfile,requirements.txt,mnist.py} mlcube/build

# Download dataset
cd mlcube
mlcube configure -Pdocker.build_strategy=auto
