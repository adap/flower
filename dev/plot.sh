#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Generate plots
python -m flwr_experimental.baseline.tf_fashion_mnist.gen_plots
python -m flwr_experimental.baseline.tf_fashion_mnist.fn_plots
