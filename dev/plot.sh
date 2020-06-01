#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Generate plots
python -m flower_benchmark.tf_fashion_mnist.gen_plots
