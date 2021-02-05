#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Check if Clang-Format is already installed
if ! command -v clang-format-10 &> /dev/null
then
    # Install Clang-Format
    sudo apt install clang-format-10
    exit
else
    echo "clang-format is already installed in your environment"
fi
