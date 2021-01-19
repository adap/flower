#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Check if Clang-Format is already installed
if clang-format-10 --version | grep -q 'clang-format version 10.0.0-4ubuntu1'; 
then
   echo "Clang-Format is already installed on your Environment"
else
    # Install Clang-Format if it is not installed already
    sudo apt install clang-format-10
fi
