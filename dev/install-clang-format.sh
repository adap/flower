#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Install Clang-Format
# sudo apt install clang-format

if clang-format --version | grep -q 'clang-format version 10.0.0-4ubuntu1'; 
then
   echo "Clang-Format is already installed on your Environment"
else
    sudo apt install clang-format
fi
