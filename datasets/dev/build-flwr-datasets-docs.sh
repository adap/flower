#!/bin/bash
# Generating the docs, rename and move the files such that the meet the convention used in Flower.
# Note that it involves two runs of sphinx-build that are necessary.
# The first run generates the .rst files (and the html files that are discarded)
# The second time it is run after the files are renamed and moved to the correct place. It generates the final htmls.

set -e

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )"  >/dev/null 2>&1 && pwd )"/../doc
echo $(pwd)
# Remove the old docs from source/ref-api
REF_API_DIR="source/ref-api"
if [[ -d "$REF_API_DIR" ]]; then
  echo "Removing ${REF_API_DIR}"
  rm -r ${REF_API_DIR}
fi

# Remove the old html files
if [[ -d build ]]; then
  echo "Removing ./build"
  rm -r build
fi

# 1. Basic docs generation: Generate new rst files and html files
echo "Generating the first iteration of docs."
sphinx-build -M html source build

# 2. Add package level access of code following the strategy patter => partitioners.

# Clean up
# Remove the temp_build = html files
echo "Removing ./temp_build"
if [[ -d temp_build ]]; then
  rm -r temp_build
fi
# Remove the temp source_files
echo "Removing ./temp_source"
if [[ -d temp_source ]]; then
  rm -r temp_source
fi

# Correct rst files created from the code following strategy pattern.
# Currently: e.g. flwr_datasets.partitioner.iid_partitioner.IidPartitioner.rst
# But we don't want the "iid_partitioner" part. Instead an access from "partitioner" directly is desired.

# Fix the creation of the documentation for the files following the strategy pattern
# This needs to be specified for each script
STRATEGY_PATTERN_CODE_CREATION='
.. autosummary::
   :toctree: api-ref-strategy-pattern
   :template: autosummary/module.rst
   :recursive:

      flwr_datasets.partitioner
'
TO_KEEP="flwr_datasets".rst
# Create a temporary directory to create needed rst files for better access
TEMP_DIR="./temp_source/_templates/autosummary"
echo "Removing ${TEMP_DIR}"
if [[ ! -d "${TEMP_DIR}" ]]; then
  mkdir -p "${TEMP_DIR}"
fi
# Copy the templates
echo "Coping /source/_templates/autosummary/ to ./temp_source/_templates/autosummary"
cp -r ./source/_templates/autosummary/ ./temp_source/_templates/autosummary
# Copy the configuration
echo "Coping ./source/conf.py to ./temp_source/conf.py"
cp ./source/conf.py ./temp_source/conf.py
# Create the file that will instruct the new rst files creation
touch ./temp_source/index.rst
echo "${STRATEGY_PATTERN_CODE_CREATION}" > ./temp_source/index.rst
sphinx-build -M html temp_source temp_build -D autosummary_ignore_module_all=0

# Move the created files

# Define the first and second directory paths
first_directory=temp_source/api-ref-strategy-pattern
second_directory=./../../source/ref-api

# Go to the first directory and get the list of files
cd "$first_directory"
files=$(ls)

# Create an array to hold the unique endings
endings=()

# Extract the endings from the first directory's filenames
shopt -u nocasematch
for file in $files; do
  ending=$(echo "$file" | rev | cut -d. -f1-2 | rev)
  if [[ ! " ${endings[*]}" =~ ^$ending ]]; then
    endings+=("$ending") # Add the ending if it's not already in the array
  fi
done
echo "All endings"
echo "${endings[@]}"

# Go to the second directory
cd "$second_directory"
echo $(pwd)
# Loop through the endings and remove matching files in the second directory
for ending in "${endings[@]}"; do
  echo "ending"
  echo $ending
  TO_DELETE=($(ls *.${ending}))
  echo to_deleta
  echo "${TO_DELETE}"
  echo "${#TO_DELETE[@]}"
  TO_DELETE_LENGTH="${#TO_DELETE[@]}"
  if [[ "${TO_DELETE_LENGTH}" -eq 1 ]];then
    # Remove
    SECOND_FILE_TO_REMOVE="$( echo ${TO_DELETE[0]} | rev | cut -d. -f3- | rev).rst"
    echo "Would be removed"
      if [[ "${TO_DELETE[0]}" != "${TO_KEEP}" ]];then
        echo "${TO_DELETE[0]}"
        rm "${TO_DELETE[0]}"
      fi
      if [[ "${SECOND_FILE_TO_REMOVE}" != "${TO_KEEP}" ]];then
        echo "${SECOND_FILE_TO_REMOVE}"
        rm "${SECOND_FILE_TO_REMOVE}"
      fi
#  elif [[ "${TO_DELETE_LENGTH}" -eq 2 ]];then
#    #e.g. partitioner.rst and partitioner.partitioner.rst
#    echo "Would be removed"
#    echo "${TO_DELETE[@]}"
#    rm "${TO_DELETE[@]}"

  fi
done
#
# Print out remaining files for confirmation
echo "Remaining files in $second_directory:"
ls
pwd
echo "files"
for file in $files; do
  mv ./../../temp_source/api-ref-strategy-pattern/$file ./
done
cd ./../..
sphinx-build -M html source build -D autosummary_generate=0

