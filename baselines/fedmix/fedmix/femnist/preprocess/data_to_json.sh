#!/usr/bin/env bash

# assumes that the script is run in the preprocess folder

if [ ! -d "../data" ]; then
  mkdir ../data
fi
if [ ! -d "../data/raw_data" ]; then
  echo "------------------------------"
  echo "downloading data"
  mkdir ../data/raw_data
  ./get_data.sh
  echo "finished downloading data"
fi

if [ ! -d "../data/intermediate" ]; then # stores .pkl files during preprocessing
  mkdir ../data/intermediate
fi

if [ ! -f ../data/intermediate/class_file_dirs.pkl ]; then
  echo "------------------------------"
  echo "extracting file directories of images"
  python3 get_file_dirs.py
  echo "finished extracting file directories of images"
fi

if [ ! -f ../data/intermediate/class_file_hashes.pkl ]; then
  echo "------------------------------"
  echo "calculating image hashes"
  python3 get_hashes.py
  echo "finished calculating image hashes"
fi

if [ ! -f ../data/intermediate/write_with_class.pkl ]; then
  echo "------------------------------"
  echo "assigning class labels to write images"
  python3 match_hashes.py
  echo "finished assigning class labels to write images"
fi

if [ ! -f ../data/intermediate/images_by_writer.pkl ]; then
  echo "------------------------------"
  echo "grouping images by writer"
  python3 group_by_writer.py
  echo "finished grouping images by writer"
fi

if [ ! -d "../data/all_data" ]; then
  mkdir ../data/all_data
fi
if [ ! "$(ls -A ../data/all_data)" ]; then
  echo "------------------------------"
  echo "converting data to .json format"
  python3 data_to_json.py
  echo "finished converting data to .json format"
fi
