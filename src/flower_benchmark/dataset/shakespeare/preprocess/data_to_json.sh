#!/usr/bin/env bash

if [ ! -d "../data" ]; then
    mkdir ../data
fi

if [ ! -d "../data/raw_data" ]; then
    mkdir ../data/raw_data
fi

if [ ! -f ../data/raw_data/raw_data.txt ]; then
    ./get_data.sh
fi

if [ ! -d "../data/raw_data/by_play_and_character" ]; then
    echo "dividing txt data between users"
    python3 preprocess_shakespeare.py ../data/raw_data/raw_data.txt ../data/raw_data/
fi

RAWTAG=""
if [[ $@ = *"--raw"* ]]; then
  RAWTAG="--raw"
fi
if [ ! -d "../data/all_data" ]; then
    mkdir ../data/all_data
fi
if [ ! "$(ls -A ../data/all_data)" ]; then
    echo "generating all_data.json"
    python3 gen_all_data.py $RAWTAG
fi