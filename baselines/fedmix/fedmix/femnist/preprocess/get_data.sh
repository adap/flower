#!/usr/bin/env bash

# assumes that the script is run in the preprocess folder

cd ../data/raw_data
wget https://s3.amazonaws.com/nist-srd/SD19/by_class.zip
wget https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
unzip by_class.zip
rm by_class.zip
unzip by_write.zip
rm by_write.zip
cd ../../preprocess
