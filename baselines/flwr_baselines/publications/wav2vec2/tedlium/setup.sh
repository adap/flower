#!/bin/sh

mkdir -p data

# Here we are downloading the client files and the associated CSV files
# that described which part of the data each client holds.
# We download this from another repo in order to not upload the full 
# file tree to the Flower repo.
git clone https://github.com/tuanct1997/Federated-Learning-ASR-based-on-wav2vec-2.0.git _temp && mv _temp/data/* data/ && rm -rf _temp

python preprocessing/data_prepare.py

