#!/bin/sh

mkdir -p data
git clone https://github.com/tuanct1997/Federated-Learning-ASR-based-on-wav2vec-2.0.git _temp && mv _temp/data/* data/ && rm -rf _temp
python dataset/data_prepare.py

