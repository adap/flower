#!/bin/bash

# Extract the data_splits.tar file
tar -xvf data_splits.tar

# Create datasets directory and navigate into it
mkdir datasets
cd datasets

# Create speech_commands directory and navigate into it
mkdir speech_commands
cd speech_commands

# Download and extract the training data
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -O train.tar.gz
mkdir -p ./Data/Train
tar -xvzf ./train.tar.gz -C ./Data/Train

# Download and extract the testing data
wget http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz -O test.tar.gz
mkdir -p ./Data/Test
tar -xvzf ./test.tar.gz -C ./Data/Test

# # Rename the _background_noise_ directory to _silence_
mv ./Data/Train/_background_noise_ ./Data/Train/_silence_

cd ..
# Create speech_commands directory and navigate into it
mkdir ambient_context
cd ambient_context

wget http://sensix.tech/datasets/ambientacousticcontext/audioset_1sec_v1.tar.gz -O ambient_data.tar.gz
mkdir -p ./Data
mkdir -p ./temp_data
tar -xvzf ./ambient_data.tar.gz -C ./temp_data
mv ./temp_data/audioset_1sec/* ./Data
rm -r ./temp_data
cd ..
cd ..

# Print completion message
echo "Dataset setup completed."