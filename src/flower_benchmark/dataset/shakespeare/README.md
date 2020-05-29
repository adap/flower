Shakespeare Dataset

Follow the preposseing pipline from LEAF project: https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare .
Need to run ./preprocess/get_data.sh to downlowd the dataset from: http://www.gutenberg.org/files/100/old/1994-01-100.zip .

Then following the settup instruction from LEAF project:
e.g. 
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 

Make sure to delete the rem_user_data, sampled_data, test, and train subfolders in the data directory before re-running preprocess.sh