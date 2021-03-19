# Clone LEAF repository
git clone https://github.com/TalwalkarLab/leaf ${LEAF_ROOT}
# Process Shakespeare for non-iid, 20% testset, minimum two lines
cd ${LEAF_ROOT}/data/shakespeare
rm -rf data/rem_user_data data/sampled_data data/test data/train
./preprocess.sh -s niid --sf 1.0 -k 2 -t sample -tf 0.8

# Format for Flower experiments. Fraction set so validation/total=20%
cd ${FLOWER_ROOT}/baselines/datasets/shakespeare
python split_json_data.py \
--save_root ${HOME}/datasets/partitions/shakespeare \
--leaf_train_json ${LEAF_ROOT}/data/shakespeare/data/train/all_data_niid_0_keep_2_train_9.json \
--val_frac 0.25 \
--leaf_test_json ${LEAF_ROOT}/data/shakespeare/data/test/all_data_niid_0_keep_2_test_9.json