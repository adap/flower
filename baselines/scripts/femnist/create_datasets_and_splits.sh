# Process Shakespeare for non-iid, 20% testset, minimum two lines
# Delete previous partitions if needed then extract the entire dataset
echo 'Deleting previous dataset split.'
cd ${LEAF_ROOT}/data/femnist
if [ -d ${LEAF_ROOT}/data/rem_user_data ]; then rm -rf ${LEAF_ROOT}/data/rem_user_data ; fi && \
if [ -d ${LEAF_ROOT}/data/sampled_data ]; then rm -rf ${LEAF_ROOT}/data/sampled_data ; fi && \
if [ -d ${LEAF_ROOT}/data/test ]; then rm -rf ${LEAF_ROOT}/data/test ; fi && \
if [ -d ${LEAF_ROOT}/data/train ]; then rm -rf ${LEAF_ROOT}/data/train ; fi && \
echo 'Creating new LEAF dataset split.'
/preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.8

# Format for Flower experiments. Val/train fraction set to 0.25 so validation/total=0.20
cd ${FLOWER_ROOT}/baselines/scripts/shakespeare
python split_json_data.py \
--save_root ${HOME}/datasets/partitions/shakespeare \
--leaf_train_json ${LEAF_ROOT}/data/femnist/data/train/all_data_niid_0_keep_2_train_9.json \
--val_frac 0.25 \
--leaf_test_json ${LEAF_ROOT}/data/femnist/data/test/all_data_niid_0_keep_2_test_9.json
echo 'Done'
