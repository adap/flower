# Copyright 2021 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Delete previous partitions if needed then extract the entire dataset
echo 'Deleting previous dataset split.'
cd ${LEAF_ROOT}/data/femnist
if [ -d ${LEAF_ROOT}/data/femnist/data ]; then rm -rf ${LEAF_ROOT}/data/femnist/data ; fi 
echo 'Creating new LEAF dataset split.'
./preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.8 

# Format for Flower experiments. Val/train fraction set to 0.25 so validation/total=0.20
cd ${FLOWER_ROOT}/baselines/flwr_baselines/scripts/leaf/femnist
python split_json_data.py \
--save_root ${SAVE_ROOT}/femnist \
--leaf_train_jsons_root ${LEAF_ROOT}/data/femnist/data/train \
--leaf_test_jsons_root ${LEAF_ROOT}/data/femnist/data/test \
--val_frac 0.25 
echo 'Done'
