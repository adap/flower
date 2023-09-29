'''
splits data into train and test sets
'''

import argparse
import json
import os
import random
import time
import sys

from collections import OrderedDict

from constants import DATASETS, SEED_FILES

def create_jsons_for(user_files, which_set, max_users, include_hierarchy):
    """used in split-by-user case"""
    user_count = 0
    json_index = 0
    users = []
    num_samples = []
    user_data = {}
    prev_dir = None
    for (i, t) in enumerate(user_files):
        (u, ns, f) = t

        file_dir = os.path.join(subdir, f)
        if prev_dir != file_dir:
            with open(file_dir, "r") as inf:
                data = json.load(inf)
            prev_dir = file_dir

        users.append(u)
        num_samples.append(ns)
        user_data[u] = data['user_data'][u]
        user_count += 1

    if (user_count == max_users) or (i == len(user_files) - 1):

        all_data = {}
        all_data['users'] = users
        all_data['num_samples'] = num_samples
        all_data['user_data'] = user_data

        data_i = f.find('data')
        num_i = data_i + 5
        num_to_end = f[num_i:]
        param_i = num_to_end.find('_')
        param_to_end = '.json'
        if param_i != -1:
            param_to_end = num_to_end[param_i:]
        nf = '%s_%d%s' % (f[:(num_i-1)], json_index, param_to_end)
        file_name = '%s_%s_%s.json' % ((nf[:-5]), which_set, arg_label)
        ouf_dir = os.path.join(dir, which_set, file_name)

        print('writing %s' % file_name)
        with open(ouf_dir, 'w') as outfile:
            json.dump(all_data, outfile)

        user_count = 0
        json_index += 1
        users = []
        num_samples = []
        user_data = {}

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                help='name of dataset to parse; default: sent140;',
                type=str,
                choices=DATASETS,
                default='sent140')
parser.add_argument('--by_user',
                help='divide users into training and test set groups;',
                dest='user', action='store_true')
parser.add_argument('--by_sample',
                help="divide each user's samples into training and test set groups;",
                dest='user', action='store_false')
parser.add_argument('--frac',
                help='fraction in training set; default: 0.9;',
                type=float,
                default=0.9)
parser.add_argument('--seed',
                help='seed for random partitioning of test/train data',
                type=int,
                default=None)

parser.set_defaults(user=False)

args = parser.parse_args()

print('------------------------------')
print('generating training and test sets')

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dir = os.path.join(parent_path, args.name, 'data')
subdir = os.path.join(dir, 'rem_user_data')
files = []
if os.path.exists(subdir):
    files = os.listdir(subdir)
if len(files) == 0:
    subdir = os.path.join(dir, 'sampled_data')
    if os.path.exists(subdir):
        files = os.listdir(subdir)
if len(files) == 0:
    subdir = os.path.join(dir, 'all_data')
    files = os.listdir(subdir)
files = [f for f in files if f.endswith('.json')]

rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
rng = random.Random(rng_seed)
if os.environ.get('LEAF_DATA_META_DIR') is not None:
    seed_fname = os.path.join(os.environ.get('LEAF_DATA_META_DIR'), SEED_FILES['split'])
    with open(seed_fname, 'w+') as f:
        f.write("# split_seed used by sampling script - supply as "
                "--spltseed to preprocess.sh or --seed to utils/split_data.py\n")
        f.write(str(rng_seed))
    print ("- random seed written out to {file}".format(file=seed_fname))
else:
    print ("- using random seed '{seed}' for sampling".format(seed=rng_seed))

arg_label = str(args.frac)
arg_label = arg_label[2:]

# check if data contains information on hierarchies
file_dir = os.path.join(subdir, files[0])
with open(file_dir, 'r') as inf:
    data = json.load(inf)
include_hierarchy = 'hierarchies' in data

if (args.user):
    print('splitting data by user')

    # 1 pass through all the json files to instantiate arr
    # containing all possible (user, .json file name) tuples
    user_files = []
    for f in files:
        file_dir = os.path.join(subdir, f)
        with open(file_dir, 'r') as inf:
            # Load data into an OrderedDict, to prevent ordering changes
            # and enable reproducibility
            data = json.load(inf, object_pairs_hook=OrderedDict)
            user_files.extend([(u, ns, f) for (u, ns) in
                zip(data['users'], data['num_samples'])])

    # randomly sample from user_files to pick training set users
    num_users = len(user_files)
    num_train_users = int(args.frac * num_users)
    indices = [i for i in range(num_users)]
    train_indices = rng.sample(indices, num_train_users)
    train_blist = [False for i in range(num_users)]
    for i in train_indices:
        train_blist[i] = True
    train_user_files = []
    test_user_files = []
    for i in range(num_users):
        if (train_blist[i]):
            train_user_files.append(user_files[i])
        else:
            test_user_files.append(user_files[i])

    max_users = sys.maxsize
    if args.name == 'femnist':
        max_users = 50 # max number of users per json file
    create_jsons_for(train_user_files, 'train', max_users, include_hierarchy)
    create_jsons_for(test_user_files, 'test', max_users, include_hierarchy)

else:
    print('splitting data by sample')

    for f in files:
        file_dir = os.path.join(subdir, f)
        with open(file_dir, 'r') as inf:
            # Load data into an OrderedDict, to prevent ordering changes
            # and enable reproducibility
            data = json.load(inf, object_pairs_hook=OrderedDict)

        num_samples_train = []
        user_data_train = {}
        num_samples_test = []
        user_data_test = {}

        user_indices = [] # indices of users in data['users'] that are not deleted

        removed = 0
        for i, u in enumerate(data['users']):

            curr_num_samples = len(data['user_data'][u]['y'])
            if curr_num_samples >= 2:
                # ensures number of train and test samples both >= 1
                num_train_samples = max(1, int(args.frac * curr_num_samples))
                if curr_num_samples == 2:
                    num_train_samples = 1

                num_test_samples = curr_num_samples - num_train_samples

                indices = [j for j in range(curr_num_samples)]
                if args.name in ['shakespeare']:
                    train_indices = [i for i in range(num_train_samples)]
                    test_indices = [i for i in range(num_train_samples + 80 - 1, curr_num_samples)]
                else:
                    train_indices = rng.sample(indices, num_train_samples)
                    test_indices = [i for i in range(curr_num_samples) if i not in train_indices]

                if len(train_indices) >= 1 and len(test_indices) >= 1:
                    user_indices.append(i)
                    num_samples_train.append(num_train_samples)
                    num_samples_test.append(num_test_samples)
                    user_data_train[u] = {'x': [], 'y': []}
                    user_data_test[u] = {'x': [], 'y': []}

                    train_blist = [False for _ in range(curr_num_samples)]
                    test_blist = [False for _ in range(curr_num_samples)]

                    for j in train_indices:
                        train_blist[j] = True
                    for j in test_indices:
                        test_blist[j] = True

                    for j in range(curr_num_samples):
                        if (train_blist[j]):
                            user_data_train[u]['x'].append(data['user_data'][u]['x'][j])
                            user_data_train[u]['y'].append(data['user_data'][u]['y'][j])
                        elif (test_blist[j]):
                            user_data_test[u]['x'].append(data['user_data'][u]['x'][j])
                            user_data_test[u]['y'].append(data['user_data'][u]['y'][j])

        users = [data['users'][i] for i in user_indices]

        all_data_train = {}
        all_data_train['users'] = users
        all_data_train['num_samples'] = num_samples_train
        all_data_train['user_data'] = user_data_train

        all_data_test = {}
        all_data_test['users'] = users
        all_data_test['num_samples'] = num_samples_test
        all_data_test['user_data'] = user_data_test
        file_name_train = '%s_train_%s.json' % ((f[:-5]), arg_label)
        file_name_test = '%s_test_%s.json' % ((f[:-5]), arg_label)
        ouf_dir_train = os.path.join(dir, 'train', file_name_train)
        ouf_dir_test = os.path.join(dir, 'test', file_name_test)
        print('writing %s' % file_name_train)
        with open(ouf_dir_train, 'w') as outfile:
            json.dump(all_data_train, outfile)
        print('writing %s' % file_name_test)
        with open(ouf_dir_test, 'w') as outfile:
            json.dump(all_data_test, outfile)
