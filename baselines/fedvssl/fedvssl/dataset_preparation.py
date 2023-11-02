"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

# make sure you have installed unrar package. One can install it using `sudo apt install unrar`.

import json
import subprocess

# Data downloading and preprocessing
# ----------------------------------

# first download the raw videos from the official website

subprocess.run(["mkdir -p data/ucf101/"], shell=True)
subprocess.run(
    [
        "wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar \
        -O data/ucf101/UCF101.rar \
        --no-check-certificate"
    ],
    shell=True,
)

print("---Unzipping the compressed file---")
subprocess.run(["unrar e data/ucf101/UCF101.rar data/ucf101/UCF101_raw/"], shell=True)

print("---Downloading the train/test split---")
subprocess.run(
    [
        "wget \
        https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip \
        -O data/ucf101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate"
    ],
    shell=True,
)

subprocess.run(
    ["unzip data/ucf101/UCF101TrainTestSplits-RecognitionTask.zip -d data/ucf101/"],
    shell=True,
)

print("--Pre-processing the dataset script---")
subprocess.run(
    [
        "python CtP/scripts/process_ucf101.py --raw_dir data/ucf101/UCF101_raw/ \
--ann_dir data/ucf101/ucfTrainTestlist/ --out_dir data/ucf101/"
    ],
    shell=True,
)


# We use the .json files for the annotations.
# One can convert the train_split_1.txt to train_split_1.json
# by using the following code:

ann_path = [
    "data/ucf101/annotations/train_split_1.txt",
    "data/ucf101/annotations/test_split_1.txt",
]
out_path = [
    "data/ucf101/annotations/train_split_1.json",
    "data/ucf101/annotations/test_split_1.json",
]

assert len(ann_path) == len(out_path)

for i in range(len(ann_path)):
    with open(ann_path[i], "r") as f:
        lines = f.read().splitlines()
    anns = []
    for line in lines:
        if line.strip() == "":
            continue
        name, label = line.split(" ")
        anns.append({"name": name, "label": int(label)})  # +1))
    with open(out_path[i], "w") as f:
        json.dump(anns, f, indent=2)


# optional
# ----------
# rm data/ucf101/UCF101.rar
# rm -r data/ucf101/UCF101_raw/


# Data partitioning for federated learning
# ---------------------------------------
# We provide `data_partitioning_ucf.py`
# to generate non-iid data from UCF-101 dataset.
# The above scripts will generate the client_x.json file,
# where "x" denotes the client number.
# To perform partitioning on UCF-101:

subprocess.run(
    [
        "python data_partitioning_ucf.py --json_path data/ucf101/annotations \
--output_path data/ucf101/annotations/client_distribution/ \
--num_clients 5"
    ],
    shell=True,
)
