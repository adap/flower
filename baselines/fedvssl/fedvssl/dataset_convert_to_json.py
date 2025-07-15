"""Dataset pre-processing: convert .txt files to .json files."""

import json

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

for i, _ in enumerate(ann_path):
    with open(ann_path[i], "r") as f:
        lines = f.read().splitlines()
    anns = []
    for line in lines:
        if line.strip() == "":
            continue
        name, label = line.split(" ")
        anns.append({"name": name, "label": int(label)})
    with open(out_path[i], "w") as f:
        json.dump(anns, f, indent=2)
