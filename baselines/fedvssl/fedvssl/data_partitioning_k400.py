"""Data partitioning script for K-400 dataset."""

import argparse
import json
import os
import random


def parse_args():
    """Parse argument to the main function."""
    parser = argparse.ArgumentParser(
        description="Build non-iid partition Kinetics-400 dataset"
    )
    parser.add_argument(
        "--out_path", default="./non_iid/", type=str, help="output directory path."
    )
    parser.add_argument(
        "--input_path",
        default="/path/to/Kinetics-processed/annotations/train_in_official.txt",
        type=str,
        help="input index file",
    )
    args = parser.parse_args()

    return args


# pylint: disable=too-many-locals
def main():
    """Define the main function for data partitioning."""
    args = parse_args()
    out_path = args.out_path
    input_path = args.input_path

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # initialise label_name_dict
    label_name_dict = {}
    for i in range(1, 401):
        label_name_dict[i] = []

    # generate label_name_dict.
    # key is label index while value is list of sample names.
    with open(input_path, "r") as f_r:
        for line in f_r.readlines():
            cls_ind = int(line.split(" ")[1])
            name = line.split(" ")[0]
            label_name_dict[cls_ind].append(name)

    # generate a list where elements are tuple (label_ind, num_samples)
    label_len_list = []
    for i in range(1, 401):
        length = len(label_name_dict[i])
        label_len_list.append((i, length))

    # order classes by number of samples
    label_len_list.sort(key=lambda x: x[1])

    for i in range(50):
        cid1 = i + 1
        cid2 = i + 51
        client_list1 = []
        client_list2 = []
        print("partition: ", i)

        # select the first and last 4 classes seperately.
        partition_1 = label_len_list[i : i + 4]
        partition_2 = label_len_list[-i - 5 : -i - 1]

        for j in range(4):
            lab1, length1 = partition_1[j]
            lab2, length2 = partition_2[j]

            # randomly shuffle samples
            random.seed(1234)
            random.shuffle(label_name_dict[lab1])
            random.seed(4321)
            random.shuffle(label_name_dict[lab2])

            # halve samples for this class.
            first_half1 = label_name_dict[lab1][: length1 // 2]
            second_half1 = label_name_dict[lab1][length1 // 2 :]

            first_half2 = label_name_dict[lab2][: length2 // 2]
            second_half2 = label_name_dict[lab2][length2 // 2 :]

            # adding elements for the first client.
            for ele in first_half1:
                client_list1.append({"name": ele, "label": int(lab1)})
            for ele in first_half2:
                client_list1.append({"name": ele, "label": int(lab2)})

            # adding elements for the second client.
            for ele in second_half1:
                client_list2.append({"name": ele, "label": int(lab1)})
            for ele in second_half2:
                client_list2.append({"name": ele, "label": int(lab2)})

        # generate json file for each client.
        with open(
            os.path.join(out_path, "client_dist{}.json".format(cid1)), "w"
        ) as f_w:
            json.dump(client_list1, f_w, indent=2)

        with open(
            os.path.join(out_path, "client_dist{}.json".format(cid2)), "w"
        ) as f_w:
            json.dump(client_list2, f_w, indent=2)


if __name__ == "__main__":
    main()
