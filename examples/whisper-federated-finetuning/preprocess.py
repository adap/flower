from multiprocessing import Pool
from time import time

from whisper_example.task import load_data

from datasets import load_dataset

TOTAL_TRAIN_PARTITIONS = 2112


def process_one_partition(partition_id: int):
    _ = load_data(
        partition_id,
    )


if __name__ == "__main__":

    # Download train set
    _ = load_dataset("speech_commands", "v0.02", split="train", token=False)

    # Parallelize the processing of each partition in the dataset
    t_start = time()
    num_proc = None  # set it if you want to limit the number of processes
    with Pool(num_proc) as pool:
        pool.map(process_one_partition, range(TOTAL_TRAIN_PARTITIONS))
    print(
        f"Pre-processing {TOTAL_TRAIN_PARTITIONS} partitions took: {time() - t_start:.2f} s"
    )
