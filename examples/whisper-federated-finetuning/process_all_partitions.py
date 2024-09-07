from multiprocessing import Pool

from whisper_example.task import load_data

from datasets import load_dataset

TOTAL_TRAIN_PARTITIONS = 2112


def process_one_partition(partition_id: int):
    _ = load_data(
        partition_id,
        save_partition_to_disk=True,
        partitions_save_path="processed_partitions",
    )


if __name__ == "__main__":

    # Download train set
    _ = load_dataset("speech_commands", "v0.02", split="train", token=False)

    # Parallelize the processing of each partition in the dataset
    num_proc = 8
    with Pool(num_proc) as pool:
        pool.map(process_one_partition, range(TOTAL_TRAIN_PARTITIONS))
