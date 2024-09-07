from multiprocessing import Pool

from whisper_example.task import load_data

TOTAL_TRAIN_PARTITIONS = 2112

def process_one_partition(partition_id: int):
    _ = load_data(
        partition_id,
        save_partition_to_disk=True,
        partitions_save_path="processed_partitions",
    )


if __name__ == "__main__":

    # Parallelize the processing of each partition in the dataset
    # One process will be created per CPU in your system
    with Pool() as pool:
        pool.map(process_one_partition, range(TOTAL_TRAIN_PARTITIONS))
