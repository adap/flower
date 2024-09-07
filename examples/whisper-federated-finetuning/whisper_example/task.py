"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

import random
from typing import List

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import WeightedRandomSampler
from transformers import WhisperProcessor

from datasets import Dataset, concatenate_datasets

fds = None  # Cache FederatedDataset
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")


def load_data(
    partition_id: int,
    remove_cols: List[str],
):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = NaturalIdPartitioner(partition_by="speaker_id")
        fds = FederatedDataset(
            dataset="speech_commands",
            subset="v0.02",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)

    encoding_fn = get_encoding_fn(processor)

    remove_cols = remove_cols.split(",")
    partition = partition.map(encoding_fn, num_proc=2, remove_columns=remove_cols)

    # Now let's add some _silence_ training examples (add 10% of total examples in this client's data)
    partitioner = fds.partitioners["train"]
    ratio_silences_for_client = 0.1 * (len(partition) / len(partitioner.dataset))
    silence_dataset = prepare_silences_dataset(
        partitioner.dataset, ratio_silences_for_client
    )
    if len(silence_dataset) > 0:
        silence_enc = silence_dataset.map(encoding_fn)
        partition = concatenate_datasets([partition, silence_enc])

    return partition


def get_encoding_fn(processor):
    """Return a function to use to pre-process/encode the SpeechCommands dataset.

    We are working with the 12classes version of this dataset, therefore we need to do
    some reassignment of labels.
    """

    def prepare_dataset(batch):
        audio = batch["audio"]
        data = {}
        data["data"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features

        # All unknown keywords are assigned label 11. The silence clips get assigned label 10
        # In this way we have 12 classes with labels 0-11
        data["targets"] = (
            11
            if batch["is_unknown"]
            else (10 if batch["label"] == 35 else batch["label"])
        )
        return data

    return prepare_dataset


def prepare_silences_dataset(train_dataset, ratio_silence: float = 0.1) -> Dataset:
    """Generate silences for the train set.

    One of the classes in the SpeechCommands datatset is `silence`. However, the dataset
    does not include clips of silence. It does however include 5 long files with
    different background sounds. The taks of this function is to extract several
    (defined by `ratio_silence`) one-second long clips from those background audio
    files. Later, those audio clips will be included into the training set.
    """
    # Retrieve original silence audio clips
    silences = train_dataset.filter(lambda x: x["label"] == 35)
    # Figure out how many to add
    num_silence_total = int(len(train_dataset) * ratio_silence)
    # Num new entries per background noise clip
    num_silence_per_bkg = num_silence_total // len(silences)

    silence_to_add = []
    for sil in silences:
        sil_array = sil["audio"]["array"]
        sr = sil["audio"]["sampling_rate"]
        # print(f"Extracting audio from: {sil['file']} ...")
        for _ in range(num_silence_per_bkg):
            random_offset = random.randint(0, len(sil_array) - sr - 1)
            sil_array_crop = sil_array[random_offset : random_offset + sr]

            entry = sil
            silence_to_add.append(entry)
            silence_to_add[-1]["audio"]["array"] = sil_array_crop

    return Dataset.from_list(silence_to_add)


def construct_balanced_sampler(trainset):
    hist, _ = np.histogram(trainset["targets"], bins=12)
    # Mask of non-zeros
    hist_mask = hist > 0
    w_per_class = len(trainset) / (
        hist + 1
    )  # avoid dividing by zeros  # doesn't have to add up to 1 (relative is what matters)
    w_per_class += 1  # needed in case trainset has very few samples
    # Apply mask so we don't attempt sampling classes that aren't present
    w_per_class *= hist_mask
    w_ss = [w_per_class[t] for t in trainset["targets"]]
    return WeightedRandomSampler(w_ss, len(w_ss))
