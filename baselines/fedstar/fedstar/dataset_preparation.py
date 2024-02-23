"""Generate 1-second audio clip from background sounds in speechcommands."""

import os
import random

import scipy.io.wavfile as wav
from tqdm import tqdm

# Folder path where your audio files are stored
FOLDER_PATH = "datasets/speech_commands/Data/Train/_silence_"
# Path of the text file where you want to append the filenames
TEXT_FILE_PATH = "data_splits/speech_commands/train_split.txt"


def split_audio_v2(read_file_path, samples_per_clip=1000):
    """Split an audio file into a specified number of clips.

    Parameters
    ----------
    read_file_path : str
        The file path of the audio file to be split.
    samples_per_clip : int, optional
        The number of one-second clips to extract from the audio file. Defaults to 1000.
    """
    sample_rate, data = wav.read(read_file_path)

    # extract `samples_per_clip` 1-second long clips at random
    for i in tqdm(range(samples_per_clip)):
        start = random.randint(0, len(data) - sample_rate)
        segment = data[start : start + sample_rate]
        new_filename = file_path.replace(".wav", "") + f"_{i:05d}.wav"

        wav.write(new_filename, sample_rate, segment)
        sound_name = new_filename.split("/")[-1]
        # Append the new filename to the text file
        with open(TEXT_FILE_PATH, "a", encoding="utf-8") as write_file:
            write_file.write(f"\n_silence_/{sound_name}")


# Iterate over files in the folder
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".wav"):  # Check if the file is a WAV file
        file_path = os.path.join(FOLDER_PATH, filename)
        split_audio_v2(file_path)
