"""Required imports for silence_processing.py script."""
import os
import random

import scipy.io.wavfile as wav

# Folder path where your audio files are stored
folder_path = "datasets/speech_commands/Data/Train/_silence_"
# Path of the text file where you want to append the filenames
text_file_path = "data_splits/speech_commands/train_split.txt"


def split_audio_v2(file_path, samples_per_clip=1000):
    """Split an audio file into a specified number of clips.

    Parameters
    ----------
    file_path : str
        The file path of the audio file to be split.
    samples_per_clip : int, optional
        The number of one-second clips to extract from the audio file. Defaults to 1000.
    """
    sample_rate, data = wav.read(file_path)

    # extract `samples_per_clip` 1-second long clips at random
    for i in range(samples_per_clip):
        start = random.randint(0, len(data) - sample_rate)
        segment = data[start : start + sample_rate]
        new_filename = file_path.replace(".wav", "") + f"_{i:05d}.wav"

        wav.write(new_filename, sample_rate, segment)
        sound_name = new_filename.split("/")[-1]
        # Append the new filename to the text file
        with open(text_file_path, "a") as f:
            f.write(f"\n_silence_/{sound_name}")


# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Check if the file is a WAV file
        file_path = os.path.join(folder_path, filename)
        split_audio_v2(file_path)
