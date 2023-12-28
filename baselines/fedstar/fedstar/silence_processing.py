import os
import random
import scipy.io.wavfile as wav

# Folder path where your audio files are stored
folder_path = "datasets/speech_commands/Data/Train/_silence_"
# Path of the text file where you want to append the filenames
text_file_path = "data_splits/speech_commands/train_split.txt"

# Function to split audio
def split_audio(file_path):
    # Load the audio file
    sample_rate, data = wav.read(file_path)

    # Calculate the number of samples for 20 seconds
    samples_per_20_sec = 20 * sample_rate

    # Split audio into 20-second segments
    for start in range(0, len(data), samples_per_20_sec):
        print(start)
        end = start + samples_per_20_sec
        segment = data[start:end]
        # Construct new filename
        new_filename = file_path.replace('.wav', '') + f"_{start//samples_per_20_sec:02d}.wav"
        # Export the segment
        wav.write(new_filename, sample_rate, segment)
        print(filename)

        sound_name = new_filename.split("/")[-1]
        # Append the new filename to the text file
        with open(text_file_path, 'a') as f:
            f.write("_silence_/"+sound_name + '\n')
            print("_silence_/"+sound_name + '\n')


def split_audio_v2(file_path, samples_per_clip=1000):

    # Load the audio file
    sample_rate, data = wav.read(file_path)

    # extract `samples_per_clip` 1-second long clips at random
    for i in range(samples_per_clip):

        start = random.randint(0, len(data)-sample_rate)
        segment = data[start: start+sample_rate]
        new_filename = file_path.replace('.wav', '') + f"_{i:05d}.wav"

        wav.write(new_filename, sample_rate, segment)
        sound_name = new_filename.split("/")[-1]
        # Append the new filename to the text file
        with open(text_file_path, 'a') as f:
            f.write(F"\n_silence_/{sound_name}")



# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Check if the file is a WAV file
        file_path = os.path.join(folder_path, filename)
        split_audio_v2(file_path)