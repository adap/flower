import os
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
        end = start + samples_per_20_sec
        segment = data[start:end]
        # Construct new filename
        new_filename = file_path.replace('.wav', '') + f"_{start//samples_per_20_sec:02d}.wav"
        # Export the segment
        wav.write(new_filename, sample_rate, segment)
        sound_name = new_filename.split("/")[-1]
        # Append the new filename to the text file
        with open(text_file_path, 'a') as f:
            f.write("_silence_/"+sound_name + '\n')

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Check if the file is a WAV file
        file_path = os.path.join(folder_path, filename)
        split_audio(file_path)